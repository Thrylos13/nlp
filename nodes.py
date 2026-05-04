"""
NLP2SQL LangGraph Nodes

Defines all nodes for the main SQL generation workflow and sub-workflows.
Each node corresponds to a discrete step in the multi-agent pipeline.
"""

import json
import time
from typing import List, Dict, Any

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from utils.state import (
    AgentState, LabelAgentState, VectorAgentState, SQLAgentState,
    QueryRefinementResponse,
)
from utils.config_loader import config
from app.observability.logging import logger
from utils.memory import HybridMemoryManager
import mlflow

memory_manager = HybridMemoryManager()


# ------------------------------------------------------------------ #
#  Main workflow nodes                                                 #
# ------------------------------------------------------------------ #

@mlflow.trace(name="retrieve_memory")
async def retrieve_conversation_history_node(state: AgentState) -> AgentState:
    """Retrieves conversation history from persistent storage using the session_id."""
    try:
        session_id = state.get("session_id")
        if not session_id:
            logger.info("[Memory Retrieval] No session ID provided, skipping history retrieval.")
            state["conversation_history"] = []
            return state

        logger.info(f"[Memory Retrieval] Retrieving history for session: {session_id}")
        state["conversation_history"] = memory_manager.get_conversation_history(session_id)
        logger.info(f"[Memory Retrieval] Found {len(state['conversation_history'])} previous conversations.")
        return state
    except Exception as e:
        logger.exception(f"[Memory Retrieval] Error: {e}")
        state["conversation_history"] = []
        return state


@mlflow.trace(name="query_refinement")
async def query_refinement_node(state: AgentState) -> AgentState:
    """Refines the user's query based on conversation history."""
    try:
        original_query = state["original_query"]
        if not state.get("conversation_history"):
            logger.info("[Query Refinement] No history found, using original query.")
            state["refined_query"] = original_query
            state["query_required_context"] = False
            state["refinement_tokens"] = {}
            return state

        context_text = "\n---\n".join([
            f"Query: {entry['user_query']}\nRefined: {entry['refined_query']}\n"
            for entry in state["conversation_history"]
        ])
        parser = PydanticOutputParser(pydantic_object=QueryRefinementResponse)
        prompt_template = f"""
You are a query refinement assistant. Your task is to turn the user's current natural language query into a clear, standalone query using only the relevant parts of the conversation history.

OBJECTIVE:
Refine the latest user query to be self-contained and unambiguous, without dragging in unrelated or unnecessary parts of the conversation history.

RULES:
- Do NOT generate SQL. Only output a refined natural language query.
- If the current query is already unambiguous and standalone, return it as-is.
- If the query depends on earlier references (e.g., "those users" or "that incident"), include only the minimal context required to resolve ambiguity.
- Do NOT include unrelated or excessive prior context unless it's necessary for understanding the current query.
- Keep the refined query concise, focused, and natural.
- Always try to refine the query on 1 or 2 lines strictly adding only pieces of context which are relevant to user's query nothing else.

INPUTS:
CONVERSATION HISTORY: {context_text}

CURRENT USER QUERY: {original_query}

STRICTLY FOLLOW THE OUTPUT FORMAT GIVEN BELOW:
Output Format: {parser.get_format_instructions()}
"""
        messages = [HumanMessage(content=prompt_template)]
        response = await state["llm"].ainvoke(messages)

        state["refinement_tokens"] = response.usage_metadata or {}

        try:
            parsed_output = parser.parse(response.content)
            state["refined_query"] = parsed_output.refined_query
            state["query_required_context"] = parsed_output.requires_context
            logger.info(
                f"[Query Refinement] Original: '{original_query}' → "
                f"Refined: '{parsed_output.refined_query}'"
            )
        except Exception as parse_error:
            logger.warning(f"[Query Refinement] Pydantic parsing failed: {parse_error}. Using original query.")
            state["refined_query"] = original_query
            state["query_required_context"] = False

        return state
    except Exception as e:
        logger.exception(f"[Query Refinement] Error: {e}")
        state["refined_query"] = state.get("original_query", "")
        return state


@mlflow.trace(name="label_agent")
async def label_agent_node(state: AgentState) -> AgentState:
    """Invokes the label agent sub-workflow to find tables based on labels."""
    from utils.workflows import label_agent_workflow
    try:
        logger.info(f"[Label Agent] Processing query: '{state['refined_query']}'")
        initial_state = LabelAgentState(query=state["refined_query"], error="", tokens={})
        result = await label_agent_workflow.ainvoke(initial_state)

        if result.get("error"):
            state["error"] = f"Label agent error: {result['error']}"
        else:
            state["label_tables"] = result.get("relevant_tables", [])
            state["label_tokens"] = result.get("tokens", {})
            logger.info(f"[Label Agent] Found tables: {state['label_tables']}")
        return state
    except Exception as e:
        logger.exception(f"[Label Agent] Error: {e}")
        state["error"] = str(e)
        return state


@mlflow.trace(name="vector_agent")
async def vector_agent_node(state: AgentState) -> AgentState:
    """Invokes the vector agent sub-workflow to refine table selection."""
    from utils.workflows import vector_agent_workflow
    try:
        if not state.get("label_tables"):
            logger.warning("[Vector Agent] No initial tables from label agent. Skipping.")
            state["vector_tables"] = []
            return state

        logger.info(f"[Vector Agent] Refining tables for query: '{state['refined_query']}'")
        initial_state = VectorAgentState(
            query=state["original_query"],
            initial_tables=state["label_tables"],
            error="",
            tokens={},
        )
        result = await vector_agent_workflow.ainvoke(initial_state)

        if result.get("error"):
            state["error"] = f"Vector agent error: {result['error']}"
        else:
            state["vector_tables"] = result.get("relevant_tables", [])
            state["vector_tokens"] = result.get("tokens", {})
            logger.info(f"[Vector Agent] Refined tables: {state['vector_tables']}")
        return state
    except Exception as e:
        logger.exception(f"[Vector Agent] Error: {e}")
        state["error"] = str(e)
        return state


@mlflow.trace(name="sql_agent")
async def sql_generate_agent_node(state: AgentState) -> AgentState:
    """Invokes the SQL generation sub-workflow."""
    from utils.workflows import sql_agent_workflow
    try:
        if not state.get("vector_tables"):
            logger.error("[SQL Agent] No tables available for SQL generation.")
            state["error"] = "No tables found to generate SQL query."
            return state

        logger.info(f"[SQL Agent] Generating SQL for query: '{state['refined_query']}'")
        initial_state = SQLAgentState(
            query=state["refined_query"],
            tables=state["vector_tables"],
            conversation_history=state["conversation_history"],
            error="",
            tokens={},
        )
        result = await sql_agent_workflow.ainvoke(initial_state)

        if result.get("error"):
            state["error"] = f"SQL agent error: {result['error']}"
        else:
            state["sql_query"] = result.get("sql_query", "")
            state["sql_tokens"] = result.get("tokens", {})
            logger.info(f"[SQL Agent] Generated SQL: {state['sql_query']}")
        return state
    except Exception as e:
        logger.exception(f"[SQL Agent] Error: {e}")
        state["error"] = str(e)
        return state


@mlflow.trace(name="save_memory")
async def save_conversation_node(state: AgentState) -> AgentState:
    """Saves the current conversation turn to persistent storage."""
    try:
        session_id = state.get("session_id")
        if session_id and state.get("sql_query"):
            logger.info(f"[Memory Save] Saving conversation for session: {session_id}")
            memory_manager.save_conversation(
                session_id=session_id,
                user_id=state["user_id"],
                user_query=state["original_query"],
                refined_query=state["refined_query"],
                generated_sql=state["sql_query"],
                tables_used=state["vector_tables"],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            logger.info("[Memory Save] Skipping save: No session ID or no SQL generated.")
        return state
    except Exception as e:
        logger.exception(f"[Memory Save] Error: {e}")
        return state


# ------------------------------------------------------------------ #
#  MCP tool helper                                                     #
# ------------------------------------------------------------------ #

@mlflow.trace(name="invoke_mcp_tool")
async def invoke_mcp_tool(server_url: str, tool_name: str, tool_input: Dict) -> Any:
    """
    Generic helper to connect to an MCP server and invoke a tool.

    Args:
        server_url (str): URL of the MCP server.
        tool_name (str): Name of the tool to invoke.
        tool_input (dict): Input payload for the tool.

    Returns:
        Any: Raw tool response.

    Raises:
        RuntimeError: If the tool is not found on the server.
    """
    try:
        async with streamablehttp_client(server_url) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                target_tool = next((t for t in tools if t.name == tool_name), None)
                if not target_tool:
                    raise RuntimeError(f"Tool '{tool_name}' not found on server {server_url}")

                return await target_tool.ainvoke(tool_input)
    except Exception as e:
        logger.exception(f"invoke_mcp_tool: Error invoking '{tool_name}' on {server_url}: {e}")
        raise


# ------------------------------------------------------------------ #
#  Sub-workflow nodes                                                  #
# ------------------------------------------------------------------ #

async def get_table_labels_node(state: LabelAgentState) -> LabelAgentState:
    try:
        result = await invoke_mcp_tool(
            config["NLP2SQL_LABEL_SERVER_URL"], "get_table_labels", {}
        )
        state["table_labels"] = json.loads(result) if isinstance(result, str) else result
        return state
    except Exception as e:
        logger.exception(f"[Label Agent - get_table_labels] Error: {e}")
        state["error"] = str(e)
        return state


async def get_unique_labels_node(state: LabelAgentState) -> LabelAgentState:
    if state.get("error"):
        return state
    try:
        result = await invoke_mcp_tool(
            config["NLP2SQL_LABEL_SERVER_URL"], "get_unique_table_labels", {}
        )
        state["unique_labels"] = json.loads(result) if isinstance(result, str) else result
        return state
    except Exception as e:
        logger.exception(f"[Label Agent - get_unique_labels] Error: {e}")
        state["error"] = str(e)
        return state


async def find_relevant_tables_node(state: LabelAgentState) -> LabelAgentState:
    if state.get("error"):
        return state
    try:
        tool_input = {
            "user_query": state["query"],
            "labels_dict": json.dumps(state["table_labels"]),
            "unique_labels": json.dumps(state["unique_labels"]),
        }
        response = await invoke_mcp_tool(
            config["NLP2SQL_LABEL_SERVER_URL"], "find_relevant_tables", tool_input
        )
        response_data = json.loads(response)
        state["relevant_tables"] = response_data.get("tables", [])
        state["tokens"] = response_data.get("tokens", {})
        return state
    except Exception as e:
        logger.exception(f"[Label Agent - find_relevant_tables] Error: {e}")
        state["error"] = str(e)
        return state


async def vector_search_node(state: VectorAgentState) -> VectorAgentState:
    try:
        tool_input = {
            "query": state["query"],
            "table_names": json.dumps(state["initial_tables"]),
        }
        result = await invoke_mcp_tool(
            config["NLP2SQL_VECTOR_SERVER_URL"], "get_relevant_tables", tool_input
        )
        result_data = json.loads(result)
        state["relevant_tables"] = result_data.get("relevant_tables", [])
        state["tokens"] = {}
        return state
    except Exception as e:
        logger.exception(f"[Vector Agent - vector_search] Error: {e}")
        state["error"] = str(e)
        return state


async def prepare_sql_schema_node(state: SQLAgentState) -> SQLAgentState:
    try:
        tool_input = {"table_names": json.dumps(state["tables"])}
        result = await invoke_mcp_tool(
            config["NLP2SQL_SQL_SERVER_URL"], "prepare_table_schema", tool_input
        )
        state["schema_info"] = result
        return state
    except Exception as e:
        logger.exception(f"[SQL Agent - prepare_schema] Error: {e}")
        state["error"] = str(e)
        return state


async def generate_sql_query_node(state: SQLAgentState) -> SQLAgentState:
    if state.get("error"):
        return state
    try:
        context = "\n".join([
            f"Query: {entry['refined_query']} -> SQL: {entry['generated_sql']}"
            for entry in state.get("conversation_history", [])
        ])
        tool_input = {
            "query": state["query"],
            "tables_schema": state["schema_info"],
            "conversation_context": context,
        }
        response = await invoke_mcp_tool(
            config["NLP2SQL_SQL_SERVER_URL"], "generate_sql_query", tool_input
        )
        response_data = json.loads(response)
        state["sql_query"] = response_data.get("generated_sql", "")
        state["tokens"] = response_data.get("tokens", {})
        return state
    except Exception as e:
        logger.exception(f"[SQL Agent - generate_sql] Error: {e}")
        state["error"] = str(e)
        return state
