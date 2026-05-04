"""
NLP2SQL Service

Encapsulates all business logic for the NLP2SQL module:
- SQL generation via the LangGraph multi-agent workflow
- Session management (start, end, memory clear)
- Cache-first lookup before invoking the workflow
- MCP tool invocation for SQL execution and fixing
- Feedback marking in memory
- Audit event delegation to AuditRepository
"""

import json
import time
import uuid
import yaml
import sys
import os

from app.observability.logging import logger
from app.repositories.audit_repository import AuditRepository
from app.exceptions.nlp2sql_exception import (
    NLP2SQLException, PromptFetchException, WorkflowExecutionException, SessionException
)

try:
    with open("configs/nlp2sql_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info("NLP2SQLService: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"NLP2SQLService: Failed to load configuration: {e}")
    raise

if config.get('PROJECT_ROOT') and config['PROJECT_ROOT'] not in sys.path:
    sys.path.append(config['PROJECT_ROOT'])

try:
    from utils.services import initialize_llm
    from utils.workflows import main_sql_generation_graph
    from utils.nodes import invoke_mcp_tool
    from utils.memory import HybridMemoryManager
    from utils.state import AgentState
    from fetch_details_from_database import fetch_prompt
    logger.info("NLP2SQLService: Internal imports loaded successfully.")
except ImportError as e:
    logger.exception(f"NLP2SQLService: Failed to import internal modules: {e}")
    raise


class NLP2SQLService:
    """
    Service class for natural language to SQL generation.

    Delegates DB/queue operations to AuditRepository.
    Owns workflow orchestration, cache lookups, session management,
    and MCP tool invocation.
    """

    def __init__(self):
        try:
            self.audit_repo = AuditRepository()
            self.memory_manager = HybridMemoryManager()
            logger.info("NLP2SQLService: Initialised successfully.")
        except Exception as e:
            logger.error(f"NLP2SQLService: Initialisation failed: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def decode_tool_result(self, raw_result, tool_name: str):
        """
        Normalises MCP tool responses to a Python object.
        Tools may return either a JSON string or a dict/list.

        Args:
            raw_result: Raw value returned by the MCP tool.
            tool_name (str): Tool name for error context.

        Returns:
            dict | list: Parsed Python object.

        Raises:
            ValueError: If result cannot be decoded.
        """
        try:
            if isinstance(raw_result, (dict, list)):
                return raw_result
            if isinstance(raw_result, str):
                return json.loads(raw_result)
            raise ValueError(
                f"Unsupported response type from tool '{tool_name}': {type(raw_result).__name__}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned by tool '{tool_name}': {raw_result}") from e

    def extract_exception_message(self, e: BaseException) -> str:
        """Recursively extracts messages from exception groups."""
        if isinstance(e, ExceptionGroup):
            return " | ".join(self.extract_exception_message(sub) for sub in e.exceptions)
        return str(e)

    # ------------------------------------------------------------------ #
    #  Session management                                                  #
    # ------------------------------------------------------------------ #

    async def start_session(self, user_id: int) -> str:
        """
        Creates and returns a new unique session ID.

        Args:
            user_id (int): Requesting user identifier.

        Returns:
            str: New session UUID.
        """
        try:
            session_id = str(uuid.uuid4())
            logger.info(f"NLP2SQLService: New session started for user {user_id}: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"NLP2SQLService: Failed to start session for user {user_id}: {e}")
            raise SessionException("Unable to initialize a session.", "NLP2SQL0001")

    async def end_session(self, user_id: int, session_id: str):
        """
        Clears in-memory cache for the given session.

        Args:
            user_id (int): Requesting user identifier.
            session_id (str): Session to end.

        Raises:
            SessionException: If memory clearing fails.
        """
        try:
            self.memory_manager.clear_session_from_ignite(session_id)
            logger.info(f"NLP2SQLService: Session ended for user {user_id}: {session_id}")
        except Exception as e:
            logger.exception(f"NLP2SQLService: Error ending session {session_id}: {e}")
            raise SessionException("Unable to end session.", "NLP2SQL0003")

    # ------------------------------------------------------------------ #
    #  SQL generation                                                      #
    # ------------------------------------------------------------------ #

    async def generate_sql(self, request) -> dict:
        """
        Orchestrates the full NLP → SQL pipeline.

        Steps:
        1. Cache lookup — return immediately on hit.
        2. Audit request logging if queueFlag is set.
        3. LLM initialisation.
        4. Prompt fetching.
        5. LangGraph workflow execution.
        6. SQL extraction from workflow result.
        7. Audit response logging.

        Args:
            request (SQLRequest): Validated Pydantic request object.

        Returns:
            dict: Response payload with 'query', 'refined_query',
                  'tables_used', and 'token_details'.

        Raises:
            NLP2SQLException: For domain-specific failures.
        """
        start_time = time.time()
        req_param = request.requestParams
        query = req_param.get("query")
        session_id = req_param.get("sessionId")

        logger.info(
            f"NLP2SQLService: Received request for session {session_id} with query: '{query}'"
        )

        # ---- Cache lookup ------------------------------------------ #
        if session_id:
            try:
                cached_result = self.memory_manager.get_exact_match(session_id, query)
                if cached_result:
                    total_time = round(time.time() - start_time, 2)
                    logger.info(f"NLP2SQLService: Cache hit in {total_time}s for session {session_id}")
                    return {
                        "query": cached_result.get("query"),
                        "token_details": {"cached": True},
                        "refined_query": cached_result.get("refined_query"),
                        "cached": True,
                    }
            except Exception as e:
                logger.warning(f"NLP2SQLService: Cache lookup failed (non-fatal): {e}")

        # ---- Audit request ----------------------------------------- #
        request_id = None
        if request.queueFlag:
            try:
                request_id = self.audit_repo.generate_request_id()
                self.audit_repo.log_event('log_request_mst', {
                    'requestId': request_id, 'userId': request.userId,
                    'appId': request.appId, 'aiModelId': request.aiModelId,
                    'os': request.os, 'browserClientId': request.browserClientId,
                })
                self.audit_repo.log_event('log_request_params', {
                    'requestId': request_id, 'query': query,
                    'target_database': config["NLP2SQL_TARGET_DB"],
                })
            except Exception as e:
                logger.error(f"NLP2SQLService: Audit request logging failed (non-fatal): {e}")

        # ---- LLM initialisation ------------------------------------ #
        try:
            llm = initialize_llm(request.aiModelId, request.moduleDetails)
        except Exception as e:
            logger.error(f"NLP2SQLService: LLM initialisation failed: {e}")
            raise NLP2SQLException("Failed to initialise LLM.", "NLP2SQL0005")

        # ---- Prompt fetching --------------------------------------- #
        try:
            prompts = {
                "label": fetch_prompt(request.userId, request.appId, request.aiModelId, "nlp_sql_label_agent"),
                "vector": fetch_prompt(request.userId, request.appId, request.aiModelId, "nlp_sql_vector_search_agent"),
                "sql": fetch_prompt(request.userId, request.appId, request.aiModelId, "nlp_sql_query_agent"),
            }
        except Exception as e:
            logger.error(f"NLP2SQLService: Prompt fetching failed: {e}")
            raise PromptFetchException("Unexpected error occurred while fetching prompts.", "NLP2SQL0005")

        # ---- Workflow execution ------------------------------------- #
        try:
            initial_state = AgentState(
                original_query=query,
                refined_query="",
                user_id=str(request.userId),
                app_id=request.appId,
                model_id=request.aiModelId,
                llm=llm,
                label_tables=[],
                vector_tables=[],
                sql_query="",
                label_tokens={},
                vector_tokens={},
                sql_tokens={},
                refinement_tokens={},
                error="",
                prompts=prompts,
                conversation_history=[],
                session_id=session_id,
                query_required_context=False,
            )
            result = await main_sql_generation_graph.ainvoke(initial_state)

            if result.get("error"):
                logger.error(f"NLP2SQLService: Workflow error: {result['error']}")
                raise WorkflowExecutionException("Error occurred while generating SQL query.", "NLP2SQL0006")

        except (NLP2SQLException, PromptFetchException, WorkflowExecutionException):
            raise
        except Exception as e:
            logger.exception(f"NLP2SQLService: Workflow execution failed: {e}")
            raise WorkflowExecutionException("Error occurred while generating SQL query.", "NLP2SQL0006")

        # ---- SQL extraction ---------------------------------------- #
        generated_sql = result.get("sql_query", "")
        refined_query = result.get("refined_query", "")
        tables_used = result.get("vector_tables", [])

        try:
            if isinstance(generated_sql, str):
                parsed = json.loads(generated_sql)
                if isinstance(parsed, dict) and "query" in parsed:
                    generated_sql = parsed["query"]
            elif isinstance(generated_sql, dict) and "query" in generated_sql:
                generated_sql = generated_sql["query"]
        except (json.JSONDecodeError, TypeError):
            raise NLP2SQLException("Unable to extract SQL query from response.", "NLP2SQL0006")

        total_time = round(time.time() - start_time, 2)
        logger.info(f"NLP2SQLService: SQL generated in {total_time}s for session {session_id}")

        token_details = {
            "refinement": result.get("refinement_tokens"),
            "label_agent": result.get("label_tokens"),
            "vector_agent": result.get("vector_tokens"),
            "sql_agent": result.get("sql_tokens"),
        }

        response_payload = {
            "query": generated_sql,
            "refined_query": refined_query,
            "tables_used": tables_used,
            "token_details": token_details,
            "cached": False,
        }
        logger.info(f"NLP2SQLService: Response payload: {response_payload}")

        # ---- Audit response ---------------------------------------- #
        if request.queueFlag and request_id:
            try:
                response_id = self.audit_repo.generate_response_id()
                self.audit_repo.log_event('log_response_mst', {
                    'responseId': response_id, 'requestId': request_id,
                    'processingTime': total_time, 'status': "success",
                    'errorCode': "NLP2SQL0000", 'errorMessage': None,
                })
                self.audit_repo.log_event('log_response_params', {
                    'responseId': response_id, 'requestId': request_id,
                    'generated_sql': generated_sql,
                    'tokendetails': token_details,
                    'refined_query': refined_query,
                })
            except Exception as e:
                logger.error(f"NLP2SQLService: Audit response logging failed (non-fatal): {e}")

        return response_payload

    # ------------------------------------------------------------------ #
    #  SQL execution                                                       #
    # ------------------------------------------------------------------ #

    async def execute_sql(self, sql_query: str, session_id: str) -> dict:
        """
        Directly executes a SQL query via the SQL MCP server.

        Args:
            sql_query (str): Raw SQL query to execute.
            session_id (str): Session identifier for CSV file naming.

        Returns:
            dict: Result with 'result', 'isTable', and 'csv' keys.

        Raises:
            NLP2SQLException: If execution fails or tool returns an error.
        """
        logger.info(f"NLP2SQLService: Executing SQL: {sql_query}")
        try:
            raw_result = await invoke_mcp_tool(
                config["NLP2SQL_SQL_SERVER_URL"],
                "execute_sql_query",
                {"query": sql_query, "session_id": session_id},
            )
            result = self.decode_tool_result(raw_result, "execute_sql_query")
            logger.info(f"NLP2SQLService: execute_sql_query result: {result}")

            if isinstance(result, dict) and result.get("error"):
                raise NLP2SQLException(result["error"], "NLP2SQL0004")

            return {
                "result": result.get("results", []),
                "isTable": True,
                "csv": result.get("csv"),
            }
        except NLP2SQLException:
            raise
        except ValueError as e:
            logger.exception(f"NLP2SQLService: Invalid response from execute_sql_query: {e}")
            raise NLP2SQLException("Invalid response received from SQL execution service.", "NLP2SQL0008")
        except Exception as e:
            logger.exception(f"NLP2SQLService: SQL execution failed: {e}")
            raise NLP2SQLException(self.extract_exception_message(e), "NLP2SQL0004")

    # ------------------------------------------------------------------ #
    #  SQL fixing                                                          #
    # ------------------------------------------------------------------ #

    async def fix_sql(self, params, session_id: str = None) -> dict:
        """
        Uses the LLM via the SQL MCP server to fix a broken SQL query.

        Args:
            params (FixSQLRequestParams): Validated fix parameters.
            session_id (str | None): Used to fetch user_query from history if not provided.

        Returns:
            dict: Fixed SQL result from the MCP tool.

        Raises:
            NLP2SQLException: If fixing fails.
        """
        logger.info(f"NLP2SQLService: Attempting to fix SQL: {params.sql_query}")
        try:
            user_query = params.user_query
            if not user_query and params.session_id:
                history = self.memory_manager.get_conversation_history(params.session_id, limit=1)
                if history:
                    user_query = history[0]["user_query"]

            if not user_query:
                logger.warning("NLP2SQLService: No user_query for fix context. Fix quality may be affected.")

            logger.info(f"NLP2SQLService: Fetching schema for tables: {params.table_names}")
            schema_info = await invoke_mcp_tool(
                config["NLP2SQL_SQL_SERVER_URL"],
                "prepare_table_schema",
                {"table_names": json.dumps(params.table_names)},
            )

            raw_result = await invoke_mcp_tool(
                config["NLP2SQL_SQL_SERVER_URL"],
                "fix_sql_query",
                {
                    "query": params.sql_query,
                    "error_message": params.error_message,
                    "table_schemas": schema_info,
                    "user_query": user_query or "",
                },
            )

            result = self.decode_tool_result(raw_result, "fix_sql_query")
            logger.debug(f"NLP2SQLService: Fixed SQL result: {result}")

            if isinstance(result, dict) and result.get("error"):
                raise NLP2SQLException(result["error"], "NLP2SQL0006")

            return result

        except NLP2SQLException:
            raise
        except ValueError as e:
            logger.exception(f"NLP2SQLService: Invalid response from fix_sql_query: {e}")
            raise NLP2SQLException("Invalid response from SQL fix service.", "NLP2SQL0009")
        except Exception as e:
            logger.exception(f"NLP2SQLService: SQL fixing failed: {e}")
            raise NLP2SQLException(f"Failed to fix SQL: {self.extract_exception_message(e)}", "NLP2SQL0006")

    # ------------------------------------------------------------------ #
    #  Feedback                                                            #
    # ------------------------------------------------------------------ #

    async def log_feedback(self, session_id: str, user_query: str, is_negative: bool):
        """
        Marks a conversation entry with feedback in memory.

        Args:
            session_id (str): Session identifier.
            user_query (str): The query to mark.
            is_negative (bool): True for negative feedback, False for positive.

        Raises:
            NLP2SQLException: If feedback logging fails.
        """
        try:
            logger.info(
                f"NLP2SQLService: Logging feedback for session {session_id} "
                f"on query: '{user_query}' | is_negative={is_negative}"
            )
            self.memory_manager.mark_feedback(
                session_id=session_id,
                user_query=user_query,
                is_negative=is_negative,
            )
        except Exception as e:
            logger.exception(f"NLP2SQLService: Failed to log feedback: {e}")
            raise NLP2SQLException("Internal error while logging feedback.", "NLP2SQL0007")
