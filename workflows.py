"""
NLP2SQL LangGraph Workflows

Builds and compiles all LangGraph workflows:
- label_agent_workflow: table identification via label matching
- vector_agent_workflow: semantic similarity-based table filtering
- sql_agent_workflow: schema preparation and SQL generation
- main_sql_generation_graph: orchestrates all sub-workflows
"""

from langgraph.graph import StateGraph, END
from utils.state import AgentState, LabelAgentState, VectorAgentState, SQLAgentState
from utils.nodes import (
    retrieve_conversation_history_node,
    query_refinement_node,
    label_agent_node,
    vector_agent_node,
    sql_generate_agent_node,
    save_conversation_node,
    get_table_labels_node,
    get_unique_labels_node,
    find_relevant_tables_node,
    vector_search_node,
    prepare_sql_schema_node,
    generate_sql_query_node,
)


def create_label_agent_workflow():
    """Builds and compiles the label agent sub-workflow."""
    workflow = StateGraph(LabelAgentState)
    workflow.add_node("get_table_labels", get_table_labels_node)
    workflow.add_node("get_unique_labels", get_unique_labels_node)
    workflow.add_node("find_relevant_tables", find_relevant_tables_node)

    workflow.set_entry_point("get_table_labels")
    workflow.add_edge("get_table_labels", "get_unique_labels")
    workflow.add_edge("get_unique_labels", "find_relevant_tables")
    workflow.add_edge("find_relevant_tables", END)

    return workflow.compile()


def create_vector_agent_workflow():
    """Builds and compiles the vector agent sub-workflow."""
    workflow = StateGraph(VectorAgentState)
    workflow.add_node("vector_search", vector_search_node)
    workflow.set_entry_point("vector_search")
    workflow.add_edge("vector_search", END)
    return workflow.compile()


def create_sql_agent_workflow():
    """Builds and compiles the SQL generation sub-workflow."""
    workflow = StateGraph(SQLAgentState)
    workflow.add_node("prepare_schema", prepare_sql_schema_node)
    workflow.add_node("generate_sql", generate_sql_query_node)

    workflow.set_entry_point("prepare_schema")
    workflow.add_edge("prepare_schema", "generate_sql")
    workflow.add_edge("generate_sql", END)
    return workflow.compile()


def create_main_sql_generation_graph():
    """Builds and compiles the main overarching SQL generation workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_memory", retrieve_conversation_history_node)
    workflow.add_node("refine_query", query_refinement_node)
    workflow.add_node("label_agent", label_agent_node)
    workflow.add_node("vector_agent", vector_agent_node)
    workflow.add_node("sql_agent", sql_generate_agent_node)
    workflow.add_node("save_memory", save_conversation_node)

    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "refine_query")
    workflow.add_edge("refine_query", "label_agent")
    workflow.add_edge("label_agent", "vector_agent")
    workflow.add_edge("vector_agent", "sql_agent")
    workflow.add_edge("sql_agent", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


# Compile all workflows for import
label_agent_workflow = create_label_agent_workflow()
vector_agent_workflow = create_vector_agent_workflow()
sql_agent_workflow = create_sql_agent_workflow()
main_sql_generation_graph = create_main_sql_generation_graph()
