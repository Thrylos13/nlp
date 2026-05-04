"""
NLP2SQL State

TypedDict definitions for LangGraph agent states.
Pydantic BaseModel request/response models live in app/pydantic_model/pydmodel.py.
"""

from typing import List, Dict, Any, TypedDict, Optional

# Re-export Pydantic models so existing imports from utils.state still work
from app.pydantic_model.pydmodel import (
    SessionRequest,
    EndSessionRequest,
    SQLRequest,
    ExecuteSQLRequestParams,
    FixSQLRequestParams,
    ExecuteAndFixSQLRequestParams,
    QueryRefinementResponse,
    ConversationEntry,
)


# ------------------------------------------------------------------ #
#  LangGraph TypedDicts                                                #
# ------------------------------------------------------------------ #

class LabelAgentState(TypedDict):
    query: str
    table_labels: Dict[str, Any]
    unique_labels: List[str]
    relevant_tables: List[str]
    error: str
    tokens: Dict[str, Any]


class VectorAgentState(TypedDict):
    query: str
    initial_tables: List[str]
    relevant_tables: List[str]
    error: str
    tokens: Dict[str, Any]


class SQLAgentState(TypedDict):
    query: str
    tables: List[str]
    schema_info: Dict[str, Any]
    sql_query: str
    conversation_history: List[Dict]
    error: str
    tokens: Dict[str, Any]


class AgentState(TypedDict):
    original_query: str
    refined_query: str
    user_id: str
    app_id: int
    model_id: int
    llm: Any
    label_tables: List[str]
    vector_tables: List[str]
    sql_query: str
    label_tokens: Dict[str, Any]
    vector_tokens: Dict[str, Any]
    sql_tokens: Dict[str, Any]
    refinement_tokens: Dict[str, Any]
    error: str
    prompts: Dict[str, str]
    conversation_history: List[ConversationEntry]
    session_id: Optional[str]
    query_required_context: bool
