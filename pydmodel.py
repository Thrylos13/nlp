"""
NLP2SQL Pydantic Models

All request/response Pydantic BaseModel definitions used across the
NLP2SQL API routes. TypedDicts for LangGraph state remain in utils/state.py.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SessionRequest(BaseModel):
    """Request body for starting a new conversation session."""
    userId: int
    appId: int


class EndSessionRequest(BaseModel):
    """Request body for ending a session or logging feedback."""
    userId: int
    appId: int
    requestParams: dict


class SQLRequest(BaseModel):
    """
    Main request body for SQL generation, execution, and fix endpoints.

    Attributes:
        userId (int): Requesting user identifier.
        appId (int): Application identifier.
        aiModelId (int): AI model identifier.
        os (str): Client operating system.
        browserClientId (str): Client browser identifier.
        requestParams (dict): Endpoint-specific parameters.
        moduleDetails (dict): LLM model server details.
        queueFlag (bool): Whether to log this request to the audit queue.
    """
    userId: int
    appId: int
    aiModelId: int
    os: str
    browserClientId: str
    requestParams: dict
    moduleDetails: dict
    queueFlag: bool


class ExecuteSQLRequestParams(BaseModel):
    """Parameters for the /execute_sql endpoint."""
    sql_query: str = Field(..., description="The SQL query to execute directly.")
    session_id: str = Field(..., description="Session ID of the requesting session.")


class FixSQLRequestParams(BaseModel):
    """Parameters for the /fix_sql endpoint."""
    sql_query: str = Field(..., description="The incorrect SQL query to fix.")
    error_message: str = Field(..., description="The database error message from the failed execution.")
    table_names: List[str] = Field(..., description="Relevant table names for schema fetching.")
    user_query: Optional[str] = Field(
        None,
        description="Original natural language query for fix context. Fetched from session history if not provided."
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID to retrieve user_query from memory if not provided directly."
    )


class ExecuteAndFixSQLRequestParams(BaseModel):
    """Parameters for the /execute_and_fix_sql endpoint."""
    sql_query: str = Field(..., description="The initial SQL query to execute.")
    max_retries: int = Field(3, description="Maximum fix attempts after failure.")
    user_query: Optional[str] = Field(None, description="Original natural language query for fix context.")
    session_id: Optional[str] = Field(None, description="Session ID for context retrieval.")


class QueryRefinementResponse(BaseModel):
    """LLM output schema for the query refinement node."""
    refined_query: str = Field(
        description="Refined and contextually complete query based on conversation history."
    )
    requires_context: bool = Field(
        description="Whether the original query required context from previous conversations."
    )


class ConversationEntry(BaseModel):
    """Represents a single conversation turn stored in memory."""
    user_query: str
    refined_query: str
    generated_sql: str
    tables_used: List[str]
    timestamp: str
    is_negative_feedback: bool = False
