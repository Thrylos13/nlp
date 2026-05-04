"""
NLP2SQL Routes

Thin route handlers for all NLP2SQL endpoints.
All business logic is delegated to NLP2SQLService.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.pydantic_model.pydmodel import (
    SQLRequest, SessionRequest, EndSessionRequest,
    ExecuteSQLRequestParams, FixSQLRequestParams,
)
from app.service.nlp2sql_service import NLP2SQLService
from app.exceptions.nlp2sql_exception import (
    NLP2SQLException, SessionException, PromptFetchException, WorkflowExecutionException
)
from app.observability.logging import logger
import mlflow

router = APIRouter()
service = NLP2SQLService()


@router.post("/start_session", summary="Start a new conversation session")
async def start_session(request: SessionRequest):
    """
    Creates a new unique session ID to track a conversation.
    This ID should be passed in subsequent requests to maintain context.
    """
    try:
        session_id = await service.start_session(request.userId)
        return JSONResponse(content={
            "errorMessage": None,
            "response": session_id,
            "status": "success",
            "errorCode": "NLP2SQL0000",
        }, status_code=200)
    except SessionException as e:
        logger.error(f"Session start failed: {e}")
        return JSONResponse(content={
            "errorMessage": "Unable to initialize a session.",
            "response": None,
            "status": "fail",
            "errorCode": e.code,
        }, status_code=500)
    except Exception as e:
        logger.exception(f"Unexpected error starting session: {e}")
        return JSONResponse(content={
            "errorMessage": "Unable to initialize a session.",
            "response": None,
            "status": "fail",
            "errorCode": "NLP2SQL0001",
        }, status_code=500)


@router.post("/end_session", summary="End a conversation session")
async def end_session(request: EndSessionRequest):
    """
    Formally ends a conversation session and clears its in-memory cache.
    """
    session_id = request.requestParams.get('sessionId')
    if not session_id:
        logger.error(f"end_session called without sessionId: {request}")
        return JSONResponse(content={
            "errorMessage": "Missing required parameter: sessionId.",
            "response": None,
            "status": "fail",
            "errorCode": "NLP2SQL0002",
        }, status_code=400)

    try:
        await service.end_session(request.userId, session_id)
        return JSONResponse(content={
            "errorMessage": None,
            "response": "Session Ended Successfully",
            "status": "success",
            "errorCode": "NLP2SQL0000",
        }, status_code=200)
    except SessionException as e:
        logger.exception(f"Session end failed for {session_id}: {e}")
        return JSONResponse(content={
            "errorMessage": "Unable to end session.",
            "response": None,
            "status": "fail",
            "errorCode": e.code,
        }, status_code=500)
    except Exception as e:
        logger.exception(f"Unexpected error ending session {session_id}: {e}")
        return JSONResponse(content={
            "errorMessage": "Unable to end session.",
            "response": None,
            "status": "fail",
            "errorCode": "NLP2SQL0003",
        }, status_code=500)


@router.post("/", summary="Generate SQL from a natural language query")
@mlflow.trace(name="generate_sql_request")
async def generate_sql_from_query(request: SQLRequest):
    """
    Main endpoint to process a natural language query and return a SQL statement.
    """
    query = request.requestParams.get("query")
    session_id = request.requestParams.get("sessionId")

    if not query:
        logger.error(f"generate_sql called without query: {request}")
        return JSONResponse(content={
            "errorMessage": "Missing required parameter: query.",
            "response": None,
            "status": "fail",
            "errorCode": "NLP2SQL0004",
        }, status_code=400)

    try:
        response_payload = await service.generate_sql(request)
        return JSONResponse(content={
            "status": "success",
            "errorCode": "NLP2SQL0000",
            "errorMessage": None,
            "response": {
                "query": response_payload["query"],
                "refined_query": response_payload["refined_query"],
                "tables_used": response_payload.get("tables_used", []),
                "token_details": response_payload["token_details"],
            },
        })
    except (NLP2SQLException, PromptFetchException, WorkflowExecutionException) as e:
        logger.exception(f"Domain error generating SQL for session {session_id}: {e}")
        return JSONResponse(content={
            "status": "fail",
            "errorCode": e.code,
            "errorMessage": "An internal error occurred while generating the SQL query.",
            "response": None,
        }, status_code=500)
    except Exception as e:
        logger.exception(f"Unexpected error generating SQL for session {session_id}: {e}")
        return JSONResponse(content={
            "status": "fail",
            "errorCode": "NLP2SQL0006",
            "errorMessage": "An internal error occurred while generating the SQL query.",
            "response": None,
        }, status_code=500)


@router.post("/feedback", summary="Log feedback for a query")
async def log_feedback(request: EndSessionRequest):
    """
    Allows a client to mark a specific query in a session as having
    received negative or positive feedback.
    """
    try:
        req_param = request.requestParams
        await service.log_feedback(
            session_id=req_param["sessionId"],
            user_query=req_param["query"],
            is_negative=req_param["isNegative"],
        )
        return JSONResponse(content={
            "status": "success",
            "response": "Feedback logged successfully.",
            "errorMessage": None,
            "errorCode": "NLP2SQL0000",
        }, status_code=200)
    except NLP2SQLException as e:
        logger.exception(f"Failed to log feedback: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": "Internal error occurred while logging feedback.",
            "errorCode": e.code,
        }, status_code=500)
    except Exception as e:
        logger.exception(f"Unexpected error logging feedback: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": "Internal error occurred while logging feedback.",
            "errorCode": "NLP2SQL0007",
        }, status_code=500)


@router.post("/execute_sql", summary="Execute a raw SQL query")
async def execute_sql(request: SQLRequest):
    """
    Directly executes a provided SQL query against the database.
    """
    try:
        params = ExecuteSQLRequestParams(**request.requestParams)
    except ValidationError as e:
        logger.exception(f"Validation error in /execute_sql: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": "Invalid request payload.",
            "errorCode": "NLP2SQL0004",
        }, status_code=422)

    try:
        result = await service.execute_sql(params.sql_query, params.session_id)
        return JSONResponse(content={
            "status": "success",
            "response": result,
            "errorMessage": None,
            "errorCode": "NLP2SQL0000",
        }, status_code=200)
    except NLP2SQLException as e:
        logger.exception(f"SQL execution failed: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": str(e),
            "errorCode": e.code,
        }, status_code=400 if e.code == "NLP2SQL0004" else 500)
    except Exception as e:
        logger.exception(f"Unexpected error in /execute_sql: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": "Unexpected error during SQL execution.",
            "errorCode": "NLP2SQL0004",
        }, status_code=500)


@router.post("/fix_sql", summary="Fix a broken SQL query")
async def fix_sql(request: SQLRequest):
    """
    Attempts to fix a broken SQL query using an LLM.
    """
    try:
        params = FixSQLRequestParams(**request.requestParams)
    except ValidationError as e:
        logger.exception(f"Validation error in /fix_sql: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": "Invalid request payload.",
            "errorCode": "NLP2SQL0005",
        }, status_code=422)

    try:
        result = await service.fix_sql(params)
        return JSONResponse(content={
            "status": "success",
            "response": result,
            "errorMessage": None,
            "errorCode": "NLP2SQL0000",
        }, status_code=200)
    except NLP2SQLException as e:
        logger.exception(f"SQL fix failed: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": str(e),
            "errorCode": e.code,
        }, status_code=500)
    except Exception as e:
        logger.exception(f"Unexpected error in /fix_sql: {e}")
        return JSONResponse(content={
            "status": "fail",
            "response": None,
            "errorMessage": "Unexpected error during SQL fix.",
            "errorCode": "NLP2SQL0006",
        }, status_code=500)
