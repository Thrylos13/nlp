"""
NLP2SQL Exceptions

Defines custom exception types used across the NLP2SQL application
to represent domain-specific errors with structured error codes.
"""


class NLP2SQLException(Exception):
    """
    Custom exception for NLP2SQL domain errors.

    Args:
        message (str): Human-readable error description.
        code (str): Structured error code (e.g. "NLP2SQL0005").
    """
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


class PromptFetchException(NLP2SQLException):
    """Raised when prompt retrieval from the database fails."""
    pass


class WorkflowExecutionException(NLP2SQLException):
    """Raised when the LangGraph workflow execution fails."""
    pass


class SessionException(NLP2SQLException):
    """Raised for session start/end/validation errors."""
    pass
