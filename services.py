"""
Services

Utility functions for LLM initialisation and audit queue logging.
Used by the NLP2SQL service layer and LangGraph nodes.
"""

import os
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from utils.config_loader import config
from app.observability.logging import logger

load_dotenv()


def initialize_llm(model_id: int, module_details: dict):
    """
    Initialises and returns the appropriate LangChain LLM instance
    based on model type.

    Args:
        model_id (int): AI model identifier (unused directly, kept for signature).
        module_details (dict): Dict with modelType, modelName, modelAppServer keys.

    Returns:
        LangChain LLM instance (ChatOpenAI or ChatOllama).

    Raises:
        Exception: If LLM initialisation fails.
    """
    try:
        logger.info(f"initialize_llm: Module details: {module_details}")
        if module_details["modelType"] == 330011:
            logger.info("initialize_llm: Initialising ChatOpenAI LLM.")
            return ChatOpenAI(
                model=module_details["modelName"],
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("AZURE_ENDPOINT"),
            )
        else:
            logger.info(f"initialize_llm: Initialising ChatOllama with model: {module_details.get('modelName')}")
            return ChatOllama(
                base_url=module_details.get("modelAppServer"),
                model=module_details.get("modelName"),
                num_ctx=config.get("NLP2SQL_LABEL_SERVER_N_CTX", 4096),
            )
    except Exception as e:
        logger.error(f"initialize_llm: Failed to initialise LLM: {e}")
        raise


def log_audit_event(op: str, payload: dict):
    """
    Sends a formatted payload to the audit queue.

    Note: Prefer AuditRepository.log_event() for service-layer code.
    This helper is kept for use in LangGraph nodes and legacy callers.

    Args:
        op (str): Queue operation name (e.g. 'log_request_mst').
        payload (dict): Data payload to enqueue.
    """
    try:
        from artemis_queues import send_to_queue
        send_to_queue.send({'op': op, 'payload': payload})
        logger.debug(f"log_audit_event: Logged '{op}' successfully.")
    except Exception as e:
        logger.error(f"log_audit_event: Failed to log event '{op}': {e}")
