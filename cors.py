"""
CORS Middleware

Centralises CORS policy configuration for the NLP2SQL application.
Register by calling apply_cors(app) from main.py.
"""

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.observability.logging import logger

try:
    with open("configs/nlp2sql_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info("CORS Middleware: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"CORS Middleware: Failed to load configuration: {e}")
    raise


def apply_cors(app: FastAPI):
    """
    Registers CORSMiddleware on the provided FastAPI application instance.

    Args:
        app (FastAPI): The FastAPI application instance to attach middleware to.
    """
    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get("origins", ["*"]),
            allow_credentials=True,
            allow_methods=config.get("methods", ["*"]),
            allow_headers=config.get("headers", ["*"]),
        )
        logger.info("CORS middleware applied successfully.")
    except Exception as e:
        logger.error(f"Failed to apply CORS middleware: {e}")
        raise
