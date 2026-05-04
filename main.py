"""
NLP2SQL — Main Application Entry Point

Initialises the FastAPI application with:
- CORS middleware
- MLflow tracing
- All API route registrations

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import sys
import os
import yaml
import mlflow
import uvicorn
from fastapi import FastAPI

from app.observability.logging import logger
from app.middleware.cors import apply_cors

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
try:
    with open("configs/nlp2sql_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info("Main: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"Main: Failed to load configuration: {e}")
    raise

os.environ["no_proxy"] = config.get("NO_PROXY_ENV", "localhost,127.0.0.1,::1")

if config.get('PROJECT_ROOT') and config['PROJECT_ROOT'] not in sys.path:
    sys.path.append(config['PROJECT_ROOT'])

# ------------------------------------------------------------------ #
#  MLflow                                                              #
# ------------------------------------------------------------------ #
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nlp2sql_traces")
    mlflow.openai.autolog()
    mlflow.langchain.autolog()
    logger.info("Main: MLflow configured successfully.")
except Exception as e:
    logger.warning(f"Main: MLflow configuration failed (non-fatal): {e}")

# ------------------------------------------------------------------ #
#  FastAPI app                                                         #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="NLP to SQL Generation Service",
    description="An advanced agent-based system to convert natural language queries into SQL.",
    version="1.0.0",
)

apply_cors(app)

# ------------------------------------------------------------------ #
#  Router registration                                                 #
# ------------------------------------------------------------------ #
from app.api.v1.routes.nlp2sql import router as nlp2sql_router
app.include_router(nlp2sql_router, tags=["NLP2SQL"])

# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
