"""
Vector Filter MCP Server

Provides MCP tools for semantic similarity-based table filtering using
bge-large-v1.5 embeddings and PGVector cosine search. Runs as a standalone
FastAPI/MCP server.

Start:
    uvicorn mcp_servers.vector_server_streamable_http:app --port 8002
"""

import os
import sys
import psycopg2
import yaml
from typing import List
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from fastapi import FastAPI

# ------------------------------------------------------------------ #
#  Path setup                                                          #
# ------------------------------------------------------------------ #
_MCP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MCP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.observability.logging import logger

os.environ["no_proxy"] = "localhost,127.0.0.1,::1,10.170.80.113"

try:
    with open(os.path.join(_PROJECT_ROOT, "configs", "nlp2sql_config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    logger.info("VectorServer: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"VectorServer: Failed to load configuration: {e}")
    raise

project_root = config.get("PROJECT_ROOT")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.utils.embedding_model import generate_embedding
    logger.info("VectorServer: Embedding model import loaded successfully.")
except ImportError as e:
    logger.exception(f"VectorServer: Failed to import embedding model: {e}")
    raise

DB_HOST = config["NLP2SQL_DB_HOST"]
DB_PORT = config["NLP2SQL_DB_PORT"]
DB_USER = config["NLP2SQL_DB_USER"]
DB_PASSWORD = config["NLP2SQL_DB_PASSWORD"]
DB_NAME = config["NLP2SQL_DB_NAME"]


class VectorResponse(BaseModel):
    relevant_tables: List[str] = Field(
        description="List of relevant table names from the given table list"
    )


mcp = FastMCP("VectorFilterServer", stateless_http=True)
app = FastAPI(
    title="VectorFilterServer",
    lifespan=lambda app: mcp.session_manager.run(),
)
app.mount("/vector_server", mcp.streamable_http_app())


def text_embedder(text: str) -> list:
    """
    Generates a dense vector embedding for the given text using bge-large-v1.5.

    Args:
        text (str): Input text to embed.

    Returns:
        list: Embedding vector.

    Raises:
        RuntimeError: If embedding generation fails.
    """
    logger.debug("VectorServer: Generating embedding for input text.")
    try:
        embedding = generate_embedding(text, "bge-large-v1.5")
        logger.debug("VectorServer: Embedding generated successfully.")
        return embedding
    except Exception as e:
        logger.exception("VectorServer: Failed to generate embedding.")
        raise RuntimeError(f"Embedding generation failed: {str(e)}") from e


@mcp.tool()
def get_relevant_tables(query: str, table_names: list):
    """
    Given a query, returns top-k most relevant tables via PGVector cosine similarity.

    Args:
        query: The user query to embed and search against.
        table_names: List of candidate table names to filter.

    Returns:
        dict: {'relevant_tables': [table_name, ...]} or {'error': str}
    """
    logger.info(f"VectorServer: Tool call: get_relevant_tables | Query: '{query}' | Tables: {table_names}")
    conn = None
    cur = None
    try:
        query_embedding = text_embedder(query)
        if not query_embedding:
            logger.error("VectorServer: Query embedding is empty. Aborting.")
            return {"error": "Failed to generate query embedding."}

        top_k = config["NLP2SQL_VECTOR_SEARCH_TOP_K"]
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
        )
        cur = conn.cursor()

        placeholders = ','.join(['%s'] * len(table_names))
        sql = f"""
            SELECT table_name FROM table_vector_embeddings
            WHERE table_name IN ({placeholders})
            AND db_name = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = table_names + [config["NLP2SQL_TARGET_DB"], query_embedding, top_k]
        cur.execute(sql, params)
        result = cur.fetchall()

        table_results = [row[0] for row in result]
        logger.info(f"VectorServer: Relevant tables retrieved: {table_results}")
        return {"relevant_tables": table_results}

    except Exception as e:
        logger.exception("VectorServer: Error in get_relevant_tables.")
        return {"error": "Failed to retrieve relevant tables."}
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
