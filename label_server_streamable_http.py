"""
Label Identification MCP Server

Provides MCP tools for identifying relevant database tables based on
semantic labels. Runs as a standalone FastAPI/MCP server.

Start:
    uvicorn mcp_servers.label_server_streamable_http:app --port 8001
"""

import os
import sys
import psycopg2
import yaml
from typing import List
from mcp.server.fastmcp import FastMCP
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
#  Path setup — mcp_servers/ is one level below project root          #
# ------------------------------------------------------------------ #
_MCP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MCP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.observability.logging import logger
from fetch_details_from_database import fetch_prompt, fetch_app_id_module_id, fetch_model_id

load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,::1,10.170.80.113"

try:
    with open(os.path.join(_PROJECT_ROOT, "configs", "nlp2sql_config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    logger.info("LabelServer: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"LabelServer: Failed to load configuration: {e}")
    raise

DB_HOST = config["NLP2SQL_DB_HOST"]
DB_PORT = config["NLP2SQL_DB_PORT"]
DB_USER = config["NLP2SQL_DB_USER"]
DB_PASSWORD = config["NLP2SQL_DB_PASSWORD"]
DB_NAME = config["NLP2SQL_DB_NAME"]


class LabelResponse(BaseModel):
    labels: List[str] = Field(description="List of relevant labels extracted from the query")


mcp = FastMCP("LabelIdentificationServer", stateless_http=True)
app = FastAPI(
    title="LabelIdentificationServer",
    lifespan=lambda app: mcp.session_manager.run(),
)
app.mount("/label_server", mcp.streamable_http_app())


def fetch_labels_from_db(database: str) -> dict:
    """
    Fetches table names and their label clusters from the vector DB.

    Args:
        database (str): Database name to connect to.

    Returns:
        dict: {table_name: [label, ...]} mapping.
    """
    logger.info(f"LabelServer: Fetching labels from database: {database}")
    target_db_name = config["NLP2SQL_TARGET_DB"]
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=database,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=15,
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name, cluster FROM table_vector_embeddings WHERE db_name = %s",
            (target_db_name,),
        )
        rows = cur.fetchall()

        label_dict = {}
        for table_name, cluster in rows:
            if isinstance(cluster, str):
                labels = [label.strip() for label in cluster.split(",")]
            elif isinstance(cluster, list):
                labels = cluster
            else:
                labels = []
            label_dict[table_name] = labels

        logger.info("LabelServer: Successfully fetched labels from DB.")
        return label_dict
    except Exception as e:
        logger.exception(f"LabelServer: Failed to fetch labels from DB: {e}")
        return {}
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@mcp.tool()
def get_table_labels() -> dict:
    """
    Fetches names of all tables and their corresponding labels from the database.
    Required by the `find_relevant_tables` tool.
    """
    try:
        logger.info("LabelServer: Tool call: get_table_labels")
        labels = fetch_labels_from_db(DB_NAME)
        if labels:
            logger.info(f"LabelServer: Labels fetched: {labels}")
            return labels
        logger.warning("LabelServer: No labels found.")
        return {"error": "No labels found or failed to fetch data."}
    except Exception as e:
        logger.error(f"LabelServer: Error in get_table_labels: {e}")
        return {}


@mcp.tool()
def get_unique_table_labels() -> List:
    """
    Fetches and returns a unique list of all labels present in the database.
    Required by the `find_relevant_tables` tool.
    """
    logger.info("LabelServer: Tool call: get_unique_table_labels")
    conn = None
    cur = None
    try:
        target_db_name = config["NLP2SQL_TARGET_DB"]
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=15,
        )
        cur = conn.cursor()
        query = """
        SELECT DISTINCT unnest(cluster)
        FROM public.table_vector_embeddings
        WHERE db_name = %s AND cluster IS NOT NULL;
        """
        cur.execute(query, (target_db_name,))
        result = cur.fetchall()
        unique_clusters = [row[0] for row in result]
        logger.info(f"LabelServer: Unique clusters: {unique_clusters}")
        return unique_clusters
    except Exception as e:
        logger.exception(f"LabelServer: Error fetching unique labels: {e}")
        return {"error": str(e)}
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@mcp.tool()
def find_relevant_tables(user_query: str, labels_dict: dict, unique_labels: List):
    """
    Returns a list of table names relevant to the user query.

    Args:
        user_query: The user query for which relevant tables should be identified.
        labels_dict: Key-value pairs of table names and their clusters (from get_table_labels).
        unique_labels: All unique labels in the database (from get_unique_table_labels).

    Returns:
        dict: {'tables': [table_name, ...], 'tokens': usage_metadata}
    """
    try:
        logger.info(f"LabelServer: find_relevant_tables for query: '{user_query}'")
        label_str = "\n".join(unique_labels)
        logger.info(f"LabelServer: Unique labels: {label_str}")

        parser = PydanticOutputParser(pydantic_object=LabelResponse)
        ids = fetch_app_id_module_id()
        app_id = int(ids[0])
        model_details = fetch_model_id(app_id)
        model_id = int(model_details[0])
        model_name = model_details[1]
        model_host = model_details[2]
        model_port = int(model_details[3])
        model_type = int(model_details[4])

        if model_type == 330011:
            model = ChatOpenAI(
                model=model_name,
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("AZURE_ENDPOINT"),
            )
        else:
            model = ChatOllama(
                base_url=f"http://{model_host}:{model_port}",
                model=model_name,
                num_ctx=config["NLP2SQL_LABEL_SERVER_N_CTX"],
            )

        prompt_template = fetch_prompt(
            user_id=9999, app_id=app_id, model_id=model_id,
            prompt_identifier="nlp_sql_label_server",
        )
        logger.info(f"LabelServer: Fetched prompt: {prompt_template}")

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "label_list"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        formatted_prompt = prompt.invoke({"query": user_query, "label_list": label_str})
        raw_response = model.invoke(formatted_prompt)
        usage_metadata = raw_response.usage_metadata
        response = StrOutputParser().invoke(raw_response)

        try:
            parsed = parser.parse(response)
            relevant_labels = parsed.labels
        except Exception as e:
            logger.warning(f"LabelServer: Failed to parse labels: {e}")
            return ["[ERROR] Could not parse labels", str(e)]

        logger.info(f"LabelServer: Relevant labels: {relevant_labels}")

        matching_tables = [
            table for table, labels in labels_dict.items()
            if any(label in labels for label in relevant_labels)
        ]
        logger.info(f"LabelServer: Matching tables: {matching_tables}")

        return {"tables": matching_tables, "tokens": usage_metadata}

    except Exception as e:
        logger.exception(f"LabelServer: Error in find_relevant_tables: {e}")
        return {"error": str(e)}
