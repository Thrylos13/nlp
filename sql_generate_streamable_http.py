"""
SQL Generation MCP Server

Provides MCP tools for table schema preparation, SQL generation,
SQL execution, and SQL fixing. Runs as a standalone FastAPI/MCP server.

Start:
    uvicorn mcp_servers.sql_generate_streamable_http:app --port 8003
"""

import os
import sys
import psycopg2
import csv
import yaml
from typing import List, Union
from mcp.server.fastmcp import FastMCP
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from fastapi import FastAPI
from dotenv import load_dotenv
from datetime import datetime

# ------------------------------------------------------------------ #
#  Path setup                                                          #
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
    logger.info("SQLServer: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"SQLServer: Failed to load configuration: {e}")
    raise

VECTOR_DB_HOST = config["NLP2SQL_DB_HOST"]
VECTOR_DB_PORT = config["NLP2SQL_DB_PORT"]
VECTOR_DB_USER = config["NLP2SQL_DB_USER"]
VECTOR_DB_PASSWORD = config["NLP2SQL_DB_PASSWORD"]
VECTOR_DB_NAME = config["NLP2SQL_DB_NAME"]
SQL_DB_NAME = config["DIGI_DEV_DB_NAME"]
SQL_DB_USER = config["DIGI_DEV_DB_USER"]
SQL_DB_PASSWORD = config["DIGI_DEV_DB_PASSWORD"]
SQL_DB_HOST = config["DIGI_DEV_DB_HOST"]
SQL_DB_PORT = config["DIGI_DEV_DB_PORT"]


class SQLResponse(BaseModel):
    query: str = Field(description="The generated SQL query as a string.")


mcp = FastMCP("SQLGenerationServer", stateless_http=True)
app = FastAPI(
    title="SQLGenerationServer",
    lifespan=lambda app: mcp.session_manager.run(),
)
app.mount("/sql_server", mcp.streamable_http_app())


@mcp.tool()
def prepare_table_schema(table_names: List) -> str:
    """
    Returns table schema, description, and column descriptions for all
    tables in the provided list.

    Args:
        table_names: List of table names to fetch schema for.

    Returns:
        str: Formatted schema string for all tables.
    """
    logger.info(f"SQLServer: prepare_table_schema called for tables: {table_names}")
    conn = None
    try:
        conn = psycopg2.connect(
            host=VECTOR_DB_HOST, port=VECTOR_DB_PORT,
            dbname=VECTOR_DB_NAME, user=VECTOR_DB_USER,
            password=VECTOR_DB_PASSWORD,
        )
        table_details = []
        for name in table_names:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT table_description, column_description, schema "
                    "FROM table_vector_embeddings WHERE table_name = %s",
                    (name,),
                )
                rows = cur.fetchone()
            finally:
                cur.close()

            if not rows:
                logger.warning(f"SQLServer: No schema info found for table: {name}")
                continue

            desc, col_desc, schema = rows
            table_details.append(f"""
###########################
TABLE NAME: {name}
TABLE DESCRIPTION: {desc}
TABLE SCHEMA & FIRST 3 ROWS: {schema}
TABLE COLUMNS DESCRIPTION: {col_desc}
###########################
""")

        tables_schema = "\n".join(table_details)
        logger.debug("SQLServer: Schema prepared successfully.")
        return tables_schema

    except Exception as e:
        logger.exception("SQLServer: Failed during prepare_table_schema.")
        return {"error": f"Failed to prepare schema: {str(e)}"}
    finally:
        if conn:
            conn.close()


@mcp.tool()
def generate_sql_query(query: str, tables_schema: str, conversation_context: str):
    """
    Given a user query and table schema, generates a SQL query.

    Args:
        query: Natural language user query.
        tables_schema: Schema details for relevant tables (from prepare_table_schema).
        conversation_context: Prior conversation context for multi-turn queries.

    Returns:
        dict: {'generated_sql': SQLResponse, 'tokens': usage_metadata}
    """
    logger.info("SQLServer: generate_sql_query called.")
    try:
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
                num_ctx=config["NLP2SQL_SQL_SERVER_N_CTX"],
            )

        prompt_type = "nlp_sql_sql_server_digi_dev"
        prompt_template = fetch_prompt(
            user_id=9999, app_id=app_id, model_id=model_id,
            prompt_identifier=prompt_type,
        )
        logger.info(f"SQLServer: Fetched prompt: {prompt_template}")

        parser = PydanticOutputParser(pydantic_object=SQLResponse)
        prompt = PromptTemplate(
            input_variables=["query", "tables_schema", "conversation_context"],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        formatted_prompt = prompt.invoke({
            "query": query,
            "tables_schema": tables_schema,
            "conversation_context": conversation_context,
        })
        raw_response = model.invoke(formatted_prompt)
        usage_metadata = raw_response.usage_metadata
        logger.info(f"SQLServer: LLM Token Usage: {usage_metadata}")

        try:
            parsed = parser.parse(raw_response.content)
            logger.info(f"SQLServer: Generated SQL: {raw_response.content}")
            return {"generated_sql": parsed, "tokens": usage_metadata}
        except Exception as e:
            logger.warning(f"SQLServer: Failed to parse generated SQL: {e}")
            return ["[ERROR] Could not parse generated SQL", str(e)]

    except Exception as e:
        logger.exception(f"SQLServer: Failed to generate SQL: {e}")
        raise


@mcp.tool()
def execute_sql_query(query: str, session_id: str = "") -> Union[dict, str]:
    """
    Executes a SQL query against the PostgreSQL database.

    Args:
        query: SQL query string to execute.
        session_id: Optional session ID for CSV output file naming.

    Returns:
        dict: {'results': [...], 'csv': filename_or_None}
              or {'error': str} on failure.
    """
    logger.info("SQLServer: execute_sql_query called.")
    query = query.strip().strip(";")
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=SQL_DB_HOST, port=SQL_DB_PORT,
            dbname=SQL_DB_NAME, user=SQL_DB_USER,
            password=SQL_DB_PASSWORD,
        )
        cursor = conn.cursor()
        cursor.execute(query)

        if cursor.description is None:
            return {"results": [], "csv": None}

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]

        csv_file = None
        if session_id:
            save_dir = "results"
            os.makedirs(save_dir, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            csv_file = f"{session_id}_{now}.csv"
            csv_path = os.path.join(save_dir, csv_file)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
            logger.info(f"SQLServer: Results saved to: {csv_path}")

        return {"results": result, "csv": csv_file}

    except Exception as e:
        logger.exception("SQLServer: SQL execution failed.")
        return {"error": str(e)}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@mcp.tool()
def fix_sql_query(
    query: str,
    error_message: str,
    table_schemas: str,
    user_query: str = "",
) -> dict:
    """
    Uses an LLM to fix a broken SQL query given the error, schema, and original query.

    Args:
        query: The erroneous SQL query.
        error_message: The database error returned from the failed execution.
        table_schemas: Table schema details for context.
        user_query: Original natural language query for additional context.

    Returns:
        dict: {'fixed_sql': str, 'tokens': usage_metadata}
              or {'error': str, 'raw_output': str} on parse failure.
    """
    logger.info("SQLServer: fix_sql_query called.")
    try:
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
                num_ctx=config["NLP2SQL_SQL_SERVER_N_CTX"],
            )

        prompt_template = """
You are a helpful SQL expert. Given an erroneous SQL query, the error message, and table schemas, your task is to fix the SQL query.

User Query:
{user_query}

Original Query:
{query}

Error Message:
{error_message}

Table Schemas:
{table_schemas}

Your output format should be {format_instructions}
"""
        parser = PydanticOutputParser(pydantic_object=SQLResponse)
        prompt = PromptTemplate(
            input_variables=["query", "error_message", "table_schemas", "user_query"],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        filled_prompt = prompt.invoke({
            "query": query,
            "error_message": error_message,
            "table_schemas": table_schemas,
            "user_query": user_query,
        })

        logger.debug("SQLServer: Invoking LLM for SQL fix.")
        raw_response = model.invoke(filled_prompt)
        logger.info(f"SQLServer: LLM Token Usage: {raw_response.usage_metadata}")

        try:
            parsed = parser.parse(raw_response.content)
            return {"fixed_sql": parsed.query, "tokens": raw_response.usage_metadata}
        except Exception as parse_err:
            logger.warning(f"SQLServer: Parsing LLM fix output failed: {parse_err}")
            return {"error": "Failed to parse corrected SQL", "raw_output": raw_response.content}

    except Exception as e:
        logger.exception("SQLServer: Error while fixing SQL query.")
        return {"error": f"Failed to fix SQL query: {str(e)}"}
