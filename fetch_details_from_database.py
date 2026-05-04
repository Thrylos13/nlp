"""
Fetch Details From Database

Provides helper functions for fetching model metadata, app IDs,
and prompt templates from the AI framework database.
Used by both the main service and the MCP server tools.
"""

import psycopg2
import yaml
import sys

from app.observability.logging import logger

try:
    with open("configs/nlp2sql_config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    logger.info("fetch_details_from_database: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"fetch_details_from_database: Failed to load configuration: {e}")
    raise

if config.get('PROJECT_ROOT') and config['PROJECT_ROOT'] not in sys.path:
    sys.path.append(config['PROJECT_ROOT'])

try:
    from src.digigov_database.sql_prompt_db import sql_prompt
    logger.info("fetch_details_from_database: Internal imports loaded successfully.")
except ImportError as e:
    logger.exception(f"fetch_details_from_database: Failed to import sql_prompt_db: {e}")
    raise

DB_HOST = config["AI_SQL_DB_HOSTNAME"]
DB_PORT = config["AI_SQL_DB_PORT"]
DB_USER = config["AI_SQL_DB_USERNAME"]
DB_PASSWORD = config["AI_SQL_DB_PASSWORD"]
DB_NAME = config["AI_SQL_DB_NAME"]


def fetch_model_id(app_id: int):
    """
    Fetches model details for the given app_id from the AI framework DB.

    Args:
        app_id (int): Application identifier.

    Returns:
        tuple: (MODEL_ID, MODEL_NAME, MODEL_HOST, MODEL_PORT, MODEL_TYPE)
               or None if not found.
    """
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        )
        cur = conn.cursor()
        sql = """
            SELECT
                mpg.MODEL_ID,
                mmst.MODEL_NAME,
                mmst.MODEL_HOST,
                mmst.MODEL_PORT,
                mmst.MODEL_TYPE
            FROM
                AI_MODEL_APP_MPG mpg
            JOIN
                AI_MODEL_MST mmst ON mpg.MODEL_ID = mmst.MODEL_ID
            WHERE
                mpg.APP_ID = %s
                AND mpg.ACTIVE_FLAG = TRUE
                AND mmst.ACTIVE_FLAG = TRUE;
        """
        cur.execute(sql, (app_id,))
        result = cur.fetchone()
        logger.info(f"fetch_model_id: Fetched model details for app_id={app_id}: {result}")
        return result
    except Exception as e:
        logger.error(f"fetch_model_id: Failed for app_id={app_id}: {e}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def fetch_app_id_module_id(app_name: str = "NLP2SQL"):
    """
    Fetches the application ID for the given app_name from the AI framework DB.

    Args:
        app_name (str): Application name (default "NLP2SQL").

    Returns:
        tuple: (APP_ID,) or None if not found.
    """
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        )
        cur = conn.cursor()
        sql = """
            SELECT
                amst.APP_ID
            FROM
                AI_APP_MST amst
            JOIN AI_MODEL_APP_MPG mmpg ON amst.APP_ID = mmpg.APP_ID
            WHERE
                amst.APP_NAME = %s
                AND amst.ACTIVE_FLAG = TRUE
                AND mmpg.ACTIVE_FLAG = TRUE;
        """
        cur.execute(sql, (app_name,))
        result = cur.fetchone()
        logger.info(f"fetch_app_id_module_id: Fetched app_id for app_name='{app_name}': {result}")
        return result
    except Exception as e:
        logger.error(f"fetch_app_id_module_id: Failed for app_name='{app_name}': {e}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def fetch_prompt(user_id: int, app_id: int, model_id: int, prompt_identifier: str) -> str:
    """
    Retrieves a prompt template from the AI framework database.

    Args:
        user_id (int): User identifier.
        app_id (int): Application identifier.
        model_id (int): Model identifier.
        prompt_identifier (str): Prompt key name.

    Returns:
        str: Prompt template string.
    """
    try:
        prompt_db = sql_prompt(model_id=model_id, app_id=app_id, user_id=user_id)
        prompt = prompt_db.get_prompt(prompt_identifier)
        logger.info(f"fetch_prompt: Retrieved prompt '{prompt_identifier}'.")
        return prompt
    except Exception as e:
        logger.error(f"fetch_prompt: Failed to fetch prompt '{prompt_identifier}': {e}")
        raise
