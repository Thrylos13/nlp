"""
Audit Repository

Centralises all audit queue logging and AuditDB ID generation for the
NLP2SQL module. All send_to_queue calls and AuditDB interactions live here.
"""

import yaml
import sys
import os

from app.observability.logging import logger

try:
    with open("configs/nlp2sql_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info("AuditRepository: Configuration loaded successfully.")
except Exception as e:
    logger.error(f"AuditRepository: Failed to load configuration: {e}")
    raise

if config.get('PROJECT_ROOT') and config['PROJECT_ROOT'] not in sys.path:
    sys.path.append(config['PROJECT_ROOT'])

try:
    from src.digigov_database.audit_db import AuditDB
    from artemis_queues import send_to_queue
    logger.info("AuditRepository: Internal imports loaded successfully.")
except ImportError as e:
    logger.exception(f"AuditRepository: Failed to import internal modules: {e}")
    raise


class AuditRepository:
    """
    Handles all audit trail persistence — unique ID generation
    and audit event queue logging — for the NLP2SQL module.
    """

    def __init__(self):
        try:
            self.audit_db = AuditDB()
            logger.info("AuditRepository: Initialised successfully.")
        except Exception as e:
            logger.error(f"AuditRepository: Initialisation failed: {e}")
            raise

    def generate_request_id(self) -> str:
        """Generates and returns a unique request master ID."""
        try:
            request_id = self.audit_db.generate_unique_id('REQUEST_MST')
            logger.info(f"AuditRepository: Generated request ID: {request_id}")
            return request_id
        except Exception as e:
            logger.error(f"AuditRepository: Failed to generate request ID: {e}")
            raise

    def generate_response_id(self) -> str:
        """Generates and returns a unique response master ID."""
        try:
            response_id = self.audit_db.generate_unique_id('RESPONSE_MST')
            logger.info(f"AuditRepository: Generated response ID: {response_id}")
            return response_id
        except Exception as e:
            logger.error(f"AuditRepository: Failed to generate response ID: {e}")
            raise

    def log_event(self, op: str, payload: dict):
        """
        Sends a formatted payload to the audit queue.

        Args:
            op (str): Queue operation name (e.g. 'log_request_mst').
            payload (dict): Data payload to enqueue.
        """
        try:
            send_to_queue.send({'op': op, 'payload': payload})
            logger.debug(f"AuditRepository: Logged audit event '{op}' successfully.")
        except Exception as e:
            logger.error(f"AuditRepository: Failed to log audit event '{op}': {e}")
