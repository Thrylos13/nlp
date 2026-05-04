"""
Observability — Logging

Configures and exports the application-wide loguru logger instance
for the NLP2SQL module. Log files are written relative to the project
root where the app is launched.
"""

from loguru import logger

log_file_path = 'logs/digigov_nlp2sql/querygpt_implement.log'
logger.add(log_file_path, rotation="2 MB", backtrace=True, diagnose=True)

logger = logger
