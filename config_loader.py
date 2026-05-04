"""
Config Loader

Loads the NLP2SQL configuration file and sets up environment variables
and system path. Imported by all utils and service modules.
"""

import yaml
import sys
import os

from app.observability.logging import logger

try:
    with open("configs/nlp2sql_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Config Loader: Configuration loaded successfully.")
except FileNotFoundError:
    logger.error("Config Loader: Configuration file not found. Please ensure the path is correct.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Config Loader: Error loading configuration: {e}")
    sys.exit(1)

os.environ["no_proxy"] = config.get("NO_PROXY_ENV", "localhost,127.0.0.1,::1")

project_root = config.get('PROJECT_ROOT')
if project_root and project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"Config Loader: Added '{project_root}' to system path.")
