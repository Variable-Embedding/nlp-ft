"""A list of constants used for the workflow.
"""
from os.path import dirname, join

WORKFLOW_ROOT = dirname(dirname(dirname(__file__)))
LOGGING_PATH = join(WORKFLOW_ROOT, "logs")
OUTPUT_PATH = join(WORKFLOW_ROOT, "output")
TMP_PATH = join(WORKFLOW_ROOT, "tmp")
DATA_PATH = join(WORKFLOW_ROOT, "data")
SQL_SCRIPTS_PATH = join(WORKFLOW_ROOT, "sql_scripts")
CONFIG_PATH = join(WORKFLOW_ROOT, "configs")
EMBEDDINGS_PATH = join(WORKFLOW_ROOT, "embeddings")

WIKIDATA_URL = "https://query.wikidata.org/sparql"
