import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE_PATH)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY not found. "
        "Make sure it is set in your .env file or as an environment variable."
    )

BASE_STORAGE_RAW_DATA_FOLDER = BASE_DIR / "file_storage" / "raw_data"

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256

VECTOR_DB_FILE_PATH = BASE_DIR / "api_main" / "db" / "vector_store.json"
CHAT_HISTORY_FILE_PATH = BASE_DIR / "api_main" / "db" / "chat_history.json"
KEYWORD_DB_FILE_PATH = BASE_DIR / "api_main" / "db" / "keyword_store.json"
