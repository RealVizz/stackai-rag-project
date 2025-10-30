from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_STORAGE_RAW_DATA_FOLDER = BASE_DIR / "file_storage" / "raw_data"

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256

VECTOR_DB_FILE_PATH = BASE_DIR / "api_main" / "db" /"vector_store.json"
CHAT_HISTORY_FILE_PATH = BASE_DIR / "api_main" / "db" / "chat_history.json"
KEYWORD_DB_FILE_PATH = BASE_DIR / "api_main" / "db" / "keyword_store.json"