from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_STORAGE_RAW_DATA_FOLDER = BASE_DIR / "file_storage" / "raw_data"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

VECTOR_DB_FILE_PATH = BASE_DIR / "api_main" / "db" /"vector_store.json"