from fastapi import FastAPI, Form, UploadFile, File, HTTPException
import uuid

from config.config import BASE_STORAGE_RAW_DATA_FOLDER
import shutil

from api_main.utils.pdf_helper import process_pdf
from api_main.utils.mistral_helper import get_embeddings_from_str_list
from api_main.utils.vector_db_helper import add_embeddings

app = FastAPI()


@app.get("/")
async def heartbeat():
    return {"status": "ok"}


@app.post("/upload-pdf/")
async def upload_pdf_file(user_id: str = Form(...), chat_id: str = Form(...), file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type, Only '.pdf' files are accepted."
        )

    if not chat_id.strip() or not user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Both 'chat_id' and 'user_id' are required and cannot be empty."
        )

    try:
        file_uuid = str(uuid.uuid4())
        new_filename = f"{file_uuid}.pdf"

        storage_path = BASE_STORAGE_RAW_DATA_FOLDER / f"user_id_{user_id}" / f"chat_id_{chat_id}"
        storage_path.mkdir(parents=True, exist_ok=True)
        full_file_path = storage_path / new_filename

        with full_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text_chunks = process_pdf(full_file_path)
        embeddings = get_embeddings_from_str_list(text_chunks)
        add_embeddings(user_id, chat_id, file_uuid, text_chunks, embeddings)

    finally:
        await file.close()

    return {
        "status": "success",
        "message": "File uploaded successfully.",
    }
