from contextlib import asynccontextmanager
import shutil
import uuid

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request

from config.config import BASE_STORAGE_RAW_DATA_FOLDER
from api_main.utils.pdf_helper import process_pdf
from api_main.utils.mistral_helper import get_embeddings_from_str_list, get_embedding_from_str, get_llm_response
from api_main.utils.vector_db_helper import (
    add_embeddings,
    load_vector_db_from_persistent_storage,
    get_top_k_vector_results
)
from api_main.utils.chat_memory_helper import (
    load_chat_from_persistent_storage,
    get_chat_history,
    add_chat_turn
)

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vector_db_from_persistent_storage()
    load_chat_from_persistent_storage()
    yield
app = FastAPI(lifespan=lifespan)

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
        add_embeddings(user_id, chat_id, file_uuid, file.filename, text_chunks, embeddings)

    finally:
        await file.close()

    return {
        "status": "success",
        "message": "File uploaded successfully.",
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid input. Please provide all required fields.",
            # "errors": exc.errors()  # Include validation error details
        },
    )

@app.post("/query/")
async def query(user_id: str = Form(...), chat_id: str = Form(...), query_str: str = Form(...)):
    if not chat_id.strip() or not user_id.strip() or not query_str.strip():
        raise HTTPException(
            status_code=400,
            detail="All 'chat_id', 'user_id' and 'query_str' are required and cannot be empty."
        )

    query_embeddings = get_embedding_from_str(query_str)
    if not query_embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    vector_res = get_top_k_vector_results(user_id, chat_id, query_embeddings, k=5, threshold=0.5)
    current_chat_history = get_chat_history(user_id, chat_id)
    resp = get_llm_response(user_query=query_str, vector_results=vector_res, keyword_results=[],
                            chat_history=current_chat_history, max_tokens=8192)
    add_chat_turn(user_id, chat_id, query_str, resp)
    return {
        "status": "success",
        "resp": resp
    }