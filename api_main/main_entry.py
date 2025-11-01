import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api_main.schemas import QueryRequest, HistoryRequest, ChatsRequest
from api_main.services.rag_service import process_pdf_upload, process_query
from api_main.utils.chat_memory_helper import (
    load_chat_from_persistent_storage,
    get_chat_history,
    get_all_user_ids,
    get_all_chat_ids_for_user
)
from api_main.utils.keyword_db_helper import (
    load_keyword_db_from_persistent_storage
)
from api_main.utils.pdf_helper import PDFProcessingError
from api_main.utils.vector_db_helper import (
    load_vector_db_from_persistent_storage
)
from config.config import BASE_STORAGE_RAW_DATA_FOLDER


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vector_db_from_persistent_storage()
    load_chat_from_persistent_storage()
    load_keyword_db_from_persistent_storage()
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

    full_file_path = None
    try:
        file_uuid = str(uuid.uuid4())
        new_filename = f"{file_uuid}.pdf"

        storage_path = BASE_STORAGE_RAW_DATA_FOLDER / f"user_id_{user_id}" / f"chat_id_{chat_id}"
        storage_path.mkdir(parents=True, exist_ok=True)
        full_file_path = storage_path / new_filename

        with full_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        message = process_pdf_upload(user_id=user_id, chat_id=chat_id, file=file, file_uuid=file_uuid,
                                     full_file_path=full_file_path)

        return {"status": "success", "message": message}

    except PDFProcessingError as e:
        print(f"PDF processing failed for user {user_id}, chat {chat_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        print(f"Service layer error during upload for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise  # Re-raise HTTPExceptions

    except Exception as e:
        print(f"Unexpected error in /upload-pdf/ for user {user_id}: {e}")
        if full_file_path and full_file_path.exists():
            full_file_path.unlink()  # File cleanup.
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

    finally:
        await file.close()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"message": "Invalid input. Please provide all required fields."}, )


@app.post("/query/")
async def query(request: QueryRequest):
    try:
        response_text = process_query(user_id=request.user_id, chat_id=request.chat_id, query_str=request.query_str)
        return {"status": "success", "resp": response_text}

    except ValueError as e:
        print(f"Service layer error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        print(f"Unexpected error in /query/: {e}")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")


@app.post("/get-chat-history/")
async def get_chat_history_endpoint(request: HistoryRequest):
    try:
        history = get_chat_history(request.user_id, request.chat_id)
        return {
            "status": "success",
            "history": history
        }
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve chat history.")


@app.get("/users/")
async def get_users():
    try:
        user_ids = get_all_user_ids()
        return {
            "status": "success",
            "user_ids": user_ids
        }
    except Exception as e:
        print(f"Error retrieving user list: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve user list."
        )


@app.post("/chats/")
async def get_chats_for_user(request: ChatsRequest):

    try:
        chat_ids = get_all_chat_ids_for_user(request.user_id)

        return {
            "status": "success",
            "user_id": request.user_id,
            "chat_ids": chat_ids
        }
    except Exception as e:
        print(f"Error retrieving chat list: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve chat list."
        )
