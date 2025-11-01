from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api_main.schemas import QueryRequest, HistoryRequest, ChatsRequest, UploadForm
from api_main.services.rag_service import process_pdf_upload, process_query
from api_main.utils.chat_memory_helper import (
    load_chat_from_persistent_storage, get_chat_history, get_all_user_ids, get_all_chat_ids_for_user
)
from api_main.utils.keyword_db_helper import load_keyword_db_from_persistent_storage
from api_main.utils.pdf_helper import PDFProcessingError
from api_main.utils.vector_db_helper import load_vector_db_from_persistent_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vector_db_from_persistent_storage()
    load_chat_from_persistent_storage()
    load_keyword_db_from_persistent_storage()
    yield


app = FastAPI(lifespan=lifespan)


# --- Global exception handlers to keep endpoints clean ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"message": "Invalid input. Please provide all required fields."})


@app.exception_handler(PDFProcessingError)
async def pdf_processing_exception_handler(request: Request, exc: PDFProcessingError):
    print(f"PDF processing failed: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    print(f"Service layer value error: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/")
async def heartbeat():
    return {"status": "ok"}


@app.post("/upload-pdf/")
async def upload_pdf_file(form_data: UploadForm = Depends(), file: UploadFile = File(...)):
    try:
        message = process_pdf_upload(user_id=form_data.user_id, chat_id=form_data.chat_id, file=file)
        return {"status": "success", "message": message}

    except Exception as e:
        print(f"Unexpected error in /upload-pdf/: {e}")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

    finally:
        await file.close()


@app.post("/query/")
async def query(request: QueryRequest):
    response_text = process_query(user_id=request.user_id, chat_id=request.chat_id, query_str=request.query_str)
    return {"status": "success", "resp": response_text}


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
