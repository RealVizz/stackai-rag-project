from fastapi import Form
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)
    query_str: str = Field(..., min_length=1)


class HistoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)


class UploadedFilesRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)


class ChatsRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class UploadForm:
    def __init__(self, user_id: str = Form(..., min_length=1), chat_id: str = Form(..., min_length=1)):
        self.user_id = user_id
        self.chat_id = chat_id
