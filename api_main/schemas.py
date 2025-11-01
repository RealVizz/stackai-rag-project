from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)
    query_str: str = Field(..., min_length=1)


class HistoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)


class ChatsRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
