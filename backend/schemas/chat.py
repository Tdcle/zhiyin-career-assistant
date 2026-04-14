"""Chat API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    jobs: list[dict]


class ChatSessionItem(BaseModel):
    session_id: str
    title: str
    updated_at: str = Field(default="")


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionItem]
    current_session_id: str | None = None


class CreateChatSessionRequest(BaseModel):
    title: str | None = None


class CreateChatSessionResponse(BaseModel):
    session: ChatSessionItem
