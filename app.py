"""FastAPI app: serves the BudduBro site and the /chat endpoint used by the chat widget.

Run with:  uv run uvicorn ecommerce_chatbot.app:app --reload
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import RAG, config

app = FastAPI(title="BudduBro Chatbot")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Message] = []


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.SITE_FILE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        reply = RAG.answer(req.message, [m.model_dump() for m in req.history])
    except Exception as exc:  # surface config problems (e.g. missing API key) clearly
        raise HTTPException(status_code=503, detail=f"Chat service unavailable: {exc}") from exc
    return ChatResponse(reply=reply)
