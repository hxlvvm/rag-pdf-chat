from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .chain import RagPipeline, default_llm
from .config import settings
from .ingest import build_store

app = FastAPI(title="RAG PDF Chat")


class Question(BaseModel):
    question: str = Field(min_length=3)


@lru_cache
def pipeline() -> RagPipeline:
    store = build_store([], persist_dir=settings.db_dir, collection=settings.collection)
    return RagPipeline(store, default_llm(settings.groq_model), top_k=settings.top_k)


@app.post("/ask")
def ask(q: Question) -> dict:
    result = pipeline().ask(q.question)
    return {"answer": result.answer, "sources": result.sources}
