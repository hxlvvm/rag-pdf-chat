import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    docs_dir: str = os.getenv("RAG_DOCS_DIR", "docs")
    db_dir: str = os.getenv("RAG_DB_DIR", ".chroma")
    collection: str = os.getenv("RAG_COLLECTION", "documents")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


settings = Settings()
