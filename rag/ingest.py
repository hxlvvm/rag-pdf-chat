"""Load PDF/TXT/MD files, split them into overlapping chunks and store them in a Chroma vector database."""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_documents(folder: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path.name, "page": i + 1}))
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name, "page": 1}))
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
    return chunks


def build_store(chunks: list[Document], embeddings=None, persist_dir: str | None = None, collection: str = "documents"):
    """Create a Chroma store. With embeddings=None, Chroma's built-in local ONNX model (all-MiniLM-L6-v2) is used."""
    from langchain_chroma import Chroma

    store = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=persist_dir)
    if chunks:
        store.add_documents(chunks, ids=[f"{c.metadata['source']}-{c.metadata['chunk']}" for c in chunks])
    return store


if __name__ == "__main__":
    from .config import settings

    raw = load_documents(settings.docs_dir)
    chunks = split_documents(raw, settings.chunk_size, settings.chunk_overlap)
    build_store(chunks, persist_dir=settings.db_dir, collection=settings.collection)
    print(f"indexed {len(raw)} pages as {len(chunks)} chunks into {settings.db_dir}")
