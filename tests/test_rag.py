"""Offline tests: deterministic fake embeddings and a fake chat model, so no API key or model download is needed."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import RagPipeline, format_context
from rag.ingest import build_store, load_documents, split_documents


def test_load_and_split(tmp_path):
    (tmp_path / "notes.txt").write_text("FastAPI is a Python web framework. " * 60)
    docs = load_documents(str(tmp_path))
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(docs) == 1 and len(chunks) > 5
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert chunks[0].metadata["source"] == "notes.txt"


def test_retrieval_returns_exact_match():
    texts = ["Chroma is a vector database.", "Bananas are yellow.", "LangGraph builds agent graphs."]
    chunks = [Document(page_content=t, metadata={"source": "kb.txt", "page": 1, "chunk": i}) for i, t in enumerate(texts)]
    store = build_store(chunks, embeddings=DeterministicFakeEmbedding(size=64), collection="test-retrieval")
    hit = store.similarity_search("Chroma is a vector database.", k=1)[0]
    assert hit.page_content == texts[0]


def test_pipeline_returns_answer_and_sources():
    chunks = [Document(page_content="The office opens at 10:30.", metadata={"source": "hr.pdf", "page": 2, "chunk": 0})]
    store = build_store(chunks, embeddings=DeterministicFakeEmbedding(size=64), collection="test-pipeline")
    rag = RagPipeline(store, FakeListChatModel(responses=["It opens at 10:30 [hr.pdf p.2]."]), top_k=1)
    out = rag.ask("When does the office open?")
    assert "10:30" in out.answer
    assert out.sources == [{"source": "hr.pdf", "page": 2}]
    assert "[hr.pdf p.2]" in format_context(store.similarity_search("office", k=1))
