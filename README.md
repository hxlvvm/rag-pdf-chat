# RAG PDF Chat

Ask questions about your own PDF, text and Markdown files. The app retrieves the most relevant
passages from a **Chroma vector database** and has an LLM answer **only from those passages**, with
source citations. Built to practise retrieval-augmented generation with **LangChain** and **FastAPI**.

## How it works
1. **Ingest** (`rag/ingest.py`): load files, split them into overlapping chunks
   (`RecursiveCharacterTextSplitter`), embed each chunk and store it in Chroma.
   Embeddings come from Chroma's built-in local model (all-MiniLM-L6-v2, ONNX), so no paid API is needed.
2. **Retrieve and answer** (`rag/chain.py`): fetch the top-k chunks for the question, put them in a
   prompt that forbids answering outside the context, and call the LLM (Groq, free tier).
3. **Serve** (`rag/api.py`): `POST /ask {"question": "..."}` returns the answer and its sources.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your GROQ_API_KEY
python -m rag.ingest            # index everything in docs/
uvicorn rag.api:app --reload    # then open http://127.0.0.1:8000/docs
```

## Test (offline)
```bash
pytest -q
```
The tests use deterministic fake embeddings and a fake chat model, so they need no API key and no
model download.

## Next steps
- Hybrid search (BM25 + vectors) and re-ranking
- Evaluation set with retrieval hit-rate and answer faithfulness
- Streaming responses
