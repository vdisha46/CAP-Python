# Accounting RAG Assistant

A production-oriented accounting dashboard with a vanilla HTML/CSS/Bootstrap frontend and FastAPI + SQLite backend.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. The dashboard includes seeded demo data. Add `GEMINI_API_KEY` to `.env` to enable Gemini 2.5 Flash responses; without it, a deterministic policy-aware fallback keeps the app usable.

The custom retrieval layer stores uploaded document text and ranks policy documents by token overlap. `sentence-transformers/all-MiniLM-L6-v2` is included in the dependency set for upgrading this local retriever to vector similarity in a deployment environment.
