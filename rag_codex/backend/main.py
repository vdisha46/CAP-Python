from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "frontend"
DB_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "accounting.db"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
_embedding_model = None

app = FastAPI(title="Accounting RAG Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def seed_database() -> None:
    with closing(connect()) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS vendors (id INTEGER PRIMARY KEY, invoice_number TEXT, vendor_name TEXT, invoice_date TEXT, due_date TEXT, amount REAL, status TEXT, payment_status TEXT);
            CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, invoice_number TEXT, customer_name TEXT, invoice_date TEXT, due_date TEXT, amount REAL, status TEXT, receipt_status TEXT);
            CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, filename TEXT UNIQUE, content TEXT, uploaded_at TEXT, chunk_count INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, sources TEXT, confidence INTEGER, created_at TEXT);
            """
        )
        if db.execute("SELECT COUNT(*) FROM vendors").fetchone()[0] == 0:
            db.executemany("INSERT INTO vendors VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                (1, "VIN-2026-0041", "Northstar Cloud Services", "2026-08-02", "2026-09-01", 185000, "Approved", "Pending"),
                (2, "VIN-2026-0042", "Apex Office Systems", "2026-08-14", "2026-09-14", 78500, "Approved", "Pending"),
                (3, "VIN-2026-0037", "Meridian Logistics", "2026-07-18", "2026-08-17", 246000, "Approved", "Overdue"),
                (4, "VIN-2026-0046", "BluePeak Consulting", "2026-08-26", "2026-09-25", 425000, "In Review", "Pending"),
                (5, "VIN-2026-0031", "Vertex Utilities", "2026-06-30", "2026-07-30", 31500, "Approved", "Paid"),
                (6, "VIN-2026-0049", "Cobalt Security", "2026-09-05", "2026-10-05", 92000, "Approved", "Pending"),
            ])
        if db.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            db.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                (1, "CIN-2026-0118", "Acme Retail Group", "2026-08-03", "2026-09-02", 310000, "Issued", "Pending"),
                (2, "CIN-2026-0122", "Kiteworks India", "2026-08-17", "2026-09-16", 127500, "Issued", "Pending"),
                (3, "CIN-2026-0102", "Olive Tree Foods", "2026-07-11", "2026-08-10", 88000, "Issued", "Overdue"),
                (4, "CIN-2026-0128", "Summit Health Partners", "2026-08-28", "2026-09-27", 265000, "Issued", "Pending"),
                (5, "CIN-2026-0098", "Pioneer Education", "2026-06-25", "2026-07-25", 145000, "Issued", "Received"),
                (6, "CIN-2026-0131", "Lumen Mobility", "2026-09-09", "2026-10-09", 99500, "Draft", "Pending"),
            ])
        if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0:
            policies = [
                ("payment_clearance_policy.md", "Payment Clearance Policy PC-004\nPayments above INR 500000 require CFO approval. Payments from INR 100000 to INR 500000 require Finance Controller approval. Payments below INR 100000 may be approved by the AP Manager. Two-way matching is required before payment release."),
                ("receipt_clearance_policy.md", "Receipt Clearance Policy RC-002\nCustomer receipts are reconciled by Accounts Receivable. Receipts more than 30 days overdue must be escalated to the Collections Lead and Finance Controller."),
            ]
            for filename, content in policies:
                db.execute("INSERT INTO documents (filename, content, uploaded_at, chunk_count) VALUES (?, ?, ?, ?)", (filename, content, datetime.now().isoformat(), 1))
        db.commit()


def rows(table: str) -> list[dict[str, Any]]:
    with closing(connect()) as db:
        return [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY id DESC")]


def retrieve(question: str, limit: int = 4) -> list[dict[str, str]]:
    terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    with closing(connect()) as db:
        documents = [dict(row) for row in db.execute("SELECT filename, content FROM documents")]
    scored = []
    try:
        global _embedding_model
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        query_vector = _embedding_model.encode([question], normalize_embeddings=True)[0]
        document_vectors = _embedding_model.encode([document["content"] for document in documents], normalize_embeddings=True)
        scored = [(float(query_vector @ vector), document) for document, vector in zip(documents, document_vectors)]
    except Exception:
        # Keep the local demo usable when ML dependencies or model weights are unavailable.
        scored = []
    for document in documents:
        if not scored:
            tokens = set(re.findall(r"[a-z0-9]{3,}", document["content"].lower()))
            scored.append((len(terms & tokens), document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"filename": item["filename"], "content": item["content"]} for score, item in scored[:limit] if score > 0] or [{"filename": item["filename"], "content": item["content"]} for item in documents[:limit]]


def local_answer(question: str, sources: list[dict[str, str]]) -> tuple[str, int]:
    normalized = question.lower()
    if "500000" in normalized or "500,000" in normalized or "approve" in normalized:
        return "According to Payment Clearance Policy PC-004, payments above INR 500000 require CFO approval. Payments from INR 100000 to INR 500000 require Finance Controller approval.", 95
    if "receipt" in normalized and "overdue" in normalized:
        return "Receipt Clearance Policy RC-002 requires customer receipts more than 30 days overdue to be escalated to the Collections Lead and Finance Controller.", 92
    if "pending payment" in normalized or "vendor payment" in normalized:
        return "There are pending vendor payments in the invoice register. Review the Vendor Invoices view for invoice-level due dates, approval status, and payment status.", 86
    return "I found relevant accounting records, but I need a more specific question to provide a policy-backed answer. Try asking about an approval threshold, overdue receipt, or pending payment.", 72


async def gemini_answer(question: str, context: str) -> tuple[str, int] | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"You are an accounting assistant. Answer only from the context. Be concise and mention when the context is insufficient.\n\nContext:\n{context}\n\nQuestion: {question}"
        response = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt)
        return response.text, 94
    except Exception:
        return None


@app.on_event("startup")
def startup() -> None:
    seed_database()

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    vendors, customers = rows("vendors"), rows("customers")
    return {"kpis": {"total_vendor_invoices": len(vendors), "total_customer_invoices": len(customers), "pending_vendor_payments": sum(v["payment_status"] == "Pending" for v in vendors), "pending_customer_receipts": sum(c["receipt_status"] == "Pending" for c in customers), "overdue_vendor_invoices": sum(v["payment_status"] == "Overdue" for v in vendors), "overdue_customer_invoices": sum(c["receipt_status"] == "Overdue" for c in customers), "outstanding_vendor_amount": sum(v["amount"] for v in vendors if v["payment_status"] != "Paid"), "outstanding_customer_amount": sum(c["amount"] for c in customers if c["receipt_status"] != "Received")}, "trend": [{"label": "Apr", "vendor": 420, "customer": 510}, {"label": "May", "vendor": 580, "customer": 640}, {"label": "Jun", "vendor": 490, "customer": 710}, {"label": "Jul", "vendor": 720, "customer": 680}, {"label": "Aug", "vendor": 650, "customer": 820}, {"label": "Sep", "vendor": 790, "customer": 900}]}

@app.get("/api/vendors")
def vendors() -> list[dict[str, Any]]:
    return rows("vendors")

@app.get("/api/customers")
def customers() -> list[dict[str, Any]]:
    return rows("customers")

@app.get("/api/history")
def history() -> list[dict[str, Any]]:
    with closing(connect()) as db:
        return [dict(row) for row in db.execute("SELECT * FROM history ORDER BY id DESC LIMIT 20")]

@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    sources = retrieve(request.question)
    context = "\n\n".join(f"[{source['filename']}]\n{source['content']}" for source in sources)
    generated = await gemini_answer(request.question, context)
    answer, confidence = generated or local_answer(request.question, sources)
    with closing(connect()) as db:
        db.execute("INSERT INTO history (question, answer, sources, confidence, created_at) VALUES (?, ?, ?, ?, ?)", (request.question, answer, json.dumps([source["filename"] for source in sources]), confidence, datetime.now().isoformat()))
        db.commit()
    return {"answer": answer, "sources": [source["filename"] for source in sources], "confidence": confidence}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Supported files: PDF, DOCX, TXT, MD, CSV")
    data = await file.read()
    content = ""
    if suffix in {".txt", ".md", ".csv"}:
        content = data.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
            content = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        except Exception as error:
            raise HTTPException(400, f"Could not parse PDF: {error}") from error
    elif suffix == ".docx":
        try:
            from docx import Document
            content = "\n".join(paragraph.text for paragraph in Document(io.BytesIO(data)).paragraphs)
        except Exception as error:
            raise HTTPException(400, f"Could not parse DOCX: {error}") from error
    if not content.strip():
        raise HTTPException(400, "The uploaded document contains no readable text")
    chunks = [content[index:index + 1200] for index in range(0, len(content), 1200)]
    with closing(connect()) as db:
        db.execute("INSERT INTO documents (filename, content, uploaded_at, chunk_count) VALUES (?, ?, ?, ?) ON CONFLICT(filename) DO UPDATE SET content=excluded.content, uploaded_at=excluded.uploaded_at, chunk_count=excluded.chunk_count", (file.filename, content, datetime.now().isoformat(), len(chunks)))
        db.commit()
    return {"filename": file.filename, "chunks": len(chunks), "message": "Document uploaded and indexed"}

@app.post("/api/reindex")
def reindex() -> dict[str, Any]:
    with closing(connect()) as db:
        documents = db.execute("SELECT id, content FROM documents").fetchall()
        for document in documents:
            chunk_count = max(1, (len(document["content"]) + 1199) // 1200)
            db.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, document["id"]))
        db.commit()
    return {"documents": len(documents), "message": "Knowledge base reindexed successfully"}

@app.get("/api/export/{kind}")
def export_csv(kind: str) -> StreamingResponse:
    if kind not in {"vendors", "customers"}:
        raise HTTPException(404, "Unknown export type")
    records = rows(kind)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=records[0].keys() if records else [])
    writer.writeheader(); writer.writerows(records); buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={kind}.csv"})

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
