import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence the fork warning

import tempfile
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
import gradio as gr
from backend.state import app_state
from qdrant_client import QdrantClient

# Assume you update these functions to accept a 'qdrant_client' argument
from backend.pipeline import run_pipeline
from backend.ingest import ingest_pdf
from backend.ui import demo
from config import QDRANT_PATH


# ─── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting app: Initializing Qdrant...")
    # 1. Initialize Qdrant ONLY here.
    app_state["qdrant"] = QdrantClient(path=str(QDRANT_PATH))
    
    yield # App runs here
    
    print("Shutting down: Closing database connections...")
    # 2. Explicitly close Qdrant here.
    if "qdrant" in app_state:
        app_state["qdrant"].close()
        print("Qdrant closed cleanly.")
        
    await asyncio.sleep(1) # Allow Gradio background tasks a moment to halt

# ─── App Initialization ───────────────────────────────────────────────
app = FastAPI(
    title="cited-rx",
    description="Cited RAG over Medical Guidelines",
    version="0.2.0",
    lifespan=lifespan
)

# ─── Request/response models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    threshold: float = 0.0
    corpus_id: str = "cited_rx_chunks"

class CitationWithPage(BaseModel):
    chunk_id: int
    page_number: int
    quote: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[CitationWithPage]
    refused: bool

class UploadResponse(BaseModel):
    corpus_id: str
    n_chunks: int
    n_pages: int
    source_doc: str

# ─── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "cited-rx", "version": "0.2.0"}

@app.post("/query/grounded", response_model=QueryResponse)
def query_grounded(req: QueryRequest) -> QueryResponse:
    client = app_state["qdrant"]

    result = run_pipeline(
        req.question,
        qdrant_client=client,
        top_k=req.top_k,
        threshold=req.threshold,
        corpus_id=req.corpus_id,
    )

    citations_with_pages = [
        CitationWithPage(
            chunk_id=c.chunk_id,
            page_number=c.page_number,
            quote=c.quote,
        )
        for c in result.citations
    ]

    return QueryResponse(
        answer=result.rendered_answer,
        confidence=result.confidence,
        citations=citations_with_pages,
        refused=result.refused,
    )

@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Got: {file.filename}")

    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Retrieve the active client
        client = app_state["qdrant"]
        
        # Pass the client to your ingest function
        result = ingest_pdf(tmp_path, source_doc=file.filename, qdrant_client=client) # <--- Pass it down
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        corpus_id=result["corpus_id"],
        n_chunks=result["n_chunks"],
        n_pages=result["n_pages"],
        source_doc=file.filename,
    )

# ─── Gradio Mount ─────────────────────────────────────────────────────

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)