
import os
from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Data paths (override via env vars for any deployment) ─────────────
DATA_DIR      = Path(os.getenv("CITED_RX_DATA_DIR",  str(ROOT / "data")))
QDRANT_PATH   = Path(os.getenv("CITED_RX_QDRANT",    str(DATA_DIR / "qdrant_storage")))
PROCESSED_DIR = Path(os.getenv("CITED_RX_PROCESSED", str(DATA_DIR / "processed")))
RAW_DIR       = Path(os.getenv("CITED_RX_RAW",       str(DATA_DIR / "raw")))
EVAL_DIR      = Path(os.getenv("CITED_RX_EVAL",      str(DATA_DIR / "eval")))

# ── LLM ────────────────────────────────────────────────────────────────

LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq")
GROQ_MODEL      = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
GROQ_EVAL_MODEL  = os.getenv("GROQ_EVAL_MODEL", "llama3-8b-8192")

# ── Retrieval / ingestion constants ────────────────────────────────────
DEFAULT_CORPUS = "cited_rx_chunks"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
EMBEDDING_DIM  = 1024
RRF_K          = 60

# ── Create local directories on first run ─────────────────────────────
for _d in (DATA_DIR, QDRANT_PATH, PROCESSED_DIR, RAW_DIR, EVAL_DIR, EVAL_DIR / "runs"):
    _d.mkdir(parents=True, exist_ok=True)
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"
