"""
ingest.py — corpus-aware PDF ingestion pipeline.

Single source of truth for chunking, embedding, and indexing. Used by:
  - CLI wrappers (read_pdf.py, embed_chunks.py, index.py) for the bundled cardiology corpus
  - The /upload API endpoint, which calls ingest_pdf() directly

Pipeline:
  PDF → chunks (RecursiveCharacterTextSplitter, 1000/200 with overlap)
      → embeddings (BGE-M3, 1024-dim, normalized)
      → Qdrant collection named {corpus_id}
"""
import json
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import VectorParams, Distance, PointStruct
from backend.retrieve import _get_model


from config import PROCESSED_DIR, DEFAULT_CORPUS, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIM


def chunks_path(corpus_id: str) -> Path:
    return PROCESSED_DIR / f"{corpus_id}_chunks.json"
def embeddings_path(corpus_id: str) -> Path:
    return PROCESSED_DIR / f"{corpus_id}_embeddings.npy"

def chunk_pdf(pdf_path: str, source_doc: Optional[str] = None) -> list[dict]:
    """Read PDF, clean per-page text, split into overlapping chunks with metadata.

    Returns list of {chunk_id, source_doc, page_number, text} dicts.
    """
    pdf_path = Path(pdf_path)
    if source_doc is None:
        source_doc = pdf_path.name

    reader = PdfReader(str(pdf_path))

    # Per-page extraction + structure-preserving cleanup
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)     
        text = re.sub(r"\n{3,}", "\n\n", text)   
        text = text.strip()
        if text:
            pages.append({"page_number": page_num, "text": text})

    if not pages:
        raise ValueError(
            f"PDF at {pdf_path} has no extractable text. "
            f"Likely a scanned/image-only PDF — OCR would be required."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    chunk_id = 0
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({
                "chunk_id": chunk_id,
                "source_doc": source_doc,
                "page_number": page["page_number"],
                "text": piece,
            })
            chunk_id += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed chunk texts in chunk_id order. Returns (N, 1024) normalized float array."""
    chunks_sorted = sorted(chunks, key=lambda c: c["chunk_id"])
    texts = [c["text"] for c in chunks_sorted]

    embeddings = _get_model().encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True, 
        convert_to_numpy=True,
    )
    return embeddings



def index_chunks(corpus_id: str, chunks: list[dict], embeddings: np.ndarray,qdrant_client) -> None:
    """Create (or recreate) Qdrant collection named {corpus_id} and upload all points.

    Idempotent: existing collection with same name is dropped before rebuild.
    """
    chunks_sorted = sorted(chunks, key=lambda c: c["chunk_id"])
    assert len(chunks_sorted) == embeddings.shape[0], (
        f"Chunks ({len(chunks_sorted)}) and embeddings "
        f"({embeddings.shape[0]}) count mismatch"
    )

    if qdrant_client.collection_exists(corpus_id):
        qdrant_client.delete_collection(corpus_id)

    qdrant_client.create_collection(
        collection_name=corpus_id,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=embeddings[i].tolist(),
            payload={
                "chunk_id": chunk["chunk_id"],
                "source_doc": chunk["source_doc"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            },
        )
        for i, chunk in enumerate(chunks_sorted)
    ]
    qdrant_client.upsert(collection_name=corpus_id, points=points)


def ingest_pdf(
    pdf_path: str,
    qdrant_client,
    corpus_id: Optional[str] = None,
    source_doc: Optional[str] = None,
) -> dict:
    """End-to-end ingestion: PDF → chunks → embeddings → Qdrant collection.

    Args:
        pdf_path: Path to source PDF (may be a tempfile).
        corpus_id: Identifier (and Qdrant collection name). If None, generates
            a UUID-based id like 'user_a1b2c3d4'.
        source_doc: Display name for the source document, embedded in each
            chunk's metadata. If None, derived from pdf_path's basename. Pass
            the original upload filename here when ingesting from a tempfile.

    Returns:
        {"corpus_id": str, "n_chunks": int, "n_pages": int}
    """
    if corpus_id is None:
        corpus_id = f"user_{uuid.uuid4().hex[:8]}"

    pdf_name = source_doc or Path(pdf_path).name
    print(f"\n[ingest] corpus_id='{corpus_id}' source='{pdf_name}'")

    # Stage 1
    chunks = chunk_pdf(pdf_path, source_doc=pdf_name)
    n_pages = len({c["page_number"] for c in chunks})
    print(f"[ingest] Chunked: {len(chunks)} chunks across {n_pages} pages")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(chunks_path(corpus_id), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Saved chunks → {chunks_path(corpus_id)}")

    # Stage 2
    embeddings = embed_chunks(chunks)
    print(f"[ingest] Embedded: shape {embeddings.shape}")

    np.save(embeddings_path(corpus_id), embeddings)
    print(f"[ingest] Saved embeddings → {embeddings_path(corpus_id)}")

    # Stage 3
    index_chunks(corpus_id, chunks, embeddings,qdrant_client)
    point_count = qdrant_client.count(collection_name=corpus_id).count
    print(f"[ingest] Indexed: {point_count} points in collection '{corpus_id}'")

    return {
        "corpus_id": corpus_id,
        "n_chunks": len(chunks),
        "n_pages": n_pages,
    }

if __name__ == "__main__":
    import argparse
    from qdrant_client import QdrantClient
    parser = argparse.ArgumentParser(description="Ingest a PDF end-to-end.")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--corpus", default=None,
                        help="Corpus id (default: auto-generated UUID-based id)")
    args = parser.parse_args()
    from config import QDRANT_PATH
    test_client = QdrantClient(path=str(QDRANT_PATH))
    result = ingest_pdf(args.pdf,qdrant_client=test_client,corpus_id=args.corpus)
    print(f"\n[ingest] Done: {result}")