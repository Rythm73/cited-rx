import json
import re
from rank_bm25 import BM25Okapi
from backend.schemas import RetrievedChunk

from config import PROCESSED_DIR, DEFAULT_CORPUS
CHUNKS_DIR = str(PROCESSED_DIR)

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip most punctuation, keep hyphens, whitespace-split.
    Keeps 'LDL-C' as one token; splits 'atorvastatin,' from its trailing comma."""
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    return text.split()


def _chunks_path(corpus_id: str) -> str:
    return f"{CHUNKS_DIR}/{corpus_id}_chunks.json"


# Per-corpus caches — built lazily on first access
_chunks_cache: dict[str, list[dict]] = {}
_bm25_indexes: dict[str, BM25Okapi] = {}


def _get_chunks(corpus_id: str) -> list[dict]:
    if corpus_id not in _chunks_cache:
        with open(_chunks_path(corpus_id), "r", encoding="utf-8") as f:
            _chunks_cache[corpus_id] = sorted(json.load(f), key=lambda c: c["chunk_id"])
    return _chunks_cache[corpus_id]


def get_bm25(corpus_id: str) -> BM25Okapi:
    """Build (on first call) and cache a BM25 index for the given corpus."""
    if corpus_id not in _bm25_indexes:
        chunks = _get_chunks(corpus_id)
        tokenized = [_tokenize(c["text"]) for c in chunks]
        _bm25_indexes[corpus_id] = BM25Okapi(tokenized)
        print(f"[retrieve_bm25] Built BM25 index for corpus '{corpus_id}' "
              f"({len(chunks)} chunks, {sum(len(t) for t in tokenized)} tokens)")
    return _bm25_indexes[corpus_id]


def retrieve_bm25(
    query: str,
    top_k: int = 5,
    corpus_id: str = DEFAULT_CORPUS,
) -> list[RetrievedChunk]:
    """Score every chunk by BM25 against the query, return top-k."""
    bm25 = get_bm25(corpus_id)
    chunks = _get_chunks(corpus_id)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            chunk_id=chunks[i]["chunk_id"],
            score=float(scores[i]),
            page_number=chunks[i]["page_number"],
            source_doc=chunks[i]["source_doc"],
            text=chunks[i]["text"],
        )
        for i in top_k_indices
    ]


if __name__ == "__main__":
    test_queries = [
        "What are performance measures for cardiovascular care?",
        "How are quality measures for chronic coronary disease developed?",
        "What outcomes are tracked to measure cardiovascular care quality?",
        "What is the role of medication adherence in CCD performance measures?",
        "What is the recommended LDL cholesterol target?",
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"Query: {q}")
        print("=" * 80)
        results = retrieve_bm25(q, top_k=3)
        for rank, r in enumerate(results, 1):
            print(f"#{rank}  bm25_score={r.score:.4f}  page={r.page_number}  chunk_id={r.chunk_id}")
            print(f"   {r.text[:200]}...")
        print()