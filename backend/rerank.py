from sentence_transformers import CrossEncoder
from backend.schemas import RetrievedChunk
from backend.retrieve_hybrid import retrieve_hybrid

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_CORPUS = "cited_rx_chunks"

print(f"Loading rerank.py: {RERANKER_MODEL} (~600MB on first run)...")
_reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
print(f"Loaded reranker. Device: {_reranker.model.device}")


def rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-score candidates with the cross-encoder, return sorted by descending relevance."""
    if not candidates:
        return []

    pairs = [(query, c.text) for c in candidates]
    scores = _reranker.predict(pairs)

    rescored = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            score=float(s),
            page_number=c.page_number,
            source_doc=c.source_doc,
            text=c.text,
        )
        for c, s in zip(candidates, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored


def retrieve_with_reranker(
    query: str,
    qdrant_client,
    top_k: int = 5,
    candidate_k: int = 20,
    corpus_id: str = DEFAULT_CORPUS,
) -> list[RetrievedChunk]:
    """Hybrid retrieval (top candidate_k) → cross-encoder rerank → top_k."""
    candidates = retrieve_hybrid(query,qdrant_client=qdrant_client, top_k=candidate_k, corpus_id=corpus_id)
    reranked = rerank(query, candidates)
    return reranked[:top_k]


if __name__ == "__main__":

    from qdrant_client import QdrantClient
    test_client = QdrantClient(path="/Users/gowthamir/cited-rx/data/qdrant_storage")
    # Test harness unchanged
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

        candidates = retrieve_hybrid(q,qdrant_client=test_client, top_k=20)

        print("\n--- Hybrid top-5 (BEFORE rerank) ---")
        for rank, r in enumerate(candidates[:5], 1):
            print(f"  #{rank}  rrf={r.score:.4f}  page={r.page_number}  chunk_id={r.chunk_id}")

        reranked = rerank(q, candidates)

        print("\n--- Reranker top-5 (AFTER rerank) ---")
        for rank, r in enumerate(reranked[:5], 1):
            print(f"  #{rank}  rerank={r.score:+.4f}  page={r.page_number}  chunk_id={r.chunk_id}")
            print(f"     {r.text[:200]}...")

        hybrid_top5 = {r.chunk_id for r in candidates[:5]}
        rerank_top5 = {r.chunk_id for r in reranked[:5]}
        promoted = rerank_top5 - hybrid_top5
        demoted = hybrid_top5 - rerank_top5
        if promoted:
            print(f"\n  Promoted INTO top-5 by reranker: {sorted(promoted)}")
        if demoted:
            print(f"  Demoted OUT of top-5 by reranker: {sorted(demoted)}")
        print()