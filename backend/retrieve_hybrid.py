from backend.retrieve import retrieve as retrieve_semantic
from backend.retrieve_bm25 import retrieve_bm25
from backend.schemas import RetrievedChunk

RRF_K = 60
DEFAULT_CORPUS = "cited_rx_chunks"


def retrieve_hybrid(
    query: str,
    qdrant_client,
    top_k: int = 5,
    candidate_k: int = 20,
    corpus_id: str = DEFAULT_CORPUS,
) -> list[RetrievedChunk]:
    """Hybrid retrieval: semantic + BM25, fused via Reciprocal Rank Fusion."""
    sem_results = retrieve_semantic(query,qdrant_client=qdrant_client, top_k=candidate_k, corpus_id=corpus_id)
    bm25_results = retrieve_bm25(query, top_k=candidate_k, corpus_id=corpus_id)

    sem_ranks = {r.chunk_id: i + 1 for i, r in enumerate(sem_results)}
    bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

    chunk_lookup: dict[int, RetrievedChunk] = {}
    for r in sem_results + bm25_results:
        if r.chunk_id not in chunk_lookup:
            chunk_lookup[r.chunk_id] = r

    all_ids = set(sem_ranks) | set(bm25_ranks)
    rrf_scores: dict[int, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in sem_ranks:
            score += 1.0 / (RRF_K + sem_ranks[cid])
        if cid in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[cid])
        rrf_scores[cid] = score

    top_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            chunk_id=chunk_lookup[cid].chunk_id,
            score=rrf_scores[cid],
            page_number=chunk_lookup[cid].page_number,
            source_doc=chunk_lookup[cid].source_doc,
            text=chunk_lookup[cid].text,
        )
        for cid in top_ids
    ]


if __name__ == "__main__":
    from qdrant_client import QdrantClient
    
    # Initialize a local client just for the test harness
    test_client = QdrantClient(path="./data")

    # Test harness unchanged — uses default corpus everywhere
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

        sem_ids = {r.chunk_id for r in retrieve_semantic(q,qdrant_client=test_client, top_k=20)}
        bm25_ids = {r.chunk_id for r in retrieve_bm25(q, top_k=20)}

        results = retrieve_hybrid(q,qdrant_client=test_client,top_k=5)
        for rank, r in enumerate(results, 1):
            in_sem = r.chunk_id in sem_ids
            in_bm25 = r.chunk_id in bm25_ids
            provenance = "BOTH" if in_sem and in_bm25 else ("sem" if in_sem else "bm25")
            print(f"#{rank}  rrf={r.score:.4f}  page={r.page_number}  "
                  f"chunk_id={r.chunk_id}  [{provenance}]")
            print(f"   {r.text[:200]}...")
        print()