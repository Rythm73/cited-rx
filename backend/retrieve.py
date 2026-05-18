from functools import lru_cache

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from backend.schemas import RetrievedChunk
from config import DEFAULT_CORPUS, EMBEDDING_MODEL

MODEL_NAME = EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load BGE-M3 once, on first call. Cached for the process lifetime."""
    print("Loading retrieve.py: BGE-M3 model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded. Model device: {model.device}")
    return model

def retrieve(
    query: str,
    qdrant_client, 
    top_k: int = 5,
    corpus_id: str = DEFAULT_CORPUS,
) -> list[RetrievedChunk]:
    """Embed query, search Qdrant collection `corpus_id`, return top-k chunks."""
    query_vec = _get_model().encode(query, normalize_embeddings=True).tolist()
    
    results = qdrant_client.query_points(
        collection_name=corpus_id,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            chunk_id=p.payload["chunk_id"],
            score=p.score,
            page_number=p.payload["page_number"],
            source_doc=p.payload["source_doc"],
            text=p.payload["text"],
        )
        for p in results
    ]


if __name__ == "__main__":

    from config import QDRANT_PATH

    test_client = QdrantClient(path=QDRANT_PATH)
    
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
        
        # Pass the test_client down here
        results = retrieve(q, qdrant_client=test_client, top_k=3)
        
        for rank, r in enumerate(results, 1):
            print(f"#{rank}  score={r.score:.4f}  page={r.page_number}  chunk_id={r.chunk_id}")
            print(f"   {r.text[:200]}...")
        print()

    test_client.close()