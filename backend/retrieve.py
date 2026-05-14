from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from backend.schemas import RetrievedChunk

DEFAULT_CORPUS = "cited_rx_chunks"
MODEL_NAME = "BAAI/bge-m3"

print("Loading retrieve.py: BGE-M3 model...")
# Keep the model global, that is perfectly fine for Modal!
_model = SentenceTransformer(MODEL_NAME)
print(f"Loaded. Model device: {_model.device}")

# 🚨 DELETED THE GLOBAL _client HERE 🚨


def retrieve(
    query: str,
    qdrant_client,  # <--- We pass the client from the API lifespan
    top_k: int = 5,
    corpus_id: str = DEFAULT_CORPUS,
) -> list[RetrievedChunk]:
    """Embed query, search Qdrant collection `corpus_id`, return top-k chunks."""
    query_vec = _model.encode(query, normalize_embeddings=True).tolist()
    
    # 🚨 Use the passed 'qdrant_client' instead of the global '_client'
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
    # Test harness unchanged — uses default corpus
    QDRANT_PATH = "/root/data/qdrant_storage"
    
    # Initialize a local client ONLY for the test harness
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