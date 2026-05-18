"""CLI: load chunks + embeddings for a corpus, push to Qdrant."""
import argparse
import json
import numpy as np
from backend.ingest import index_chunks, chunks_path, embeddings_path
from backend.retrieve import _get_model
from config import DEFAULT_CORPUS, QDRANT_PATH
from qdrant_client import QdrantClient

def main():
    parser = argparse.ArgumentParser(description="Index a corpus into Qdrant.")
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Corpus identifier (default: {DEFAULT_CORPUS})",
    )
    args = parser.parse_args()

    client = QdrantClient(path=str(QDRANT_PATH))

    with open(chunks_path(args.corpus), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(embeddings_path(args.corpus))
    print(f"Loaded {len(chunks)} chunks; embeddings shape {embeddings.shape}")

    index_chunks(args.corpus, chunks, embeddings, client)
    point_count = client.count(collection_name=args.corpus).count
    print(f"Collection '{args.corpus}' now contains {point_count} points")

    print("\n--- Verification: top-3 via Qdrant ---")
    test_query = "What are performance measures for cardiovascular care?"
    query_vec = _get_model().encode(test_query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=args.corpus,
        query=query_vec,
        limit=3,
        with_payload=True,
    ).points
    print(f"Query: {test_query}\n")
    for rank, point in enumerate(results, 1):
        p = point.payload
        print(f"#{rank}  score={point.score:.4f}  page={p['page_number']}  chunk_id={p['chunk_id']}")
        print(f"   {p['text'][:200]}...\n")

    client.close()

if __name__ == "__main__":
    main()