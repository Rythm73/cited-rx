"""CLI: load chunks for a corpus, embed, save embeddings.npy."""
import argparse
import json
import numpy as np
from backend.ingest import embed_chunks, chunks_path, embeddings_path, DEFAULT_CORPUS
from backend.retrieve import _model  # for the sanity check below


def main():
    parser = argparse.ArgumentParser(description="Embed chunks for a corpus.")
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Corpus identifier (default: {DEFAULT_CORPUS})",
    )
    args = parser.parse_args()

    with open(chunks_path(args.corpus), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path(args.corpus)}")

    embeddings = embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    out = embeddings_path(args.corpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    print(f"Saved embeddings to: {out}")

    # Sanity check — top-3 chunks via dot product, no Qdrant involved
    print("\n--- Sanity check: top-3 chunks for a test query ---")
    chunks_sorted = sorted(chunks, key=lambda c: c["chunk_id"])
    test_query = "What are performance measures for cardiovascular care?"
    query_vec = _model.encode(test_query, normalize_embeddings=True)
    sims = embeddings @ query_vec
    for rank, idx in enumerate(np.argsort(sims)[::-1][:3], 1):
        c = chunks_sorted[idx]
        print(f"#{rank}  similarity={sims[idx]:.4f}  page={c['page_number']}  chunk_id={c['chunk_id']}")
        print(f"   {c['text'][:200]}...\n")


if __name__ == "__main__":
    main()