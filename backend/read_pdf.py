"""CLI: chunk a PDF and save chunks.json for a corpus."""
import argparse
import json
from backend.ingest import chunk_pdf, chunks_path
from config import DEFAULT_CORPUS, RAW_DIR


def main():
    parser = argparse.ArgumentParser(description="Chunk a PDF and save chunks.json.")
    parser.add_argument("--pdf", default=str(RAW_DIR / "guideline.pdf"),help="Path to source PDF")
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Corpus identifier (default: {DEFAULT_CORPUS})",
    )
    args = parser.parse_args()

    chunks = chunk_pdf(args.pdf)
    n_pages = len({c["page_number"] for c in chunks})

    out = chunks_path(args.corpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks: {len(chunks)} across {n_pages} pages")
    print(f"Saved chunks to: {out}")

    # Boundary inspection — confirm splits land on sentence breaks, not mid-name
    if len(chunks) >= 2:
        print("\n--- end of chunk 0 ---")
        print(chunks[0]["text"][-200:])
        print("\n--- start of chunk 1 ---")
        print(chunks[1]["text"][:200])


if __name__ == "__main__":
    main()