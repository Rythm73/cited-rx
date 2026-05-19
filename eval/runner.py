
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from config import EVAL_DIR, DEFAULT_CORPUS, QDRANT_PATH
from backend.pipeline import run_pipeline

from config import SIMILARITY_THRESHOLD

CONFIGS: dict[str, dict[str, Any]] = {
    "baseline":     {"top_k": 5, "threshold": SIMILARITY_THRESHOLD, "use_reranker": True,  "use_gate": True},
    "no_reranker":  {"top_k": 5, "threshold": SIMILARITY_THRESHOLD, "use_reranker": False, "use_gate": True},
    "no_gate":      {"top_k": 5, "threshold": -1.0, "use_reranker": True, "use_gate": False},
}

def run_gold_set(
    gold_path: Path,
    config_name: str,
    config: dict[str, Any],
    output_path: Path,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    with open(gold_path) as f:
        gold = json.load(f)

    client = QdrantClient(path=str(QDRANT_PATH))
    records: list[dict[str, Any]] = []

    for i, item in enumerate(gold, start=1):
        question = item["question"]
        category = item.get("category") or item.get("type", "unknown")
        if verbose:
            print(f"[{i:>2}/{len(gold)}] [{category:>12}] {question[:65]}...")

        start = time.perf_counter()
        record: dict[str, Any] = {
            "id": item.get("id", f"q{i:03d}"),
            "category": category,
            "question": question,
            "gold_answer": item.get("gold_answer", ""),
            "gold_pages": item.get("gold_pages", []),
            "config_name": config_name,
        }

        try:
            result = run_pipeline(
                question=question,
                qdrant_client=client,
                top_k=config["top_k"],
                threshold=config["threshold"],
                use_reranker=config["use_reranker"],
                use_gate=config["use_gate"],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            record.update({
                "generated_answer": result.raw_answer,
                "rendered_answer": result.rendered_answer,
                "retrieved_chunks": [
                    {"chunk_id": c.chunk_id, "page_number": c.page_number,
                     "score": float(c.score), "text": c.text}
                    for c in result.retrieved_chunks
                ],
                "citations": [
                    {"chunk_id": c.chunk_id, "quote": c.quote}
                    for c in result.citations
                ],
                "confidence": float(result.confidence),
                "refused": bool(result.refused),
                "latency_ms": latency_ms,
                "error": None,
            })
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            record.update({"error": f"{type(e).__name__}: {e}", "latency_ms": latency_ms})
            if verbose:
                print(f"      ERROR: {record['error']}")

        records.append(record)

    client.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    if verbose:
        successes = sum(1 for r in records if r.get("error") is None)
        refusals  = sum(1 for r in records if r.get("refused"))
        errors    = sum(1 for r in records if r.get("error"))
        avg_lat   = sum(r["latency_ms"] for r in records if not r.get("error")) / max(successes, 1)
        print(f"\n{'='*60}\nConfig: {config_name}  Total: {len(records)}  "
              f"OK: {successes}  Refused: {refusals}  Errors: {errors}  "
              f"Avg: {avg_lat:.0f}ms\nSaved: {output_path}\n{'='*60}")

    return records

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="baseline")
    parser.add_argument("--gold", type=Path, default=EVAL_DIR / "gold.json")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR / "runs")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_path = args.out_dir / f"{timestamp}_{args.config}.json"
    run_gold_set(gold_path=args.gold, config_name=args.config,
                 config=CONFIGS[args.config], output_path=output_path)

if __name__ == "__main__":
    main()