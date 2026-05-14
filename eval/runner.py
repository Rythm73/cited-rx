"""
eval/runner.py

Iterates the gold question set through run_pipeline(), captures structured
records for downstream Ragas scoring + reporter aggregation.

Usage:
    python -m eval.runner --config baseline
    python -m eval.runner --config no_reranker --gold data/eval/gold.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sys
from pathlib import Path
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from pipeline import run_pipeline


CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "top_k": 5,
        "threshold": 0.0,
        "use_reranker": True,
        "use_gate": True,
    },
    "no_reranker": {
        "top_k": 5,
        "threshold": 0.0,
        "use_reranker": False,
        "use_gate": True,
    },
    "no_gate": {
        "top_k": 5,
        "threshold": -1.0,
        "use_reranker": True,
        "use_gate": False,
    },
}


def run_gold_set(
    gold_path: Path,
    config_name: str,
    config: dict[str, Any],
    output_path: Path,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run all gold questions through the pipeline. Returns raw records."""
    with open(gold_path) as f:
        gold = json.load(f)

    records: list[dict[str, Any]] = []

    for i, item in enumerate(gold, start=1):
        question = item["question"]
        category = item["category"]

        if verbose:
            print(f"[{i:>2}/{len(gold)}] [{category:>10}] {question[:65]}...")

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
                top_k=config["top_k"],
                threshold=config["threshold"],
                use_reranker=config["use_reranker"],
                use_gate=config["use_gate"],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)

            record.update(
                {
                    "generated_answer": result.raw_answer,
                    "rendered_answer": result.rendered_answer,
                    "retrieved_chunks": [
                        {
                            "chunk_id": c.chunk_id,
                            "page_number": c.page_number,
                            "score": float(c.score),
                            "text": c.text,
                        }
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
                }
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            record.update(
                {
                    "error": f"{type(e).__name__}: {e}",
                    "latency_ms": latency_ms,
                }
            )
            if verbose:
                print(f"      ERROR: {record['error']}")

        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    if verbose:
        successes = sum(1 for r in records if r.get("error") is None)
        refusals = sum(1 for r in records if r.get("refused"))
        errors = sum(1 for r in records if r.get("error"))
        avg_latency = (
            sum(r["latency_ms"] for r in records if r.get("error") is None)
            / max(successes, 1)
        )
        print()
        print("=" * 60)
        print(f"Config:        {config_name}")
        print(f"Total:         {len(records)}")
        print(f"Succeeded:     {successes}")
        print(f"Refused:       {refusals}")
        print(f"Errors:        {errors}")
        print(f"Avg latency:   {avg_latency:.0f} ms")
        print(f"Saved:         {output_path}")
        print("=" * 60)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gold eval set through cited-rx.")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="baseline")
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval/runs"))
    args = parser.parse_args()

    config = CONFIGS[args.config]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_path = args.out_dir / f"{timestamp}_{args.config}.json"

    run_gold_set(
        gold_path=args.gold,
        config_name=args.config,
        config=config,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
