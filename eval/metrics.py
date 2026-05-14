"""
eval/metrics.py

Wraps Ragas to score run records on three RAG metrics:
- Faithfulness: does the answer only make claims supported by retrieved context?
- LLMContextPrecisionWithReference: were retrieved chunks actually relevant?
- LLMContextRecall: did retrieval find chunks needed to support the gold answer?

Out-of-corpus questions are scored separately as refusal_precision since Ragas
metrics are ill-defined for refused responses (no claims to check).

Compatible with Ragas 0.4.x using the (legacy) EvaluationDataset API.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def score_ragas(
    records: list[dict[str, Any]],
    judge_model: str = "claude-sonnet-4-6",
) -> list[dict[str, Any]]:
    """Run Ragas metrics on eligible records (non-refused, non-error, in-corpus)."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from langchain_anthropic import ChatAnthropic

    eligible = [
        (i, r) for i, r in enumerate(records)
        if not r.get("refused")
        and not r.get("error")
        and r.get("category") != "out_of_corpus"
    ]

    if not eligible:
        print("No eligible records for Ragas scoring.")
        return records

    print(
        f"Scoring {len(eligible)} records with Ragas "
        f"(skipping {len(records) - len(eligible)} refused/error/OOC)"
    )

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["generated_answer"],
            retrieved_contexts=[c["text"] for c in r["retrieved_chunks"]],
            reference=r["gold_answer"],
        )
        for _, r in eligible
    ]
    eval_dataset = EvaluationDataset(samples=samples)

    judge = LangchainLLMWrapper(
        ChatAnthropic(model=judge_model, temperature=0, max_tokens=2048)
    )

    metrics = [
        Faithfulness(llm=judge),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    print(f"Judge: {judge_model}")
    print(f"Metrics: {[m.__class__.__name__ for m in metrics]}\n")

    run_config = RunConfig(
        max_workers=2,
        timeout=180,
        max_retries=5,
        max_wait=60,
    )

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge,
        run_config=run_config,
        show_progress=True,
    )

    scores_df = result.to_pandas()
    print(f"\nResult columns: {list(scores_df.columns)}")

    # Map possible column names (Ragas naming has shifted across versions)
    def col(*candidates):
        for c in candidates:
            if c in scores_df.columns:
                return c
        return None

    faith_col = col("faithfulness")
    prec_col = col("llm_context_precision_with_reference", "context_precision")
    recall_col = col("context_recall", "llm_context_recall")

    for sample_idx, (record_idx, _) in enumerate(eligible):
        row = scores_df.iloc[sample_idx]
        records[record_idx]["scores"] = {
            "faithfulness": float(row[faith_col]) if faith_col else None,
            "context_precision": float(row[prec_col]) if prec_col else None,
            "context_recall": float(row[recall_col]) if recall_col else None,
        }

    return records


def score_refusal_precision(records: list[dict[str, Any]]) -> dict[str, Any]:
    """OOC questions: did the system correctly refuse?"""
    ooc = [r for r in records if r.get("category") == "out_of_corpus"]
    if not ooc:
        return {"n": 0, "correctly_refused": 0, "refusal_precision": None}
    correct = sum(1 for r in ooc if r.get("refused"))
    return {
        "n": len(ooc),
        "correctly_refused": correct,
        "refusal_precision": correct / len(ooc),
    }


def aggregate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall + per-category metric averages."""
    from collections import defaultdict

    metric_keys = ["faithfulness", "context_precision", "context_recall"]
    by_cat = defaultdict(list)
    overall = []

    for r in records:
        scores = r.get("scores")
        if not scores:
            continue
        by_cat[r["category"]].append(scores)
        overall.append(scores)

    def avg(score_list, key):
        vals = [s[key] for s in score_list if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary: dict[str, Any] = {
        "overall": {k: avg(overall, k) for k in metric_keys},
        "by_category": {
            cat: {k: avg(scores_list, k) for k in metric_keys}
            for cat, scores_list in by_cat.items()
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Score a run with Ragas metrics.")
    parser.add_argument("--run", type=Path, required=True, help="Run JSON from runner.py")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    if not args.run.exists():
        sys.exit(f"Run file not found: {args.run}")

    records = json.load(open(args.run))
    print(f"Loaded {len(records)} records from {args.run}\n")

    records = score_ragas(records, judge_model=args.judge_model)
    refusal = score_refusal_precision(records)
    summary = aggregate_summary(records)

    out_path = args.out or args.run.parent / f"{args.run.stem}_scored.json"
    out = {
        "records": records,
        "refusal_precision": refusal,
        "summary": summary,
        "judge_model": args.judge_model,
        "scored_at": datetime.now().isoformat(),
    }
    json.dump(out, open(out_path, "w"), indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Judge: {args.judge_model}")
    print(f"\nOverall (in-corpus, non-refused):")
    for k, v in summary["overall"].items():
        print(f"  {k:20s} {v:.3f}" if v is not None else f"  {k:20s} N/A")
    print(f"\nBy category:")
    for cat, scores in sorted(summary["by_category"].items()):
        print(f"  {cat}:")
        for k, v in scores.items():
            print(f"    {k:20s} {v:.3f}" if v is not None else f"    {k:20s} N/A")
    print(f"\nRefusal precision (OOC): {refusal['correctly_refused']}/{refusal['n']} = {refusal['refusal_precision']:.2f}")
    print(f"\nSaved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
