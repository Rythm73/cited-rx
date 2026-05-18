import json
import math
import argparse
from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EVAL_DIR

def is_valid(v):
    """Filter out None AND NaN floats."""
    return v is not None and not (isinstance(v, float) and math.isnan(v))

def avg(score_list, key):
    vals = [s[key] for s in score_list if is_valid(s.get(key))]
    return sum(vals) / len(vals) if vals else None

def count_used(score_list, key):
    return sum(1 for s in score_list if is_valid(s.get(key)))

def main():
    parser = argparse.ArgumentParser(description="Recompute summary from a scored run JSON.")
    parser.add_argument("--run", required=True, help="Path to scored run JSON")
    args = parser.parse_args()

    data = json.load(open(args.run))
    records = data["records"]

    metric_keys = ["faithfulness", "context_precision", "context_recall"]
    by_cat = defaultdict(list)
    all_scores = []

    for r in records:
        scores = r.get("scores")
        if not scores:
            continue
        all_scores.append(scores)
        by_cat[r["category"]].append(scores)

    print("=" * 60)
    print("RECOMPUTED SUMMARY (NaN-excluded)")
    print("=" * 60)
    print(f"\nOverall (in-corpus, non-refused):")
    for k in metric_keys:
        a = avg(all_scores, k)
        n = count_used(all_scores, k)
        total = len(all_scores)
        if a is not None:
            print(f"  {k:20s} {a:.3f}    (n={n}/{total})")
        else:
            print(f"  {k:20s} no valid scores")

    print(f"\nBy category:")
    for cat in sorted(by_cat):
        print(f"  {cat}:")
        for k in metric_keys:
            a = avg(by_cat[cat], k)
            n = count_used(by_cat[cat], k)
            total = len(by_cat[cat])
            if a is not None:
                print(f"    {k:20s} {a:.3f}    (n={n}/{total})")
            else:
                print(f"    {k:20s} no valid scores")
    print("=" * 60)

if __name__ == "__main__":
    main()