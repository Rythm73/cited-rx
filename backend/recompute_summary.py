import json
import math
from collections import defaultdict

scored_path = "data/eval/runs/2026-05-11_2011_no_gate_scored.json"

data = json.load(open(scored_path))
records = data["records"]


def is_valid(v):
    """Filter out None AND NaN floats."""
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def avg(score_list, key):
    vals = [s[key] for s in score_list if is_valid(s.get(key))]
    return sum(vals) / len(vals) if vals else None


def count_used(score_list, key):
    return sum(1 for s in score_list if is_valid(s.get(key)))


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