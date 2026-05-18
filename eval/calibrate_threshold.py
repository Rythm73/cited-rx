"""
Calibration script — run once to pick SIMILARITY_THRESHOLD.
Prints the top semantic cosine score for every gold question,
grouped by in-corpus vs out-of-corpus.

Usage:
    python eval/calibrate_threshold.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from backend.retrieve import get_top_semantic_score
from config import QDRANT_PATH, EVAL_DIR

gold_path = EVAL_DIR / "gold.json"
gold = json.loads(gold_path.read_text())

client = QdrantClient(path=str(QDRANT_PATH))

in_corpus_scores = []
ooc_scores = []

print(f"\n{'Category':<15} {'ID':<20} {'Score':>6}  Question")
print("-" * 90)

for item in gold:
    score = get_top_semantic_score(item["question"], qdrant_client=client)
    category = item.get("category", "unknown")
    is_ooc = category == "out_of_corpus"
    tag = "OOC" if is_ooc else "IN "
    print(f"[{tag}] {item['id']:<20} {score:.4f}  {item['question'][:60]}")
    if is_ooc:
        ooc_scores.append(score)
    else:
        in_corpus_scores.append(score)

client.close()

print("\n" + "=" * 90)
print(f"IN-CORPUS  scores: min={min(in_corpus_scores):.4f}  max={max(in_corpus_scores):.4f}  mean={sum(in_corpus_scores)/len(in_corpus_scores):.4f}")
print(f"OOC        scores: min={min(ooc_scores):.4f}  max={max(ooc_scores):.4f}  mean={sum(ooc_scores)/len(ooc_scores):.4f}")
print("\nSuggested threshold: midpoint between OOC max and IN-CORPUS min")
midpoint = (max(ooc_scores) + min(in_corpus_scores)) / 2
print(f"  → {midpoint:.4f}")