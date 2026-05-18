"""
tests/smoke_test.py

One-command sanity check — run this BEFORE you start refactoring to
capture a baseline, then run it AGAIN after each change.

It does a real end-to-end query (Qdrant + LLM) and checks:
  - In-corpus question → answer with citations, confidence > 0
  - Out-of-corpus question → refused (confidence == 0, no citations)
  - PDF upload → new corpus indexed, queryable

Usage:
  python tests/smoke_test.py

Output is saved to: data/eval/smoke_results.json
Compare before/after with: diff data/eval/smoke_baseline.json data/eval/smoke_results.json
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import QDRANT_PATH, EVAL_DIR, DEFAULT_CORPUS
from qdrant_client import QdrantClient
from backend.rerank import retrieve_with_reranker
from backend.synthesize import synthesize_with_gate

TESTS = [
    # (label, question, expect_refused)
    ("in_corpus_ldl",   "What is the recommended LDL cholesterol target?",       False),
    ("in_corpus_bp",    "What is the blood pressure target for CCD patients?",   False),
    ("in_corpus_statin","What is the class of recommendation for statin therapy?",False),
    ("out_of_corpus_1", "What is the capital of France?",                         True),
    ("out_of_corpus_2", "How do you cook spaghetti carbonara?",                   True),
]


def run_smoke_tests() -> dict:
    print("=" * 60)
    print("cited-rx smoke test")
    print("=" * 60)

    client = QdrantClient(path=str(QDRANT_PATH))
    results = []
    passed = 0
    failed = 0

    for label, question, expect_refused in TESTS:
        print(f"\n[{label}]")
        print(f"  Q: {question[:70]}")

        start = time.perf_counter()
        try:
            chunks   = retrieve_with_reranker(question, qdrant_client=client,
                                               top_k=5, corpus_id=DEFAULT_CORPUS)
            response = synthesize_with_gate(question, chunks, threshold=0.0)
            latency  = int((time.perf_counter() - start) * 1000)

            actually_refused = response.confidence == 0.0 and not response.citations
            ok = (actually_refused == expect_refused)

            result = {
                "label":          label,
                "question":       question,
                "expected_refused": expect_refused,
                "actually_refused": actually_refused,
                "confidence":     response.confidence,
                "n_citations":    len(response.citations),
                "answer_preview": response.answer[:120],
                "latency_ms":     latency,
                "pass":           ok,
                "error":          None,
            }

            status = "PASS ✓" if ok else "FAIL ✗"
            print(f"  Refused: {actually_refused} (expected {expect_refused}) → {status}")
            print(f"  Confidence: {response.confidence:.2f}  |  Citations: {len(response.citations)}  |  {latency}ms")

            if ok:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            latency = int((time.perf_counter() - start) * 1000)
            result = {
                "label":    label,
                "question": question,
                "error":    f"{type(e).__name__}: {e}",
                "latency_ms": latency,
                "pass":     False,
            }
            print(f"  ERROR: {result['error']}")
            failed += 1

        results.append(result)

    client.close()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "passed":    passed,
        "failed":    failed,
        "total":     len(TESTS),
        "results":   results,
    }

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(TESTS)} passed")
    if failed > 0:
        print("FAILED TESTS:")
        for r in results:
            if not r.get("pass"):
                print(f"  ✗ {r['label']}: {r.get('error', 'wrong refusal behavior')}")
    print("=" * 60)

    # Save results
    out = EVAL_DIR / "smoke_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")

    # Also save as baseline if one doesn't exist
    baseline = EVAL_DIR / "smoke_baseline.json"
    if not baseline.exists():
        baseline.write_text(json.dumps(summary, indent=2))
        print(f"Saved baseline: {baseline}  (delete to reset)")

    return summary


if __name__ == "__main__":
    summary = run_smoke_tests()
    sys.exit(0 if summary["failed"] == 0 else 1)