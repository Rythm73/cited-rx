import json
from pathlib import Path

# Files to update — both broken runs and the most recent no_gate
files_to_patch = [
    "data/eval/runs/2026-05-11_2011_no_gate.json",
]

for filepath in files_to_patch:
    path = Path(filepath)
    if not path.exists():
        print(f"SKIP {filepath} (not found)")
        continue

    records = json.loads(path.read_text())
    changed = 0
    for r in records:
        if r.get("error"):
            continue
        new_refused = (r["confidence"] == 0.0) and (not r["citations"])
        if r.get("refused") != new_refused:
            r["refused"] = new_refused
            changed += 1

    path.write_text(json.dumps(records, indent=2))
    print(f"PATCHED {filepath}: {changed} records updated")