from backend.schemas import Response, Citation

# Valid construction
r = Response(
    answer="The recommended LDL-C target is <70 mg/dL for high-risk patients.",
    citations=[
        Citation(chunk_id=174, quote="LDL-C ≥70 mg/dL further reduces cardiovascular events"),
    ],
    confidence=0.85,
)
print("Valid response:")
print(r.model_dump_json(indent=2))

# Invalid - confidence above 1.0
try:
    Response(answer="Test", citations=[], confidence=1.5)
except Exception as e:
    print(f"\nValidation correctly rejected: {type(e).__name__}")