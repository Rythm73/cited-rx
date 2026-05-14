import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineResult:
    question: str
    raw_answer: str
    rendered_answer: str
    confidence: float
    citations: list[Any]
    retrieved_chunks: list[Any]
    refused: bool


def run_pipeline(
    question: str,
    top_k: int = 5,
    threshold: float = 0.0,
    use_reranker: bool = True,
    use_gate: bool = True,
    corpus_id: str = "cited_rx_chunks",
) -> PipelineResult:
    """End-to-end grounded RAG with togglable components for ablations."""

    # 1. Retrieval: with or without reranker
    if use_reranker:
        from rerank import retrieve_with_reranker
        chunks = retrieve_with_reranker(question, top_k=top_k, corpus_id=corpus_id)
    else:
        from retrieve_hybrid import retrieve_hybrid
        chunks = retrieve_hybrid(question, top_k=top_k, corpus_id=corpus_id)

    # 2. Synthesis: with or without gate
    if use_gate:
        from synthesize import synthesize_with_gate
        response = synthesize_with_gate(question, chunks, threshold=threshold)
    else:
        from synthesize import synthesize
        response = synthesize(question, chunks)

    # Unified refusal detection — captures BOTH gate-trips and LLM Rule-4 refusals
    refused_flag = response.confidence == 0.0 and not response.citations

    # 3. Render answer with page references
    chunk_to_page = {c.chunk_id: c.page_number for c in chunks}
    rendered = re.sub(
        r"\[chunk_id=(\d+)\]",
        lambda m: f"(p. {chunk_to_page.get(int(m.group(1)), '?')})",
        response.answer,
    )

    return PipelineResult(
        question=question,
        raw_answer=response.answer,
        rendered_answer=rendered,
        confidence=response.confidence,
        citations=response.citations,
        retrieved_chunks=chunks,
        refused=refused_flag,
    )


if __name__ == "__main__":
    # Smoke test 1: full pipeline
    result = run_pipeline(
        "What is the recommended LDL-C target for patients with CCD?",
        top_k=5,
        threshold=0.0,
    )
    print(f"Question:   {result.question}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Refused:    {result.refused}")
    print(f"Chunks:     {len(result.retrieved_chunks)}")
    print(f"\nAnswer:\n{result.rendered_answer}")

    # Smoke test 2: verify the use_reranker flag actually does something
    print("\n--- ABLATION SMOKE TEST ---")
    r1 = run_pipeline("LDL-C target?", use_reranker=True)
    r2 = run_pipeline("LDL-C target?", use_reranker=False)
    print("reranker chunks:    ", [c.chunk_id for c in r1.retrieved_chunks])
    print("hybrid-only chunks: ", [c.chunk_id for c in r2.retrieved_chunks])
    print("Lists differ?", [c.chunk_id for c in r1.retrieved_chunks] != [c.chunk_id for c in r2.retrieved_chunks])