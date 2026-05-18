import re
from dataclasses import dataclass
from qdrant_client import QdrantClient
from config import DEFAULT_CORPUS, QDRANT_PATH
from backend.schemas import Citation, RetrievedChunk

@dataclass
class PipelineResult:
    question: str
    raw_answer: str
    rendered_answer: str
    confidence: float
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    refused: bool

def run_pipeline(
    question: str,
    qdrant_client: QdrantClient,
    top_k: int = 5,
    threshold: float = 0.0,
    use_reranker: bool = True,
    use_gate: bool = True,
    corpus_id: str = DEFAULT_CORPUS,
) -> PipelineResult:
    """End-to-end grounded RAG with togglable components for ablations."""
    if use_reranker:
        from backend.rerank import retrieve_with_reranker
        chunks = retrieve_with_reranker(question, qdrant_client=qdrant_client, top_k=top_k, corpus_id=corpus_id)
    else:
        from backend.retrieve_hybrid import retrieve_hybrid
        chunks = retrieve_hybrid(question, qdrant_client=qdrant_client, top_k=top_k, corpus_id=corpus_id)

    if use_gate:
        from backend.synthesize import synthesize_with_gate
        response = synthesize_with_gate(question, chunks, threshold=threshold)
    else:
        from backend.synthesize import synthesize
        response = synthesize(question, chunks)

    refused_flag = response.confidence == 0.0 and not response.citations

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
    client = QdrantClient(path=str(QDRANT_PATH))

    result = run_pipeline(
        "What is the recommended LDL-C target for patients with CCD?",
        qdrant_client=client,
        top_k=5,
        threshold=0.0,
    )
    print(f"Question:   {result.question}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Refused:    {result.refused}")
    print(f"Chunks:     {len(result.retrieved_chunks)}")
    print(f"\nAnswer:\n{result.rendered_answer}")

    print("\n--- ABLATION SMOKE TEST ---")
    r1 = run_pipeline("LDL-C target?", qdrant_client=client, use_reranker=True)
    r2 = run_pipeline("LDL-C target?", qdrant_client=client, use_reranker=False)
    print("reranker chunks:    ", [c.chunk_id for c in r1.retrieved_chunks])
    print("hybrid-only chunks: ", [c.chunk_id for c in r2.retrieved_chunks])
    print("Lists differ?", [c.chunk_id for c in r1.retrieved_chunks] != [c.chunk_id for c in r2.retrieved_chunks])

    client.close()