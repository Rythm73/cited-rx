
from dataclasses import dataclass
from qdrant_client import QdrantClient
from config import DEFAULT_CORPUS, QDRANT_PATH, SIMILARITY_THRESHOLD
from backend.schemas import Citation, RetrievedChunk
from backend.synthesize import render_answer_with_pages, verify_citations
from backend.retrieve import get_top_semantic_score

@dataclass
class PipelineResult:
    question: str
    raw_answer: str
    rendered_answer: str
    confidence: float
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    refused: bool
    citation_verification_rate: float
    top_semantic_score: float

def run_pipeline(
    question: str,
    qdrant_client: QdrantClient,
    top_k: int = 5,
    threshold: float = SIMILARITY_THRESHOLD,
    use_reranker: bool = True,
    use_gate: bool = True,
    corpus_id: str = DEFAULT_CORPUS,
) -> PipelineResult:
    """End-to-end grounded RAG with togglable components for ablations."""
    top_sem_score = get_top_semantic_score(question, qdrant_client=qdrant_client, corpus_id=corpus_id)

    if use_reranker:
        from backend.rerank import retrieve_with_reranker
        chunks = retrieve_with_reranker(question, qdrant_client=qdrant_client, top_k=top_k, corpus_id=corpus_id)
    else:
        from backend.retrieve_hybrid import retrieve_hybrid
        chunks = retrieve_hybrid(question, qdrant_client=qdrant_client, top_k=top_k, corpus_id=corpus_id)

    if use_gate:
        from backend.synthesize import synthesize_with_gate
        response = synthesize_with_gate(question, chunks, threshold=threshold, top_semantic_score=top_sem_score)
    else:
        from backend.synthesize import synthesize
        response = synthesize(question, chunks)

    refused_flag = response.confidence == 0.0 and not response.citations

    verified_citations = verify_citations(response.citations, chunks)
    if verified_citations:
        verification_rate = sum(1 for c in verified_citations if c.verified) / len(verified_citations)
    else:
        verification_rate = 1.0  # no citations to verify — not a failure

    rendered = render_answer_with_pages(response, chunks)

    return PipelineResult(
        question=question,
        raw_answer=response.answer,
        rendered_answer=rendered,
        confidence=response.confidence,
        citations=verified_citations,
        retrieved_chunks=chunks,
        refused=refused_flag,
        citation_verification_rate=verification_rate,
        top_semantic_score=top_sem_score,
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