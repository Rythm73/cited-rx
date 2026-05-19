
import os
import json
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from backend.schemas import Response, RetrievedChunk
from config import GROQ_MODEL, GEMINI_MODEL
load_dotenv()

print(f"Loading synthesize.py: Groq ({GROQ_MODEL}) with Gemini fallback ({GEMINI_MODEL})")
_groq_client = Groq()

def _get_gemini_client():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


SYSTEM_PROMPT = """You are a clinical research assistant. Your job is to answer questions about clinical guidelines using ONLY the chunks of source material provided in the user message.

You MUST follow four rules without exception:

1. Answer using ONLY information present in the provided chunks. Do not use background medical knowledge. Do not invent or extrapolate.

2. Insert inline citation markers in your answer text using the format [chunk_id=N] immediately after each claim that depends on chunk N. Example: "The LDL-C target is <70 mg/dL [chunk_id=174]." Use multiple markers when a claim is supported by multiple chunks: "[chunk_id=174][chunk_id=175]".

3. Every factual claim in your answer must also be listed in the citations field, with the chunk_id and an exact verbatim quote from that chunk supporting the claim. If you cannot quote a chunk verbatim, do not cite it.

4. If the chunks do not contain enough information to answer the question, say so directly. Set confidence to 0.0–0.3 and write something like "The provided sources do not contain a specific recommendation for [topic]," optionally adding any related context that IS in the chunks.

Respond with valid JSON only matching this schema:
{"answer": "...", "citations": [{"chunk_id": N, "quote": "..."}], "confidence": 0.0}
No markdown, no preamble."""


def _coerce_response_input(raw: dict) -> dict:
    coerced = dict(raw)
    if isinstance(coerced.get("citations"), str):
        coerced["citations"] = json.loads(coerced["citations"])
    return coerced

def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[chunk_id={c.chunk_id}, page={c.page_number}]\n{c.text}"
        for c in chunks
    )

# ── Synthesis ─────────────────────────────────────────────────
def _synthesize_with_groq(user_prompt: str) -> dict:
    completion = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


def _synthesize_with_gemini(user_prompt: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    client = _get_gemini_client()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no backticks."),
        HumanMessage(content=user_prompt),
    ]
    response = client.invoke(messages)
    text = response.content.strip()
    # Strip markdown fences if Gemini adds them
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


def synthesize(query: str, chunks: list[RetrievedChunk]) -> Response:
    user_prompt = (
        f"Question: {query}\n\n"
        f"Source chunks:\n{_format_chunks(chunks)}\n\n"
        f"Answer using only these chunks. Respond with JSON only."
    )
    try:
        raw = _synthesize_with_groq(user_prompt)
        print("[synthesize] used Groq")
    except GroqRateLimitError:
        print("[synthesize] Groq rate limit hit — falling back to Gemini")
        raw = _synthesize_with_gemini(user_prompt)
    return Response.model_validate(_coerce_response_input(raw))

# ── Gate ──────────────────────────────────────────────────────
NO_EVIDENCE_RESPONSE = Response(
    answer="The provided source material does not contain sufficient evidence to answer this question.",
    citations=[],
    confidence=0.0,
)

def synthesize_with_gate(
    query: str,
    chunks: list[RetrievedChunk],
    threshold: float = 0.0,
    top_semantic_score: float = 1.0,
) -> Response:
    if not chunks or top_semantic_score < threshold:
        return NO_EVIDENCE_RESPONSE
    return synthesize(query, chunks)

# ── Rendering ─────────────────────────────────────────────────
def render_answer_with_pages(response: Response, chunks: list[RetrievedChunk]) -> str:
    chunk_to_page = {c.chunk_id: c.page_number for c in chunks}

    def replace_marker(match: re.Match) -> str:
        cid = int(match.group(1))
        page = chunk_to_page.get(cid)
        return f"(p. {page})" if page is not None else f"[chunk {cid}]"

    rendered = re.sub(r"\[chunk_id=(\d+)\]", replace_marker, response.answer)
    if not response.citations:
        return rendered

    sources = [
        f"  • p. {chunk_to_page.get(c.chunk_id, '?')} (chunk {c.chunk_id}): \"{c.quote}\""
        for c in response.citations
    ]
    return f"{rendered}\n\nSources:\n" + "\n".join(sources)

# ── Citation Verification ─────────────────────────────────────
def verify_citations(
    citations: list,
    chunks: list[RetrievedChunk],
) -> list:
    """Return citations with verified=True if the quote appears verbatim in the source chunk."""
    chunk_text = {c.chunk_id: c.text for c in chunks}
    verified = []
    for citation in citations:
        text = chunk_text.get(citation.chunk_id, "")
        is_verified = citation.quote in text
        verified.append(citation.model_copy(update={"verified": is_verified}))
    return verified

# ── __main__ ──────────────────────────────────────────────────
if __name__ == "__main__":
    from backend.rerank import retrieve_with_reranker
    from qdrant_client import QdrantClient
    from config import QDRANT_PATH

    test_client = QdrantClient(path=str(QDRANT_PATH))
    test_queries = [
        "What is the recommended LDL cholesterol target?",
        "What is the capital of France?",
    ]
    for q in test_queries:
        print("=" * 80)
        print(f"Q: {q}")
        chunks = retrieve_with_reranker(q, qdrant_client=test_client, top_k=5)
        response = synthesize_with_gate(q, chunks, threshold=0.0)
        print(f"Confidence: {response.confidence:.2f}")
        print(render_answer_with_pages(response, chunks))
        print()
    test_client.close()