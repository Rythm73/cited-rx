import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import anthropic

from backend.schemas import Response, RetrievedChunk

load_dotenv()

MODEL = "claude-sonnet-4-6"

# Module-level singleton — reuses the connection across calls
print("Loading synthesize.py: Claude API client...")
_client = anthropic.Anthropic()
print("Loaded.")

SYSTEM_PROMPT = """You are a clinical research assistant. Your job is to answer questions about clinical guidelines using ONLY the chunks of source material provided in the user message.

You MUST follow four rules without exception:

1. Answer using ONLY information present in the provided chunks. Do not use background medical knowledge. Do not invent or extrapolate.

2. Insert inline citation markers in your answer text using the format [chunk_id=N] immediately after each claim that depends on chunk N. Example: "The LDL-C target is <70 mg/dL [chunk_id=174]." Use multiple markers when a claim is supported by multiple chunks: "[chunk_id=174][chunk_id=175]".

3. Every factual claim in your answer must also be listed in the citations field, with the chunk_id and an exact verbatim quote from that chunk supporting the claim. If you cannot quote a chunk verbatim, do not cite it.

4. If the chunks do not contain enough information to answer the question, say so directly. Set confidence to 0.0–0.3 and write something like "The provided sources do not contain a specific recommendation for [topic]," optionally adding any related context that IS in the chunks.

Always respond by calling the `submit_response` tool. Do not respond conversationally."""



# The Response schema, exported as a Claude tool definition
RESPONSE_TOOL = {
    "name": "submit_response",
    "description": "Submit your structured answer with citations to the user's question.",
    "input_schema": Response.model_json_schema(),
}

import json


def _coerce_response_input(raw: dict) -> dict:
    """Defensively handle Claude occasionally returning nested fields as JSON strings."""
    coerced = dict(raw)
    if isinstance(coerced.get("citations"), str):
        coerced["citations"] = json.loads(coerced["citations"])
    return coerced

def synthesize(query: str, chunks: list[RetrievedChunk]) -> Response:
    """Generate a grounded, cited answer from query + chunks using Claude.

    Forces Claude to use the submit_response tool so the output conforms
    to the Response Pydantic schema.
    """
    # Format chunks with explicit chunk_id labels Claude can cite
    chunks_text = "\n\n".join(
        f"[chunk_id={c.chunk_id}, page={c.page_number}]\n{c.text}"
        for c in chunks
    )

    user_prompt = f"""Question: {query}

Source chunks:
{chunks_text}

Answer the question using only these chunks. Use the submit_response tool."""

    
    api_response = _client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0,
        system=SYSTEM_PROMPT,
        tools=[RESPONSE_TOOL],
        tool_choice={"type": "tool", "name": "submit_response"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract the tool call's input and validate against Pydantic
    for block in api_response.content:
        if block.type == "tool_use" and block.name == "submit_response":
            return Response.model_validate(_coerce_response_input(block.input))

    raise RuntimeError(f"Expected submit_response tool call. Got: {api_response.content}")

# Standard "I don't know" response when retrieval is too weak
NO_EVIDENCE_RESPONSE = Response(
    answer="The provided source material does not contain sufficient evidence to answer this question.",
    citations=[],
    confidence=0.0,
)


def synthesize_with_gate(
    query: str,
    chunks: list[RetrievedChunk],
    threshold: float = 0.0,
) -> Response:
    """Synthesize, but skip the LLM call entirely if retrieval is too weak.

    Returns a 'no evidence' Response without calling Claude when the top chunk's
    score is below `threshold`. Otherwise delegates to synthesize().
    """
    if not chunks or chunks[0].score < threshold:
        return NO_EVIDENCE_RESPONSE
    return synthesize(query, chunks)


def answer_query(
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
) -> Response:
    """End-to-end: retrieve → rerank → gate → synthesize → grounded Response."""
    from rerank import retrieve_with_reranker
    chunks = retrieve_with_reranker(query, top_k=top_k)
    return synthesize_with_gate(query, chunks, threshold=threshold)

import re


def render_answer_with_pages(
    response: Response,
    chunks: list[RetrievedChunk],
) -> str:
    """Replace [chunk_id=N] markers in the answer with (p. X) page references,
    and append a Sources section listing each citation with its page number.
    """
    chunk_to_page = {c.chunk_id: c.page_number for c in chunks}

    def replace_marker(match: re.Match) -> str:
        cid = int(match.group(1))
        page = chunk_to_page.get(cid)
        return f"(p. {page})" if page is not None else f"[chunk {cid}]"

    rendered_answer = re.sub(r"\[chunk_id=(\d+)\]", replace_marker, response.answer)

    if not response.citations:
        return rendered_answer

    sources_lines = []
    for c in response.citations:
        page = chunk_to_page.get(c.chunk_id, "?")
        sources_lines.append(f"  • p. {page} (chunk {c.chunk_id}): \"{c.quote}\"")

    return f"{rendered_answer}\n\nSources:\n" + "\n".join(sources_lines)

# End-to-end test: retrieve → rerank → synthesize for each query
if __name__ == "__main__":
    from rerank import retrieve_with_reranker

    test_queries = [
        "What are performance measures for cardiovascular care?",
        "What is the recommended LDL cholesterol target?",
        "What is the capital of France?",  # out-of-corpus — should hit the gate
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"Q: {q}")
        print("=" * 80)

        chunks = retrieve_with_reranker(q, top_k=5)
        response = synthesize_with_gate(q, chunks, threshold=0.0)

        print(f"\nConfidence: {response.confidence:.2f}\n")
        print(render_answer_with_pages(response, chunks))
        print()