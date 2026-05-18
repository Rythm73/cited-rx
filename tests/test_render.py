import pytest
from backend.schemas import Response, Citation, RetrievedChunk
from backend.synthesize import render_answer_with_pages


def _make_chunks(*pairs):
    """Helper: pairs of (chunk_id, page_number)"""
    return [
        RetrievedChunk(chunk_id=cid, page_number=pg, text="x", score=1.0)
        for cid, pg in pairs
    ]


def test_known_chunk_id_replaced():
    chunks = _make_chunks((42, 7))
    response = Response(
        answer="The LDL target is 70 mg/dL [chunk_id=42].",
        citations=[Citation(chunk_id=42, quote="LDL target is 70 mg/dL")],
        confidence=0.9,
    )
    result = render_answer_with_pages(response, chunks)
    assert "(p. 7)" in result
    assert "[chunk_id=42]" not in result


def test_unknown_chunk_id_fallback():
    chunks = _make_chunks((1, 3))
    response = Response(
        answer="Some claim [chunk_id=999].",
        citations=[],
        confidence=0.5,
    )
    result = render_answer_with_pages(response, chunks)
    # chunk 999 is not in chunks — should fall back gracefully, not crash
    assert "[chunk_id=999]" not in result
    assert "[chunk 999]" in result  # canonical fallback format


def test_no_citations_no_sources_block():
    chunks = _make_chunks((1, 5))
    response = Response(
        answer="No citations here.",
        citations=[],
        confidence=0.8,
    )
    result = render_answer_with_pages(response, chunks)
    assert "Sources:" not in result


def test_with_citations_sources_block_present():
    chunks = _make_chunks((10, 4))
    response = Response(
        answer="A claim [chunk_id=10].",
        citations=[Citation(chunk_id=10, quote="verbatim quote")],
        confidence=0.95,
    )
    result = render_answer_with_pages(response, chunks)
    assert "Sources:" in result
    assert "verbatim quote" in result

def _make_chunks(*pairs):
    """Helper: pairs of (chunk_id, page_number)"""
    return [
        RetrievedChunk(chunk_id=cid, page_number=pg, text="x", score=1.0, source_doc="test.pdf")
        for cid, pg in pairs
    ]