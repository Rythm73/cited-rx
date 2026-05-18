"""
tests/test_integration.py

Integration tests for the cited-rx retrieval stack: semantic search, BM25,
hybrid RRF fusion, and the cross-encoder reranker. Exercises the real
components against a live local Qdrant instance.

Requirements:
  - A local Qdrant database with the default corpus indexed (run ingestion).
  - No LLM / Groq calls — synthesis is deliberately not exercised here.

Note: local-mode Qdrant allows a single client at a time. Do not run these
tests while the app server (uvicorn) is running against the same database.
"""
import math

import pytest

from config import DEFAULT_CORPUS, QDRANT_PATH
from backend.schemas import RetrievedChunk
from backend.retrieve import retrieve
from backend.retrieve_bm25 import retrieve_bm25
from backend.retrieve_hybrid import retrieve_hybrid
from backend.rerank import rerank, retrieve_with_reranker


# A question known to have real answers in the cardiology corpus.
IN_CORPUS_QUERY = "What is the recommended LDL cholesterol target?"
# A keyword query for the BM25 relevance check: the term should appear
# verbatim in the top keyword-matched chunks.
KEYWORD_QUERY = "statin therapy"
KEYWORD = "statin"


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def qdrant_client():
    """One shared local Qdrant client for the whole module — local mode allows
    only a single connection at a time."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(QDRANT_PATH))
    yield client
    client.close()


@pytest.fixture(scope="module")
def indexed_corpus(qdrant_client):
    """Skip the retrieval tests cleanly when the default corpus has not been
    indexed yet, instead of failing with confusing downstream errors."""
    if not qdrant_client.collection_exists(DEFAULT_CORPUS):
        pytest.skip(f"Corpus '{DEFAULT_CORPUS}' is not indexed — run ingestion first")
    if qdrant_client.count(collection_name=DEFAULT_CORPUS).count == 0:
        pytest.skip(f"Corpus '{DEFAULT_CORPUS}' exists but is empty — run ingestion first")
    return DEFAULT_CORPUS


# ── 1. Corpus collection ──────────────────────────────────────────────────
class TestCorpusCollection:
    """The default corpus must exist in Qdrant and contain points."""

    def test_collection_exists(self, qdrant_client):
        assert qdrant_client.collection_exists(DEFAULT_CORPUS), (
            f"Qdrant collection '{DEFAULT_CORPUS}' does not exist — run ingestion"
        )

    def test_collection_has_points(self, qdrant_client):
        if not qdrant_client.collection_exists(DEFAULT_CORPUS):
            pytest.skip("collection does not exist — see test_collection_exists")
        count = qdrant_client.count(collection_name=DEFAULT_CORPUS).count
        assert count > 0, f"Collection '{DEFAULT_CORPUS}' contains no points"


# ── 2. Semantic retrieval ─────────────────────────────────────────────────
class TestSemanticRetrieval:
    """BGE-M3 query embedding + Qdrant vector search."""

    def test_returns_results(self, qdrant_client, indexed_corpus):
        results = retrieve(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=5)
        assert 0 < len(results) <= 5

    def test_scores_descending(self, qdrant_client, indexed_corpus):
        results = retrieve(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_fields_present(self, qdrant_client, indexed_corpus):
        results = retrieve(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=5)
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert isinstance(r.chunk_id, int)
            assert isinstance(r.score, float) and math.isfinite(r.score)
            assert isinstance(r.page_number, int)
            assert isinstance(r.source_doc, str) and r.source_doc
            assert isinstance(r.text, str) and r.text


# ── 3. BM25 retrieval ─────────────────────────────────────────────────────
class TestBM25Retrieval:
    """rank-bm25 keyword search over the chunk corpus (no Qdrant involved)."""

    def test_returns_results(self, indexed_corpus):
        results = retrieve_bm25(IN_CORPUS_QUERY, top_k=5)
        assert 0 < len(results) <= 5

    def test_scores_non_negative_and_finite(self, indexed_corpus):
        results = retrieve_bm25(IN_CORPUS_QUERY, top_k=10)
        for r in results:
            assert math.isfinite(r.score)
            assert r.score >= 0.0

    def test_scores_descending(self, indexed_corpus):
        results = retrieve_bm25(IN_CORPUS_QUERY, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_text_returned(self, indexed_corpus):
        results = retrieve_bm25(KEYWORD_QUERY, top_k=5)
        assert any(KEYWORD in r.text.lower() for r in results), (
            f"No top-5 BM25 result contains '{KEYWORD}' for query '{KEYWORD_QUERY}'"
        )


# ── 4. Hybrid retrieval ───────────────────────────────────────────────────
class TestHybridRetrieval:
    """Semantic + BM25 fused via Reciprocal Rank Fusion."""

    def test_returns_results(self, qdrant_client, indexed_corpus):
        results = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=5)
        assert 0 < len(results) <= 5

    def test_rrf_scores_positive(self, qdrant_client, indexed_corpus):
        results = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=10)
        for r in results:
            assert math.isfinite(r.score)
            assert r.score > 0.0

    def test_draws_from_both_methods(self, qdrant_client, indexed_corpus):
        # Both underlying methods must independently produce candidates.
        sem = retrieve(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=20)
        bm25 = retrieve_bm25(IN_CORPUS_QUERY, top_k=20)
        assert len(sem) > 0 and len(bm25) > 0

        # Every fused result must trace back to one of the two source methods —
        # RRF fuses existing candidates, it never invents new chunks.
        source_ids = {r.chunk_id for r in sem} | {r.chunk_id for r in bm25}
        hybrid = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=10)
        for r in hybrid:
            assert r.chunk_id in source_ids


# ── 5. Reranker ───────────────────────────────────────────────────────────
class TestReranker:
    """Cross-encoder rescoring of hybrid candidates."""

    def test_returns_all_candidates(self, qdrant_client, indexed_corpus):
        candidates = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=20)
        reranked = rerank(IN_CORPUS_QUERY, candidates)
        assert len(reranked) == len(candidates)

    def test_scores_finite(self, qdrant_client, indexed_corpus):
        # Cross-encoder scores are logits and may be negative — only finiteness
        # is guaranteed, not sign.
        candidates = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=20)
        reranked = rerank(IN_CORPUS_QUERY, candidates)
        assert all(math.isfinite(r.score) for r in reranked)

    def test_sorted_by_score(self, qdrant_client, indexed_corpus):
        candidates = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=20)
        reranked = rerank(IN_CORPUS_QUERY, candidates)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_changes_ordering(self, qdrant_client, indexed_corpus):
        candidates = retrieve_hybrid(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=20)
        before = [c.chunk_id for c in candidates]
        after = [c.chunk_id for c in rerank(IN_CORPUS_QUERY, candidates)]
        # Reranking re-scores the same chunks — same set, no additions or drops.
        assert set(before) == set(after)
        # With 20 candidates, the cross-encoder reproducing the RRF order
        # exactly is effectively impossible; it should reorder them.
        assert before != after

    def test_end_to_end_reranker_path(self, qdrant_client, indexed_corpus):
        results = retrieve_with_reranker(IN_CORPUS_QUERY, qdrant_client=qdrant_client, top_k=5)
        assert 0 < len(results) <= 5
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert math.isfinite(r.score)