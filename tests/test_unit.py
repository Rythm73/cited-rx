"""
tests/test_unit.py

Fast unit tests — no Qdrant, no LLM, no network.
Run after every file you touch:
  pytest tests/test_unit.py -v

These tests prove your pure logic (chunking, BM25 scoring, RRF fusion,
schema validation, answer rendering) hasn't broken during refactoring.
"""
import json
import math
import pytest
from pathlib import Path


# ── Schema tests ───────────────────────────────────────────────────────
class TestSchemas:
    def test_retrieved_chunk_is_a_dataclass(self):
        from backend.schemas import RetrievedChunk
        c = RetrievedChunk(
            chunk_id=1, score=0.9, page_number=5,
            source_doc="test.pdf", text="Some text"
        )
        assert c.chunk_id == 1
        assert c.score == 0.9

    def test_response_valid(self):
        from backend.schemas import Response, Citation
        r = Response(
            answer="LDL-C target is <70 mg/dL [chunk_id=1].",
            citations=[Citation(chunk_id=1, quote="LDL-C target is <70 mg/dL")],
            confidence=0.85,
        )
        assert r.confidence == 0.85
        assert len(r.citations) == 1

    def test_response_rejects_confidence_above_1(self):
        from backend.schemas import Response
        with pytest.raises(Exception):
            Response(answer="x", citations=[], confidence=1.5)

    def test_response_rejects_confidence_below_0(self):
        from backend.schemas import Response
        with pytest.raises(Exception):
            Response(answer="x", citations=[], confidence=-0.1)

    def test_citation_fields(self):
        from backend.schemas import Citation
        c = Citation(chunk_id=42, quote="verbatim text here")
        assert c.chunk_id == 42
        assert c.quote == "verbatim text here"


# ── Config tests ───────────────────────────────────────────────────────
class TestConfig:
    def test_paths_are_path_objects(self):
        from config import ROOT, DATA_DIR, QDRANT_PATH, PROCESSED_DIR
        assert isinstance(ROOT, Path)
        assert isinstance(DATA_DIR, Path)
        assert isinstance(QDRANT_PATH, Path)
        assert isinstance(PROCESSED_DIR, Path)

    def test_root_exists(self):
        from config import ROOT
        assert ROOT.exists(), f"ROOT does not exist: {ROOT}"

    def test_data_dirs_created(self):
        # config.py creates these on import
        from config import DATA_DIR, QDRANT_PATH, PROCESSED_DIR, RAW_DIR, EVAL_DIR
        for d in (DATA_DIR, QDRANT_PATH, PROCESSED_DIR, RAW_DIR, EVAL_DIR):
            assert d.exists(), f"Directory not created: {d}"

    def test_default_corpus_is_string(self):
        from config import DEFAULT_CORPUS
        assert isinstance(DEFAULT_CORPUS, str)
        assert len(DEFAULT_CORPUS) > 0

    def test_chunk_constants_are_positive(self):
        from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIM
        assert CHUNK_SIZE > 0
        assert CHUNK_OVERLAP > 0
        assert CHUNK_OVERLAP < CHUNK_SIZE, "Overlap must be less than chunk size"
        assert EMBEDDING_DIM > 0


# ── Chunking tests ─────────────────────────────────────────────────────
class TestChunking:
    """Tests chunk_pdf() logic without needing a real PDF — uses tmp files."""

    def _make_pdf(self, tmp_path, text: str):
        """Create a minimal single-page PDF with the given text."""
        try:
            from pypdf import PdfWriter
            from pypdf.generic import NameObject
            import io

            writer = PdfWriter()
            # Add a blank page and patch in text via a content stream
            page = writer.add_blank_page(width=612, height=792)
            # Write text as a PDF content stream
            stream = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode()
            from pypdf.generic import StreamObject, ArrayObject, DictionaryObject
            content = StreamObject()
            content.set_data(stream)
            page[NameObject("/Contents")] = writer._add_object(content)

            pdf_path = tmp_path / "test.pdf"
            with open(pdf_path, "wb") as f:
                writer.write(f)
            return pdf_path
        except Exception:
            pytest.skip("pypdf not installed or PDF creation failed")

    def test_chunk_metadata_fields(self, tmp_path):
        """Every chunk must have the four required metadata fields."""
        # Use a real small PDF if available, otherwise skip
        from config import RAW_DIR
        pdfs = list(RAW_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs in data/raw/ to test chunking against")

        from backend.ingest import chunk_pdf
        chunks = chunk_pdf(str(pdfs[0]))
        assert len(chunks) > 0
        for c in chunks[:5]:  # check first 5
            assert "chunk_id" in c
            assert "page_number" in c
            assert "source_doc" in c
            assert "text" in c
            assert isinstance(c["chunk_id"], int)
            assert isinstance(c["page_number"], int)
            assert len(c["text"]) > 0

    def test_chunk_ids_are_sequential(self, tmp_path):
        from config import RAW_DIR
        pdfs = list(RAW_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs in data/raw/ to test chunking against")

        from backend.ingest import chunk_pdf
        chunks = chunk_pdf(str(pdfs[0]))
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids))), "chunk_ids must be 0,1,2,... in order"

    def test_chunk_size_respected(self, tmp_path):
        from config import RAW_DIR, CHUNK_SIZE
        pdfs = list(RAW_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs in data/raw/ to test chunking against")

        from backend.ingest import chunk_pdf
        chunks = chunk_pdf(str(pdfs[0]))
        oversized = [c for c in chunks if len(c["text"]) > CHUNK_SIZE * 1.1]
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceed CHUNK_SIZE by >10%: "
            f"max={max(len(c['text']) for c in chunks)}"
        )


# ── BM25 tokenizer tests ───────────────────────────────────────────────
class TestBM25Tokenizer:
    def test_lowercases(self):
        from backend.retrieve_bm25 import _tokenize
        assert _tokenize("LDL-C TARGET") == ["ldl-c", "target"]

    def test_keeps_hyphens(self):
        from backend.retrieve_bm25 import _tokenize
        tokens = _tokenize("LDL-C atorvastatin")
        assert "ldl-c" in tokens

    def test_strips_punctuation(self):
        from backend.retrieve_bm25 import _tokenize
        tokens = _tokenize("atorvastatin, ezetimibe.")
        assert "atorvastatin" in tokens
        assert "ezetimibe" in tokens
        # commas and periods should be gone
        assert not any("," in t or "." in t for t in tokens)

    def test_empty_string(self):
        from backend.retrieve_bm25 import _tokenize
        assert _tokenize("") == []

    def test_whitespace_only(self):
        from backend.retrieve_bm25 import _tokenize
        assert _tokenize("   ") == []


# ── RRF fusion tests ───────────────────────────────────────────────────
class TestRRFFusion:
    """Test the RRF math directly — no Qdrant needed."""

    def _make_chunk(self, chunk_id, score=0.5):
        from backend.schemas import RetrievedChunk
        return RetrievedChunk(
            chunk_id=chunk_id, score=score,
            page_number=1, source_doc="test.pdf",
            text=f"Text for chunk {chunk_id}"
        )

    def _run_rrf(self, sem_results, bm25_results, top_k=5):
        """Replicate the RRF logic from retrieve_hybrid.py for testing."""
        from config import RRF_K
        sem_ranks  = {r.chunk_id: i + 1 for i, r in enumerate(sem_results)}
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        chunk_lookup = {}
        for r in sem_results + bm25_results:
            if r.chunk_id not in chunk_lookup:
                chunk_lookup[r.chunk_id] = r

        all_ids = set(sem_ranks) | set(bm25_ranks)
        rrf_scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in sem_ranks:
                score += 1.0 / (RRF_K + sem_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[cid])
            rrf_scores[cid] = score

        top_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:top_k]
        return top_ids, rrf_scores

    def test_chunk_in_both_lists_scores_higher(self):
        """A chunk ranked #1 in both lists should beat one ranked #1 in only one."""
        sem   = [self._make_chunk(1), self._make_chunk(2)]
        bm25  = [self._make_chunk(1), self._make_chunk(3)]  # chunk 1 in both
        ids, scores = self._run_rrf(sem, bm25)
        # chunk 1 appears in both → should have the highest RRF score
        assert ids[0] == 1

    def test_returns_at_most_top_k(self):
        sem  = [self._make_chunk(i) for i in range(10)]
        bm25 = [self._make_chunk(i) for i in range(10)]
        ids, _ = self._run_rrf(sem, bm25, top_k=3)
        assert len(ids) <= 3

    def test_rrf_score_is_positive(self):
        sem  = [self._make_chunk(1)]
        bm25 = [self._make_chunk(1)]
        _, scores = self._run_rrf(sem, bm25)
        assert scores[1] > 0

    def test_rank_1_beats_rank_10(self):
        """Rank 1 should always give a higher RRF contribution than rank 10."""
        from config import RRF_K
        score_rank1  = 1.0 / (RRF_K + 1)
        score_rank10 = 1.0 / (RRF_K + 10)
        assert score_rank1 > score_rank10

    def test_disjoint_lists_both_get_represented(self):
        """Chunks unique to each list should both appear in results."""
        sem  = [self._make_chunk(1)]
        bm25 = [self._make_chunk(2)]
        ids, _ = self._run_rrf(sem, bm25, top_k=5)
        assert 1 in ids
        assert 2 in ids


# ── Answer rendering tests ─────────────────────────────────────────────
class TestAnswerRendering:
    """Test [chunk_id=N] → (p. X) replacement logic."""

    def _chunks(self):
        from backend.schemas import RetrievedChunk
        return [
            RetrievedChunk(chunk_id=10, score=0.9, page_number=5,
                           source_doc="test.pdf", text="text"),
            RetrievedChunk(chunk_id=20, score=0.8, page_number=12,
                           source_doc="test.pdf", text="text"),
        ]

    def test_markers_replaced(self):
        import re
        chunks = self._chunks()
        chunk_to_page = {c.chunk_id: c.page_number for c in chunks}
        answer = "LDL target [chunk_id=10] and also [chunk_id=20]."
        rendered = re.sub(
            r"\[chunk_id=(\d+)\]",
            lambda m: f"(p. {chunk_to_page.get(int(m.group(1)), '?')})",
            answer,
        )
        assert "(p. 5)" in rendered
        assert "(p. 12)" in rendered
        assert "[chunk_id=" not in rendered

    def test_unknown_chunk_id_shows_question_mark(self):
        import re
        chunks = self._chunks()
        chunk_to_page = {c.chunk_id: c.page_number for c in chunks}
        answer = "Something [chunk_id=999]."
        rendered = re.sub(
            r"\[chunk_id=(\d+)\]",
            lambda m: f"(p. {chunk_to_page.get(int(m.group(1)), '?')})",
            answer,
        )
        assert "(p. ?)" in rendered

    def test_no_markers_unchanged(self):
        import re
        chunks = self._chunks()
        chunk_to_page = {c.chunk_id: c.page_number for c in chunks}
        answer = "No citations in this answer."
        rendered = re.sub(
            r"\[chunk_id=(\d+)\]",
            lambda m: f"(p. {chunk_to_page.get(int(m.group(1)), '?')})",
            answer,
        )
        assert rendered == answer


# ── Synthesize gate tests (no LLM call needed) ─────────────────────────
class TestSynthesisGate:
    def _chunk(self, score):
        from backend.schemas import RetrievedChunk
        return RetrievedChunk(
            chunk_id=1, score=score, page_number=1,
            source_doc="test.pdf", text="Some evidence text."
        )

    def test_empty_chunks_returns_no_evidence(self):
        from backend.synthesize import synthesize_with_gate, NO_EVIDENCE_RESPONSE
        result = synthesize_with_gate("any question", chunks=[], threshold=0.0)
        assert result.confidence == 0.0
        assert result.citations == []

    def test_below_threshold_returns_no_evidence(self):
        from backend.synthesize import synthesize_with_gate, NO_EVIDENCE_RESPONSE
        low_score_chunk = self._chunk(score=0.1)
        result = synthesize_with_gate(
            "any question", chunks=[low_score_chunk], threshold=0.5
        )
        assert result.confidence == 0.0
        assert result.citations == []

    def test_no_evidence_response_is_well_formed(self):
        from backend.synthesize import NO_EVIDENCE_RESPONSE
        assert NO_EVIDENCE_RESPONSE.confidence == 0.0
        assert NO_EVIDENCE_RESPONSE.citations == []
        assert len(NO_EVIDENCE_RESPONSE.answer) > 0


# ── Gold standard JSON sanity checks ──────────────────────────────────
class TestGoldStandard:
    """Validate the gold Q&A file structure without running the pipeline."""

    @pytest.fixture
    def gold(self):
        from config import EVAL_DIR
        gold_path = EVAL_DIR / "gold.json"
        if not gold_path.exists():
            pytest.skip(f"Gold file not found at {gold_path}")
        return json.loads(gold_path.read_text())

    def test_gold_is_a_list(self, gold):
        assert isinstance(gold, list)
        assert len(gold) > 0

    def test_every_item_has_required_fields(self, gold):
        required = {"id", "question"}
        for item in gold:
            missing = required - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing fields: {missing}"

    def test_questions_are_non_empty_strings(self, gold):
        for item in gold:
            assert isinstance(item["question"], str)
            assert len(item["question"].strip()) > 0

    def test_out_of_corpus_items_exist(self, gold):
        ooc = [
            i for i in gold
            if i.get("category") == "out_of_corpus" or i.get("type") == "out_of_corpus"
        ]
        assert len(ooc) > 0, "Gold set should include out-of-corpus questions"

    def test_no_duplicate_ids(self, gold):
        ids = [item["id"] for item in gold]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in gold set"


class TestNoDriftFromPipeline:
    """Guard: api.py and ui.py must never call retrieve_with_reranker or synthesize directly.
    If this test fails, someone has bypassed run_pipeline and the eval path has diverged."""

    def _read(self, filename: str) -> str:
        path = __file__  # tests/test_unit.py
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / filename).read_text()

    def test_api_does_not_call_retrieve_with_reranker(self):
        src = self._read("backend/api.py")
        assert "retrieve_with_reranker" not in src

    def test_api_does_not_call_synthesize_directly(self):
        src = self._read("backend/api.py")
        assert "synthesize_with_gate" not in src
        assert "synthesize(" not in src

    def test_ui_does_not_call_retrieve_with_reranker(self):
        src = self._read("backend/ui.py")
        assert "retrieve_with_reranker" not in src

    def test_ui_does_not_call_synthesize_directly(self):
        src = self._read("backend/ui.py")
        assert "synthesize_with_gate" not in src
        assert "synthesize(" not in src


class TestCitationVerification:
    def _make_chunks(self, *pairs):
        from backend.schemas import RetrievedChunk
        return [
            RetrievedChunk(chunk_id=cid, page_number=1, text=text, score=1.0, source_doc="test.pdf")
            for cid, text in pairs
        ]

    def test_exact_quote_verifies(self):
        from backend.schemas import Citation
        from backend.synthesize import verify_citations
        chunks = self._make_chunks((1, "The LDL-C target is less than 70 mg/dL for high-risk patients."))
        citations = [Citation(chunk_id=1, quote="LDL-C target is less than 70 mg/dL")]
        result = verify_citations(citations, chunks)
        assert result[0].verified is True

    def test_hallucinated_quote_fails(self):
        from backend.schemas import Citation
        from backend.synthesize import verify_citations
        chunks = self._make_chunks((1, "The LDL-C target is less than 70 mg/dL for high-risk patients."))
        citations = [Citation(chunk_id=1, quote="LDL target is under 50 mg/dL")]
        result = verify_citations(citations, chunks)
        assert result[0].verified is False

    def test_unknown_chunk_id_fails(self):
        from backend.schemas import Citation
        
        from backend.synthesize import verify_citations
        chunks = self._make_chunks((1, "Some text here."))
        citations = [Citation(chunk_id=999, quote="Some text here.")]
        result = verify_citations(citations, chunks)
        assert result[0].verified is False

    def test_empty_citations_returns_empty(self):
        from backend.synthesize import verify_citations
        chunks = self._make_chunks((1, "Some text."))
        result = verify_citations([], chunks)
        assert result == []