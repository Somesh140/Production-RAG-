"""Retrieval fusion/filtering suite (Activity 2.2) - REFERENCE SOLUTION.

Small, fixed, offline: exercises HybridRetriever.retrieve()'s filter and
fusion logic with a deterministic fake vector store and the real (but tiny,
in-memory) BM25 index - no API key, no network, no real Chroma/FAISS. Same
"test the contract, not the content" philosophy as
reference_solution/tests/test_pipeline.py: checks shape/logic (filter
correctness, union behavior, threshold behavior), never embedding/LLM
quality, and prefers loose assertions (presence/absence, set equality,
ordering) over exact fused scores.

Placement note: this file lives under reference_solution/tests/, not
tests/, on purpose. The project root conftest.py already installs
reference_solution/src's modules under their bare `src.*` names for any
file collected from inside reference_solution/tests/ (see its
pytest_collectstart), and removes that override for every other file. That
per-file mechanism is exactly what a file needing `from src.retrieval...`
to resolve to the finished reference implementation (rather than the
learner's NotImplementedError stub in the top-level src/) requires, so
putting this file here gets bare `src.*` imports for free with no conftest
changes and nothing new to fight the existing install/uninstall logic. A
copy under tests/ would need its own separate resolution mechanism (e.g. a
second conftest scoped to tests/) duplicating what already exists.

scripts/verify_reference.py's build_merged() also copies this file into
its merged temp project, and main()'s pytest invocation runs it alongside
test_pipeline.py, so `python scripts/verify_reference.py` exercises both.

Run directly:

    pytest reference_solution/tests/test_retrieval.py -v
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.chunking.chunker import Chunk
from src.retrieval.retriever import HybridRetriever


class _Doc:
    """Minimal stand-in for a langchain Document: .page_content / .metadata."""

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata


class _FakeStore:
    """Deterministic double for the object vector_store.search() calls.

    search() (reference_solution/src/indexing/vector_store.py) only ever
    calls store.similarity_search_with_relevance_scores(query, k=top_k,
    filter=metadata_filter) and expects a list of (doc, score) pairs back.
    This mirrors the _Store pattern already used in
    scripts/verify_reference.py's OFFLINE_CHECK, but takes an explicit,
    caller-supplied score per chunk_id instead of a word-overlap heuristic,
    so each test's "vector signal" is exact and independent of tokenization
    quirks. A chunk_id simply absent from `scores` is never returned by the
    vector side at all (score 0 / not in the candidate pool), which is what
    the union-vs-intersection test needs.
    """

    def __init__(self, chunks: List[Chunk], scores: Dict[str, float]):
        self._by_id = {c.metadata["chunk_id"]: c for c in chunks}
        self._scores = scores

    def similarity_search_with_relevance_scores(self, query, k=5, filter=None):
        out = []
        for cid, score in self._scores.items():
            chunk = self._by_id[cid]
            if filter and any(chunk.metadata.get(mk) != mv for mk, mv in filter.items()):
                continue
            out.append((_Doc(chunk.content, dict(chunk.metadata)), score))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out[:k]


def _chunk(chunk_id: str, content: str, **metadata: Any) -> Chunk:
    metadata["chunk_id"] = chunk_id
    return Chunk(content=content, metadata=metadata)


def _hybrid_config(vector_weight: float = 0.6, lexical_weight: float = 0.4,
                    score_threshold: float = 0.0, top_k: int = 5) -> Dict[str, Any]:
    return {
        "top_k": top_k,
        "score_threshold": score_threshold,
        "hybrid": {"enabled": True, "vector_weight": vector_weight, "lexical_weight": lexical_weight},
    }


# --- 1. Filter is a hard gate, not a ranking nudge -------------------------

def test_filter_is_hard_gate_not_ranking_weight():
    """A chunk failing metadata_filter must never appear, no matter how the
    hybrid weights are set - encodes the course's "single most consequential
    error" distinction (filter vs. ranking weight) as an assertion."""
    hr_chunk = _chunk("hr::0", "annual leave policy annual leave carryover annual leave", department="hr")
    loan_chunk = _chunk("loans::0", "unrelated filler text about nothing in particular", department="loans")
    chunks = [hr_chunk, loan_chunk]

    # The vector store ranks hr_chunk far above loan_chunk - if the filter
    # were merely a ranking nudge, a high enough vector_weight could let it
    # survive into the results despite the department mismatch.
    scores = {"hr::0": 0.95, "loans::0": 0.1}

    for vector_weight, lexical_weight in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
        store = _FakeStore(chunks, scores)
        config = _hybrid_config(vector_weight=vector_weight, lexical_weight=lexical_weight)
        retriever = HybridRetriever(store, chunks, config=config)

        results = retriever.retrieve(
            "annual leave policy", metadata_filter={"department": "loans"}, hybrid=True,
        )
        ids = {r.metadata.get("chunk_id") for r in results}
        assert "loans::0" in ids, (
            f"allowed chunk was missing from results at "
            f"vector_weight={vector_weight}, lexical_weight={lexical_weight}"
        )
        assert "hr::0" not in ids, (
            f"filtered-out chunk leaked into results at "
            f"vector_weight={vector_weight}, lexical_weight={lexical_weight}"
        )


# --- 2. Hybrid is a union, not an intersection ------------------------------

def test_hybrid_is_union_not_intersection():
    """A chunk with zero vector signal (the fake vector store never returns
    it) but a strong BM25 match must still appear in the fused, filtered
    results - confirms union-of-pools + self._by_id materialization, not an
    intersection of what both signals happened to retrieve."""
    vector_only = _chunk("v::0", "the quick brown fox jumps over the lazy dog", topic="general")
    keyword_only = _chunk("k::0", "unobtainium quantum flux capacitor unobtainium", topic="general")
    filler = _chunk("f::0", "some other completely unrelated filler content here", topic="general")
    chunks = [vector_only, keyword_only, filler]

    # keyword_only is deliberately absent from `scores`: the vector side
    # never returns it, so it only reaches the results through the BM25 pool.
    scores = {"v::0": 0.9, "f::0": 0.2}
    store = _FakeStore(chunks, scores)
    config = _hybrid_config()
    retriever = HybridRetriever(store, chunks, config=config)

    results = retriever.retrieve(
        "unobtainium quantum flux capacitor", metadata_filter={"topic": "general"}, hybrid=True,
    )
    ids = {r.metadata.get("chunk_id") for r in results}
    assert "k::0" in ids, "chunk with zero vector score but a strong BM25 match was dropped from the union"


# --- 3. BM25-only and vector-only filter paths agree ------------------------

def test_vector_and_bm25_filter_paths_agree_on_candidate_set():
    """The backend-side vector filter and the hand-rolled BM25 filter are
    two independently-implemented code paths that can silently diverge (a
    type/case mismatch in one but not the other). For a simple case, both
    must include/exclude exactly the same candidates for the same filter
    key."""
    a = _chunk("a::0", "quarterly compliance audit checklist", region="east")
    b = _chunk("b::0", "quarterly compliance audit checklist", region="west")
    c = _chunk("c::0", "quarterly compliance audit checklist", region="east")
    chunks = [a, b, c]
    scores = {"a::0": 0.5, "b::0": 0.9, "c::0": 0.3}
    store = _FakeStore(chunks, scores)
    retriever = HybridRetriever(store, chunks, config=_hybrid_config())

    query = "quarterly compliance audit checklist"
    metadata_filter = {"region": "east"}

    vector_hits = retriever._vector_scores(query, top_k=10, metadata_filter=metadata_filter)
    bm25_hits = retriever._bm25_scores(query, metadata_filter=metadata_filter)

    expected = {"a::0", "c::0"}
    assert set(vector_hits.keys()) == expected, f"vector filter path disagreed: {set(vector_hits.keys())}"
    assert set(bm25_hits.keys()) == expected, f"BM25 filter path disagreed: {set(bm25_hits.keys())}"


# --- 4. score_threshold drops low-scoring candidates ------------------------

def test_score_threshold_drops_candidate_below_threshold():
    """A candidate whose (normalized) score falls below score_threshold must
    be excluded from the final results."""
    strong = _chunk("s::0", "annual leave carryover policy annual leave", dept="hr")
    weak = _chunk("w::0", "totally unrelated filler content about nothing", dept="hr")
    chunks = [strong, weak]
    scores = {"s::0": 1.0, "w::0": 0.01}
    store = _FakeStore(chunks, scores)
    config = _hybrid_config(score_threshold=0.5)
    retriever = HybridRetriever(store, chunks, config=config)

    results = retriever.retrieve("annual leave carryover policy", hybrid=False)
    ids = {r.metadata.get("chunk_id") for r in results}

    assert "s::0" in ids, "candidate above threshold was dropped"
    assert "w::0" not in ids, "candidate below score_threshold was not dropped"
