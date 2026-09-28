"""
Retrieval optimization: metadata filtering + hybrid (vector + BM25) search.

TODO (Activity 2.2): You implement HybridRetriever.retrieve(). The BM25
index, the vector delegate and score normalization are provided.

Pure vector search misses exact terms ("FOIR", "V-CIP", a circular number).
It also can't tell a current policy from a superseded one. Production
retrieval fixes both. It fuses a lexical (BM25) signal with the vector
signal, and it applies a metadata filter, so a Cards question gets answered
from current Cards documents.

One open question this activity doesn't resolve: chunk overlap (Activity 2.1)
means neighboring chunks from the same document share text. So two of the
top-k results can be near-duplicates of each other. That costs you answer
quality, because one citation slot gets spent repeating what another
citation already said. Filtering and fusion don't fix that, and we don't
fix it here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.chunking.chunker import Chunk
from src.indexing.vector_store import RetrievedChunk, search as vector_search
from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _retrieval_config(base_path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["retrieval"]


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Scale a dict of raw scores into 0..1. If all the scores are equal,
    every score becomes 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    """Combines a vector-similarity signal with a BM25 lexical signal. Both
    can optionally be scoped by a metadata filter."""

    def __init__(self, store, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None,
                 base_path: Optional[str] = None):
        self.store = store
        self.config = config or _retrieval_config(base_path=base_path)
        self.chunks = chunks
        self._by_id = {c.metadata.get("chunk_id", str(i)): c for i, c in enumerate(chunks)}

        from rank_bm25 import BM25Okapi

        self._ids = list(self._by_id.keys())
        self._bm25 = BM25Okapi([_tokenize(self._by_id[cid].content) for cid in self._ids])

    # --- provided helpers ------------------------------------------------

    def _bm25_scores(self, query: str, metadata_filter: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Raw BM25 score per chunk_id, limited to chunks that match the
        metadata filter (an exact match on each key)."""
        raw = self._bm25.get_scores(_tokenize(query))
        out: Dict[str, float] = {}
        for cid, score in zip(self._ids, raw):
            meta = self._by_id[cid].metadata
            if metadata_filter and any(meta.get(k) != v for k, v in metadata_filter.items()):
                continue
            out[cid] = float(score)
        return out

    def _vector_scores(self, query: str, top_k: int,
                       metadata_filter: Optional[Dict[str, Any]]) -> Dict[str, RetrievedChunk]:
        """Map chunk_id to RetrievedChunk, using hits from the vector store.
        The backend has already applied metadata_filter when one is given."""
        hits = vector_search(self.store, query, top_k=top_k, metadata_filter=metadata_filter)
        return {h.metadata.get("chunk_id", h.content[:32]): h for h in hits}

    # --- IMPLEMENT THIS (Activity 2.2) --------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None,
                 hybrid: Optional[bool] = None) -> List[RetrievedChunk]:
        """
        Return the best chunks for `query`. Tier: 🟡 Intermediate.

        Walk through together (steps 1-2):
        1. Resolve settings from self.config: top_k (the argument, or
           config["top_k"]), score_threshold, and the hybrid block
           (config["hybrid"] with enabled / vector_weight / lexical_weight).
           The `hybrid` argument overrides config["hybrid"]["enabled"] when
           it isn't None.
        2. Get the vector hits with self._vector_scores(query, top_k * 4,
           metadata_filter). Over-fetch so fusion has enough candidates to
           work with. Build vec_raw = {chunk_id: hit.score}.

        The instructor narrates this part live, so everyone leaves with the
        same mental model before touching code on their own. If you ask AI
        here, use it to explain what the instructor just showed. Don't use
        it to produce the next step before it's been demoed. If you grab
        the shortcut, you skip the one part that's meant to be watched, not
        typed.

        Your turn (steps 3-5):
        3. If hybrid is off: normalize vec_raw and attach the normalized
           score to each RetrievedChunk. Drop anything below
           score_threshold, sort descending and return the top_k.
        4. If hybrid is on: also get lex_raw = self._bm25_scores(query,
           metadata_filter). Min-max normalize vec_raw and lex_raw
           separately (use _minmax_normalize). For the union of chunk_ids,
           fused = vector_weight * nvec.get(id, 0) + lexical_weight *
           nlex.get(id, 0). If an id shows up only in BM25, build a
           RetrievedChunk for it from self._by_id[id] (content + metadata).
        5. Set each RetrievedChunk.score to its fused score. Drop anything
           below score_threshold, sort descending and return the top_k.

        Use AI here in whatever way gets you to a working implementation
        you understand. Some of you will write a first pass alone and check
        it against a model's version afterwards. Others will ask AI for a
        draft and then trace through it line by line until it makes sense.
        Both are fine. What doesn't count as done is code you can't explain
        in your own words, whether AI wrote it or not.
        """
        cfg = self.config
        k = top_k or cfg.get("top_k", 5)
        threshold = cfg.get("score_threshold", 0.0)
        hcfg = cfg.get("hybrid", {}) or {}
        use_hybrid = hcfg.get("enabled", False) if hybrid is None else hybrid
        w_vec = hcfg.get("vector_weight", 0.6)
        w_lex = hcfg.get("lexical_weight", 0.4)

        vec_hits = self._vector_scores(query, k * 4, metadata_filter)
        vec_raw = {cid: h.score for cid, h in vec_hits.items()}

        if not use_hybrid:
            nvec = _minmax_normalize(vec_raw)
            results = []
            for cid, hit in vec_hits.items():
                hit.score = nvec.get(cid, 0.0)
                if hit.score >= threshold:
                    results.append(hit)
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:k]

        lex_raw = self._bm25_scores(query, metadata_filter)
        nvec = _minmax_normalize(vec_raw)
        nlex = _minmax_normalize(lex_raw)

        fused: Dict[str, float] = {}
        for cid in set(nvec) | set(nlex):
            fused[cid] = w_vec * nvec.get(cid, 0.0) + w_lex * nlex.get(cid, 0.0)

        results: List[RetrievedChunk] = []
        for cid, score in fused.items():
            if score < threshold:
                continue
            if cid in vec_hits:
                hit = vec_hits[cid]
                hit.score = score
            else:
                chunk = self._by_id[cid]
                hit = RetrievedChunk(content=chunk.content, metadata=dict(chunk.metadata), score=score)
            results.append(hit)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
