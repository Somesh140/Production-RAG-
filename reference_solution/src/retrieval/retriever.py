"""Retrieval optimization: metadata filter + hybrid - REFERENCE SOLUTION (Activity 2.2)."""
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
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, store, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None,
                 base_path: Optional[str] = None):
        self.store = store
        self.config = config or _retrieval_config(base_path=base_path)
        self.chunks = chunks
        self._by_id = {c.metadata.get("chunk_id", str(i)): c for i, c in enumerate(chunks)}

        from rank_bm25 import BM25Okapi

        self._ids = list(self._by_id.keys())
        self._bm25 = BM25Okapi([_tokenize(self._by_id[cid].content) for cid in self._ids])

    def _bm25_scores(self, query: str, metadata_filter: Optional[Dict[str, Any]]) -> Dict[str, float]:
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
        hits = vector_search(self.store, query, top_k=top_k, metadata_filter=metadata_filter)
        return {h.metadata.get("chunk_id", h.content[:32]): h for h in hits}

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None,
                 hybrid: Optional[bool] = None) -> List[RetrievedChunk]:
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
