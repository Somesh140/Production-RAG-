"""Vector store over Chroma / FAISS - REFERENCE SOLUTION (Activity 2.1)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunking.chunker import Chunk
from src.utils.config_loader import load_yaml_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Retry policy for the embedding-triggering store call in index_chunks. A
# transient rate limit or network blip should not force restarting the whole
# corpus build. For a corpus small enough to embed in one internal Chroma
# batch (this course's), a mid-build failure happens before anything is
# upserted, so a retry re-pays for nothing that already succeeded; for a
# corpus large enough to span multiple internal batches, a failure in a
# later batch means the retry re-embeds already-upserted earlier batches too
# (harmless via upsert-by-id, but not free). Auth/config errors are not
# retried - they will not resolve themselves and retrying just delays the
# real failure.
_EMBED_RETRY_ATTEMPTS = 5
_EMBED_RETRY_BASE_DELAY_SECONDS = 1.0


@dataclass
class RetrievedChunk:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")


def _vs_config(base_path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["vector_store"]


def _backend_name(override: Optional[str] = None, base_path: Optional[str] = None) -> str:
    return (override or os.getenv("VECTOR_STORE") or _vs_config(base_path=base_path).get("backend", "chroma")).lower()


def _make_embeddings(base_path: Optional[str] = None):
    from langchain_openai import OpenAIEmbeddings

    llm_cfg = load_yaml_config("config/llm_config.yaml", base_path=base_path)["models"]["embedding"]
    return OpenAIEmbeddings(
        model=llm_cfg["model_name"],
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


def _make_store(backend: str, documents=None, ids=None, base_path: Optional[str] = None):
    cfg = _vs_config(base_path=base_path)
    persist_dir = os.getenv("VECTOR_STORE_DIR") or cfg.get("persist_directory", "vector_store")
    persist_dir = str(resolve_path(persist_dir, base_path))
    collection = cfg.get("collection_name", "enterprise_kb")
    embeddings = _make_embeddings(base_path=base_path)

    if backend == "chroma":
        from langchain_chroma import Chroma

        if documents is not None:
            # Stable ids make a re-run an upsert instead of a duplicate insert.
            return Chroma.from_documents(
                documents=documents, embedding=embeddings, ids=ids,
                collection_name=collection, persist_directory=persist_dir,
            )
        return Chroma(
            collection_name=collection, embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    if backend == "faiss":
        from langchain_community.vectorstores import FAISS

        faiss_dir = str(Path(persist_dir) / "faiss")
        if documents is not None:
            store = FAISS.from_documents(documents=documents, embedding=embeddings, ids=ids)
            Path(faiss_dir).mkdir(parents=True, exist_ok=True)
            store.save_local(faiss_dir)
            return store
        return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

    raise ValueError(f"Unknown vector store backend: {backend!r} (expected 'chroma' or 'faiss')")


def load_store(backend: Optional[str] = None, base_path: Optional[str] = None):
    return _make_store(_backend_name(backend, base_path=base_path), base_path=base_path)


def build_documents(chunks: List[Chunk]) -> List[Any]:
    from langchain_core.documents import Document

    def scalar(v):
        return v if isinstance(v, (str, int, float, bool)) else str(v)

    return [
        Document(
            page_content=c.content,
            metadata={k: scalar(v) for k, v in c.metadata.items()},
        )
        for c in chunks
    ]


def _make_store_with_retry(backend: str, documents, ids,
                            attempts: int = _EMBED_RETRY_ATTEMPTS,
                            base_delay: float = _EMBED_RETRY_BASE_DELAY_SECONDS,
                            base_path: Optional[str] = None):
    """Call _make_store (which triggers embed_documents against the OpenAI
    API) with retry-with-backoff on transient failures.

    Retried: rate limits, connection drops/timeouts, and 5xx responses from
    the embeddings API - all of these can succeed if we just try again.
    Not retried: everything else (auth failures, bad requests, invalid
    config, etc.) - those fail the same way every time, so retrying only
    burns time and quota. They propagate on the first attempt.
    """
    from openai import APIConnectionError, InternalServerError, RateLimitError

    transient_errors = (RateLimitError, APIConnectionError, InternalServerError)

    for attempt in range(1, attempts + 1):
        try:
            return _make_store(backend, documents=documents, ids=ids, base_path=base_path)
        except transient_errors as e:
            if attempt == attempts:
                logger.error(
                    f"Embedding call failed after {attempts} attempts "
                    f"({type(e).__name__}: {e}); giving up"
                )
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Embedding call failed ({type(e).__name__}: {e}); "
                f"retrying (attempt {attempt}/{attempts}) in {delay:.1f}s"
            )
            time.sleep(delay)


def index_chunks(chunks: List[Chunk], backend: Optional[str] = None, base_path: Optional[str] = None):
    resolved = _backend_name(backend, base_path=base_path)
    documents = build_documents(chunks)
    ids = [c.metadata.get("chunk_id") or str(i) for i, c in enumerate(chunks)]
    store = _make_store_with_retry(resolved, documents, ids, base_path=base_path)
    logger.info(f"Indexed {len(documents)} chunks into {resolved}")
    return store


def search(store, query: str, top_k: int = 5,
           metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
    kwargs = {"k": top_k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    pairs = store.similarity_search_with_relevance_scores(query, **kwargs)
    return [
        RetrievedChunk(content=doc.page_content, metadata=dict(doc.metadata), score=float(score))
        for doc, score in pairs
    ]
