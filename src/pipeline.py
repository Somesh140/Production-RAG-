"""
End-to-end Enterprise Knowledge System pipeline. Provided, complete.

Connects the modules from each activity into two entry points:
  - build_index():        ingest -> process -> chunk -> embed -> persist
  - answer_question(q):    retrieve (hybrid + metadata) -> grounded, cited answer

The Streamlit portal, the concept explorer, the LangSmith tracing wrapper
(Activity 3.1) and Final Review all call these two functions. None of them
wire up the stages again themselves. Until you've finished the relevant
activities, the imported functions raise NotImplementedError, and the
callers show that error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.chunking.chunker import chunk_corpus
from src.generation.answer_generator import generate_answer, validate_answer
from src.indexing.vector_store import index_chunks, load_store
from src.ingestion.document_processor import load_manifest, process_corpus
from src.ingestion.loaders import load_corpus
from src.retrieval.retriever import HybridRetriever
from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_corpus_chunks(base_path: Optional[str] = None) -> List[Any]:
    """Run ingestion -> processing -> chunking and return the chunk list.

    You'll normally leave `base_path` as None, so every lookup below
    resolves relative to the current working directory. Pass it to resolve
    config and knowledge-base lookups relative to an explicit directory
    instead, e.g. for app/demo_backend.py's demo-mode merged overlay, which
    can't change the process's working directory (its module docstring
    explains why)."""
    cfg = load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["ingestion"]
    raws = load_corpus(cfg["knowledge_base_path"], base_path=base_path)
    manifest = load_manifest(cfg["manifest_path"], base_path=base_path)
    processed = process_corpus(raws, manifest)
    return chunk_corpus(processed, base_path=base_path)


def build_index(base_path: Optional[str] = None) -> Dict[str, Any]:
    """Full offline build: chunk the corpus and save it to the vector store.
    Returns a small summary dict."""
    chunks = build_corpus_chunks(base_path=base_path)
    index_chunks(chunks, base_path=base_path)
    logger.info(f"Index build complete: {len(chunks)} chunks")
    return {"chunks_indexed": len(chunks)}


class KnowledgeSystem:
    """Query-time handle: a vector store plus the in-memory chunk list that
    BM25 needs. Build it once, then ask as many questions as you like.

    With reuse_persisted=True it tries to open a saved index first, which
    is fast on repeat runs. If none exists yet, it builds one from the
    corpus and uses that handle directly.
    """

    def __init__(self, reuse_persisted: bool = False, base_path: Optional[str] = None):
        self.base_path = base_path
        self.chunks = build_corpus_chunks(base_path=base_path)
        # By default, build the index from the corpus right here. Chunk ids
        # are stable, so this upserts instead of duplicating. Pass
        # reuse_persisted=True to open an index you've already built (faster,
        # e.g. inside the container after `build_index()` has run once).
        if reuse_persisted:
            try:
                self.store = load_store(base_path=base_path)
            except Exception as e:  # noqa: BLE001
                logger.info(f"No usable persisted index ({e}); building a fresh one")
                self.store = index_chunks(self.chunks, base_path=base_path)
        else:
            self.store = index_chunks(self.chunks, base_path=base_path)
        self.retriever = HybridRetriever(self.store, self.chunks, base_path=base_path)

    def answer_question(self, question: str,
                        metadata_filter: Optional[Dict[str, Any]] = None,
                        hybrid: Optional[bool] = None) -> Dict[str, Any]:
        hits = self.retriever.retrieve(question, metadata_filter=metadata_filter, hybrid=hybrid)
        result = generate_answer(question, hits, base_path=self.base_path)
        result["validation_errors"] = validate_answer(result, base_path=self.base_path)
        return result


def answer_question(question: str, base_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """One-shot shortcut: build a KnowledgeSystem and answer a single
    question. For repeated queries, use KnowledgeSystem directly."""
    return KnowledgeSystem(base_path=base_path).answer_question(question, **kwargs)
