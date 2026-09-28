"""Chunking strategies - REFERENCE SOLUTION (Activity 2.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.document_processor import ProcessedDocument
from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _chunking_config(base_path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["chunking"]


def split_text(text: str, strategy: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        MarkdownTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "fixed":
        splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "markdown":
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    return splitter.split_text(text)


def chunk_document(doc: ProcessedDocument, config: Optional[Dict[str, Any]] = None,
                   base_path: Optional[str] = None) -> List[Chunk]:
    cfg = config or _chunking_config(base_path=base_path)
    pieces = split_text(
        doc.content,
        strategy=cfg.get("strategy", "recursive"),
        chunk_size=cfg.get("chunk_size", 800),
        chunk_overlap=cfg.get("chunk_overlap", 120),
    )
    source = doc.metadata.get("source", "unknown")
    chunks: List[Chunk] = []
    for i, piece in enumerate(pieces):
        if not piece.strip():
            continue
        meta = dict(doc.metadata)
        meta["chunk_index"] = i
        meta["chunk_id"] = f"{source}::{i}"
        chunks.append(Chunk(content=piece, metadata=meta))
    return chunks


def chunk_corpus(docs: List[ProcessedDocument],
                 config: Optional[Dict[str, Any]] = None,
                 base_path: Optional[str] = None) -> List[Chunk]:
    cfg = config or _chunking_config(base_path=base_path)
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg))
    logger.info(f"Produced {len(all_chunks)} chunks from {len(docs)} documents "
                f"(strategy={cfg.get('strategy')}, size={cfg.get('chunk_size')})")
    return all_chunks
