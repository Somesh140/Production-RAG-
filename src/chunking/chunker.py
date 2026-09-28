"""
Chunking strategies.

TODO (Activity 2.1): You implement split_text(), which covers the three
chunking strategies. Chunk, chunk_document() and chunk_corpus() are
provided.

The POC embedded whole documents. That wrecks retrieval. A 1,500-word policy
returned as one hit buries the two relevant sentences in noise, and it
blows the model's context budget. Production RAG splits every document into
small, overlapping passages, so a query matches the passage that actually
answers it.

For a demo document like this course's, a chunk_size bigger than your
longest fact keeps that fact in one piece. But RecursiveCharacterTextSplitter
takes the biggest piece that fits, and it doesn't know where a fact ends.
It splits on the nearest paragraph, line or word boundary within budget,
not where a rule actually finishes. A fact with no paragraph break nearby
can still get split, even at a generous chunk_size. So size your chunks
from the fact lengths you've measured across your corpus, not from one
example.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.document_processor import ProcessedDocument
from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """One retrievable passage. Provided, complete.

    metadata has every field from the parent ProcessedDocument, plus:
      - chunk_index: 0-based position within the document
      - chunk_id:    f"{source}::{chunk_index}", unique across the corpus
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _chunking_config(base_path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["chunking"]


# --- IMPLEMENT THIS (Activity 2.1) -------------------------------------

def split_text(text: str, strategy: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split `text` into overlapping passages. Tier: 🟡 Intermediate.

    Support three strategies, all from `langchain_text_splitters`. Call the
    splitter's `.split_text(text)` and return the list it gives back. Raise
    ValueError for an unknown strategy name.

    Walk through together (steps 1-2):
    1. "recursive": RecursiveCharacterTextSplitter(chunk_size=chunk_size,
       chunk_overlap=chunk_overlap). It splits on paragraph, then line, then
       word boundaries. This is the sensible default. We do this one as a
       group, so everyone sees the splitter-and-return pattern before
       working solo.
    2. The `else: raise ValueError(...)` branch for an unrecognized
       strategy name. It's one line, and we do it in the same group
       walkthrough.

    The instructor narrates this part live, so everyone leaves with the same
    mental model before touching code on their own. If you ask AI here, use
    it to explain what the instructor just showed. Don't use it to produce
    the next step before it's been demoed. If you grab the shortcut, you
    skip the one part that's meant to be watched, not typed.

    Your turn (steps 3-4):
    3. "fixed": CharacterTextSplitter(separator="", chunk_size=chunk_size,
       chunk_overlap=chunk_overlap). This is a blunt fixed-width cut, and
       it's handy as a baseline to compare against.
    4. "markdown": MarkdownTextSplitter(chunk_size=chunk_size,
       chunk_overlap=chunk_overlap). It keeps Markdown headings and sections
       intact.

    Use AI here in whatever way gets you to a working implementation you
    understand. Some of you will write a first pass alone and check it
    against a model's version afterwards. Others will ask AI for a draft
    and then trace through it line by line until it makes sense. Both are
    fine. What doesn't count as done is code you can't explain in your own
    words, whether AI wrote it or not.
    """
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


# --- PROVIDED, COMPLETE ----------------------------------------------

def chunk_document(doc: ProcessedDocument, config: Optional[Dict[str, Any]] = None,
                   base_path: Optional[str] = None) -> List[Chunk]:
    """Split one ProcessedDocument into Chunks. Each chunk inherits the
    document's metadata and gets its own chunk_index and chunk_id."""
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
    """Chunk every document in the corpus."""
    cfg = config or _chunking_config(base_path=base_path)
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg))
    logger.info(f"Produced {len(all_chunks)} chunks from {len(docs)} documents "
                f"(strategy={cfg.get('strategy')}, size={cfg.get('chunk_size')})")
    return all_chunks
