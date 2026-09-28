"""
Pipeline smoke suite (Activity 3.2).

A small, fixed, offline suite: it checks the *shape* of ingestion, metadata,
chunking, and citation handling - never LLM output quality - so it needs no
API key and no network. Run from the project root:

    pytest tests/test_pipeline.py -v

Two tests are prefilled. Implement the three marked TODO.
"""
from __future__ import annotations

from src.chunking.chunker import split_text
from src.generation.answer_generator import parse_citations
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.document_processor import load_manifest, process_corpus, process_document
from src.ingestion.loaders import RawDocument, _EXTENSION_LOADERS, load_corpus
from src.utils.config_loader import load_yaml_config

CFG = load_yaml_config("config/ingestion_config.yaml")


# --- Prefilled -------------------------------------------------------------

def test_loaders_support_all_configured_formats():
    """Every extension in ingestion.supported_extensions must have a loader."""
    for ext in CFG["ingestion"]["supported_extensions"]:
        assert ext in _EXTENSION_LOADERS, f"No loader registered for {ext}"


def test_ungoverned_document_is_excluded():
    """A document with no manifest row must be dropped by process_corpus,
    not indexed."""
    manifest = load_manifest(CFG["ingestion"]["manifest_path"])
    ungoverned = RawDocument(content="rogue draft", source="nowhere/rogue.md", file_type="md")
    out = process_corpus([ungoverned], manifest)
    assert out == [], "Ungoverned document should not survive processing"


# --- TODO (Activity 3.2) ------------------------------------------------

def test_all_governed_documents_have_required_metadata():
    """
    TODO: load the real corpus (load_corpus) and manifest (load_manifest),
    run process_corpus, and assert that:
      - at least 10 ProcessedDocuments come back
      - every one has a non-empty value for each name in
        CFG["metadata"]["required_fields"]
    """
    raws = load_corpus(CFG["ingestion"]["knowledge_base_path"])
    manifest = load_manifest(CFG["ingestion"]["manifest_path"])
    processed = process_corpus(raws, manifest)

    assert len(processed) >= 10
    required = CFG["metadata"]["required_fields"]
    for doc in processed:
        for field in required:
            assert str(doc.metadata.get(field, "")).strip(), \
                f"{doc.metadata.get('source')} missing metadata field {field}"


def test_chunking_respects_configured_size():
    """
    TODO: build a long string (e.g. "sentence number N. " repeated ~400
    times), call split_text(text, "recursive", chunk_size=300,
    chunk_overlap=40), and assert:
      - more than one chunk is produced
      - every chunk is non-empty
      - no chunk is longer than chunk_size * 1.5 characters
    """
    text = " ".join(f"sentence number {i}." for i in range(400))
    chunks = split_text(text, "recursive", chunk_size=300, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 300 * 1.5 for c in chunks)


def test_citation_parser_maps_markers_to_sources():
    """
    TODO: build three RetrievedChunk objects with metadata
    {"source": "a.md"/"b.md"/"c.md", "chunk_id": ...}. Call
    parse_citations("Per policy [1] and the circular [2, 3].", chunks) and
    assert the returned list has markers [1, 2, 3] mapped to sources
    ["a.md", "b.md", "c.md"]. Also assert parse_citations("No markers here.",
    chunks) returns [].
    """
    chunks = [
        RetrievedChunk(content="x", metadata={"source": "a.md", "chunk_id": "a.md::0"}),
        RetrievedChunk(content="y", metadata={"source": "b.md", "chunk_id": "b.md::0"}),
        RetrievedChunk(content="z", metadata={"source": "c.md", "chunk_id": "c.md::0"}),
    ]
    cites = parse_citations("Per policy [1] and the circular [2, 3].", chunks)
    assert [c["marker"] for c in cites] == [1, 2, 3]
    assert [c["source"] for c in cites] == ["a.md", "b.md", "c.md"]
    assert parse_citations("No markers here.", chunks) == []
