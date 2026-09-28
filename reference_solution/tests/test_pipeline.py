"""Pipeline smoke suite - REFERENCE SOLUTION (Activity 3.2)."""
from __future__ import annotations

from src.chunking.chunker import split_text
from src.generation.answer_generator import parse_citations
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.document_processor import load_manifest, process_corpus
from src.ingestion.loaders import RawDocument, _EXTENSION_LOADERS, load_corpus
from src.utils.config_loader import load_yaml_config

CFG = load_yaml_config("config/ingestion_config.yaml")


def test_loaders_support_all_configured_formats():
    for ext in CFG["ingestion"]["supported_extensions"]:
        assert ext in _EXTENSION_LOADERS, f"No loader registered for {ext}"


def test_ungoverned_document_is_excluded():
    manifest = load_manifest(CFG["ingestion"]["manifest_path"])
    ungoverned = RawDocument(content="rogue draft", source="nowhere/rogue.md", file_type="md")
    out = process_corpus([ungoverned], manifest)
    assert out == []


def test_all_governed_documents_have_required_metadata():
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
    text = " ".join(f"sentence number {i}." for i in range(400))
    chunks = split_text(text, "recursive", chunk_size=300, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 300 * 1.5 for c in chunks)


def test_citation_parser_maps_markers_to_sources():
    chunks = [
        RetrievedChunk(content="x", metadata={"source": "a.md", "chunk_id": "a.md::0"}),
        RetrievedChunk(content="y", metadata={"source": "b.md", "chunk_id": "b.md::0"}),
        RetrievedChunk(content="z", metadata={"source": "c.md", "chunk_id": "c.md::0"}),
    ]
    cites = parse_citations("Per policy [1] and the circular [2, 3].", chunks)
    assert [c["marker"] for c in cites] == [1, 2, 3]
    assert [c["source"] for c in cites] == ["a.md", "b.md", "c.md"]
    assert parse_citations("No markers here.", chunks) == []
