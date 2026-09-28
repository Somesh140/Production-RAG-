"""Document processing and enterprise metadata - REFERENCE SOLUTION (Activity 1.3)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.loaders import RawDocument
from src.utils.config_loader import load_yaml_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProcessedDocument:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_key(path: str) -> str:
    """Normalize a source/manifest-key path for lookup comparison: forward
    slashes and lowercase. The single definition of "how a source path is
    normalized" - used both when building manifest keys and when looking
    them up, so the two sides can never drift apart. Display values (e.g.
    ProcessedDocument.metadata["source"]) should keep their original,
    human-readable casing and NOT go through this helper."""
    return path.strip().replace("\\", "/").lower()


def load_manifest(manifest_path: str | Path = "data/knowledge_base/manifest.csv",
                  base_path: Optional[str | Path] = None) -> Dict[str, Dict[str, str]]:
    manifest_path = resolve_path(manifest_path, base_path)
    rows: Dict[str, Dict[str, str]] = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = _normalize_key(row["relative_path"])
            rows[key] = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    logger.info(f"Loaded manifest with {len(rows)} entries")
    return rows


def build_metadata(raw: RawDocument, manifest_row: Dict[str, str]) -> Dict[str, Any]:
    required = load_yaml_config("config/ingestion_config.yaml")["metadata"]["required_fields"]

    metadata: Dict[str, Any] = {
        "department": manifest_row.get("department", ""),
        "doc_type": manifest_row.get("doc_type", ""),
        "product": manifest_row.get("product", ""),
        "version": manifest_row.get("version", ""),
        "effective_date": manifest_row.get("effective_date", ""),
        "source": raw.source,
        "file_type": raw.file_type,
        "title": manifest_row.get("title") or raw.source,
    }
    if manifest_row.get("owner"):
        metadata["owner"] = manifest_row["owner"]

    missing = [name for name in required if not str(metadata.get(name, "")).strip()]
    if missing:
        raise ValueError(f"{raw.source}: missing required metadata field(s): {', '.join(missing)}")
    return metadata


def process_document(raw: RawDocument, manifest: Dict[str, Dict[str, str]]) -> ProcessedDocument:
    key = _normalize_key(raw.source)
    if key not in manifest:
        raise KeyError(f"No manifest entry for {raw.source} - document is ungoverned, excluded from the index")
    metadata = build_metadata(raw, manifest[key])
    return ProcessedDocument(content=raw.content, metadata=metadata)


def process_corpus(raws: List[RawDocument],
                   manifest: Dict[str, Dict[str, str]]) -> List[ProcessedDocument]:
    out: List[ProcessedDocument] = []
    for raw in raws:
        try:
            out.append(process_document(raw, manifest))
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping {raw.source}: {e}")
    logger.info(f"Processed {len(out)}/{len(raws)} documents")
    return out
