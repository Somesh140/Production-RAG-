"""
Document processing and enterprise metadata management.

TODO (Activity 1.3): You implement build_metadata() and process_document().
load_manifest() and process_corpus() are already provided and complete.

Good retrieval in an enterprise knowledge base takes more than semantic
similarity. A compliance officer asking about card chargebacks should get
the *Cards* department's current procedure, not a superseded draft from
Operations. That's what metadata is for. Every chunk carries department,
doc_type, product, version and effective_date, so the retriever
(Activity 2.2) can filter on them.
"""
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
    """A RawDocument enriched with enterprise metadata. Provided, complete."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_key(path: str) -> str:
    """Normalize a source or manifest-key path for lookup: forward slashes
    and lowercase. This is the one place that defines how a source path is
    normalized. It's used when building the manifest keys here and again
    when looking them up in process_document, so the two sides can't drift
    apart. Display values (e.g. ProcessedDocument.metadata["source"]) should
    keep their original, readable casing, so don't pass them through this
    helper."""
    return path.strip().replace("\\", "/").lower()


def load_manifest(manifest_path: str | Path = "data/knowledge_base/manifest.csv",
                  base_path: Optional[str | Path] = None) -> Dict[str, Dict[str, str]]:
    """Read the corpus manifest CSV into a dict keyed by `relative_path`.
    Provided, complete.

    Each row supplies the enterprise metadata you can't work out from the
    file itself: department, doc_type, product, version, effective_date,
    title, owner.

    base_path=None (the default): manifest_path resolves relative to the
    current working directory. Pass base_path to resolve it relative to an
    explicit directory instead. See src/utils/config_loader.py.
    """
    manifest_path = resolve_path(manifest_path, base_path)
    rows: Dict[str, Dict[str, str]] = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = _normalize_key(row["relative_path"])
            rows[key] = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    logger.info(f"Loaded manifest with {len(rows)} entries")
    return rows


# --- IMPLEMENT THESE (Activity 1.3) --------------------------------------

def build_metadata(raw: RawDocument, manifest_row: Dict[str, str]) -> Dict[str, Any]:
    """
    Assemble the metadata dict for one document. Tier: 🟢 Beginner.

    Walk through together (steps 1-2):
    1. Read the required field names with exactly this call - no base_path
       argument here, unlike load_manifest() above: build_metadata() doesn't
       take one, so passing base_path=base_path would reference a name that
       doesn't exist in this function's scope:
         required = load_yaml_config("config/ingestion_config.yaml")["metadata"]["required_fields"]
       This resolves relative to the current working directory, which is
       already the project root whenever this runs (the workbook, scripts,
       and pytest all run from there).
    2. Build a dict with:
         - department, doc_type, product, version, effective_date  -> from manifest_row
         - source      -> raw.source
         - file_type   -> raw.file_type
         - title       -> manifest_row.get("title") or raw.source
       Also copy "owner" from manifest_row if it's there.

    Your turn (steps 3-4):
    3. Validate: every name in required_fields must be in the dict AND
       non-empty. If any is missing or empty, raise ValueError that lists
       the offending field name(s) and raw.source.
    4. Return the dict.
    """
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
    """
    Turn one RawDocument into a ProcessedDocument. Tier: 🟢 Beginner.

    Your turn: this is where governance actually gets enforced. It's the
    core idea of Activity 1.3.

    1. Look up `manifest[_normalize_key(raw.source)]`. load_manifest keys
       the manifest by _normalize_key(relative_path), so your lookup has to
       normalize the same way (forward slashes and lowercase). Otherwise a
       manifest row and a real file that differ only in case will wrongly
       look "ungoverned". If there's no entry for that key, raise KeyError
       naming raw.source. A document with no manifest row can't be governed,
       so it must not enter the index.
    2. Call build_metadata(raw, row) to get the metadata dict.
    3. Return ProcessedDocument(content=raw.content, metadata=<that dict>).
    """
    key = _normalize_key(raw.source)
    if key not in manifest:
        raise KeyError(f"No manifest entry for {raw.source} - document is ungoverned, excluded from the index")
    metadata = build_metadata(raw, manifest[key])
    return ProcessedDocument(content=raw.content, metadata=metadata)


# --- PROVIDED, COMPLETE -------------------------------------------------

def process_corpus(raws: List[RawDocument],
                   manifest: Dict[str, Dict[str, str]]) -> List[ProcessedDocument]:
    """Process every RawDocument. Any that raise are skipped with a warning,
    so one ungoverned file doesn't abort the whole ingestion run."""
    out: List[ProcessedDocument] = []
    for raw in raws:
        try:
            out.append(process_document(raw, manifest))
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping {raw.source}: {e}")
    logger.info(f"Processed {len(out)}/{len(raws)} documents")
    return out
