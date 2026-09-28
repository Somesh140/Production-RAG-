"""
Multi-source document loaders.

TODO (Activity 1.2): You implement the four per-format text extractors:
_load_pdf, _load_docx, _load_html and _load_markdown. The dispatcher
load_document() and the corpus walker load_corpus() are already provided
and complete.

The POC could only read a single .txt or .md file. A real enterprise
knowledge base is spread across PDFs (regulatory circulars, product
sheets), Word documents (policies, SOPs), HTML (intranet pages) and
Markdown. So the first thing a production pipeline needs is one loader per
format, each turning its files into plain text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.utils.config_loader import load_yaml_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RawDocument:
    """One source file, turned into plain text. Provided, complete.

    - content:   the extracted plain text
    - source:    path relative to the knowledge-base root (the key
                 Activity 1.3 looks up in the manifest), e.g.
                 "loans/home_loan_policy.pdf"
    - file_type: the lowercased extension without the dot, e.g. "pdf"
    """
    content: str
    source: str
    file_type: str


# --- Per-format extractors: IMPLEMENT THESE (Activity 1.2) ------------------

def _load_markdown(path: Path) -> str:
    """
    Load a Markdown file as plain text. Tier: 🟢 Beginner.

    Walk through together: this is the simplest of the four. It sets the
    read-then-return shape the others build on.

    Read `path` as UTF-8 text and return it stripped. Chunking later on is
    Markdown-aware, so keep the raw Markdown as it is. No HTML conversion
    is needed.
    """
    return path.read_text(encoding="utf-8").strip()


def _load_pdf(path: Path) -> str:
    """
    Extract text from a PDF file. Tier: 🟢 Beginner.

    Walk through together: loop over the pages, then join them. `_load_docx`
    (your turn, below) repeats this same pattern over paragraphs instead of
    pages.

    Use `from pypdf import PdfReader`. Open `path` and loop over
    `reader.pages`. Call `page.extract_text()` on each one. It can return
    None, so treat that as "". Join the pages with "\\n\\n" and return the
    text with leading and trailing whitespace stripped.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


def _load_docx(path: Path) -> str:
    """
    Extract text from a Word .docx file. Tier: 🟢 Beginner.

    Your turn: it's the same loop-then-join shape as `_load_pdf`, just over
    paragraphs instead of pages.

    Use `import docx` (python-docx). Open `docx.Document(str(path))` and
    collect `para.text` for every paragraph in `document.paragraphs`. Drop
    the empty or whitespace-only paragraphs, then join the rest with "\\n".
    Return the result stripped.
    """
    import docx

    document = docx.Document(str(path))
    paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras).strip()


def _load_html(path: Path) -> str:
    """
    Extract visible text from an HTML page. Tier: 🟢 Beginner.

    Your turn: this is the least mechanical of the four, because you're
    stripping tags, not just looping over a list.

    Use `from bs4 import BeautifulSoup`. Parse `path.read_text(
    encoding="utf-8")` with the "lxml" parser. Remove every <script> and
    <style> tag (`for t in soup(["script", "style"]): t.decompose()`). Call
    `soup.get_text(separator="\\n")`, then collapse three-or-more consecutive
    newlines down to two with `re.sub(r"\\n\\s*\\n\\s*\\n+", "\\n\\n", text)`
    (`re` is already imported above). Return the result stripped.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# --- Dispatcher + corpus walker: PROVIDED, COMPLETE -----------------------

_EXTENSION_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".html": _load_html,
    ".htm": _load_html,
    ".md": _load_markdown,
}


def load_document(path: str | Path, kb_root: str | Path = "data/knowledge_base") -> RawDocument:
    """Load a single file into a RawDocument, picking the extractor by
    file extension. Raises ValueError for an unsupported extension."""
    path = Path(path)
    kb_root = Path(kb_root)
    ext = path.suffix.lower()
    loader = _EXTENSION_LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext} ({path.name})")
    try:
        rel = str(path.relative_to(kb_root)).replace("\\", "/")
    except ValueError:
        rel = path.name
    return RawDocument(content=loader(path), source=rel, file_type=ext.lstrip("."))


def load_corpus(kb_root: str | Path = "data/knowledge_base",
                base_path: Optional[str | Path] = None) -> List[RawDocument]:
    """Walk the knowledge-base tree and load every supported file. The
    manifest and any unsupported files are skipped, with a logged warning.

    base_path=None (the default): kb_root resolves relative to the current
    working directory. Pass base_path to resolve kb_root (and the config
    lookup below) relative to an explicit directory instead. See
    src/utils/config_loader.py."""
    kb_root = resolve_path(kb_root, base_path)
    supported = set(load_yaml_config("config/ingestion_config.yaml",
                                     base_path=base_path)["ingestion"]["supported_extensions"])
    docs: List[RawDocument] = []
    for file in sorted(kb_root.rglob("*")):
        if not file.is_file() or file.name == "manifest.csv":
            continue
        if file.suffix.lower() not in supported:
            logger.warning(f"Skipping unsupported file: {file}")
            continue
        docs.append(load_document(file, kb_root))
    logger.info(f"Loaded {len(docs)} documents from {kb_root}")
    return docs
