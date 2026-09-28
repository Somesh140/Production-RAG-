"""
Demo-mode backend selection for knowledge_portal.py. Additive plumbing only -
no src/ or reference_solution/ module logic is modified here.

knowledge_portal.py normally imports `KnowledgeSystem` from the learner's own
`src/`, by design, so the portal's citation/refusal UI can't actually be
exercised live until a learner has finished every activity.

KP_DEMO_MODE is a narrow, explicit, opt-in escape hatch for previewing that
UI: when set, the portal sources KnowledgeSystem and the six activity
modules from reference_solution/ instead. It is OFF by default under every
normal `streamlit run app/knowledge_portal.py` invocation. The name is
deliberately specific (not DEBUG/MODE) so a learner can't set it by accident
and mistake the reference solution's output for their own working pipeline.

Invocation (either syntax works, same effect):
    KP_DEMO_MODE=1 streamlit run app/knowledge_portal.py          (bash)
    $env:KP_DEMO_MODE=1; streamlit run app/knowledge_portal.py    (PowerShell)

Mechanism: the same merged-overlay approach already used by
workbook_app.py's "See reference solution run" button and by
scripts/verify_reference.py. A full copy of the real src/, config/, and
data/ goes into a fresh temp directory, with only reference_solution/'s own
files copied on top of src/. This never mutates the real src/ or
reference_solution/ trees on disk. The merge directory is put at the front
of sys.path so the first `from src...` import in this process resolves the
merged copy under the package name "src".

Caller contract (unlike workbook_app.py's subprocess-per-call mechanism):
Streamlit reruns this whole script top-to-bottom on every widget interaction
within the same long-lived process, so `prepare_demo_backend()` must only be
called ONCE per session, on first build. The caller (knowledge_portal.py)
stashes the returned merge_root in `st.session_state` and calls
`reactivate_demo_backend()` on every later rerun instead. Calling
`prepare_demo_backend()` unguarded on every rerun would build a brand-new
merge dir per interaction and grow sys.path by one entry each time.

This module never changes the process's working directory. Every
cwd-relative lookup elsewhere in src/ (config, knowledge-base, vector-store
paths) accepts an optional `base_path` argument instead, and demo mode passes
`base_path=merge_root` explicitly wherever it calls into src/ - see
src/utils/config_loader.py's `resolve_path()`. This keeps the merged copy
fully self-contained without touching global process state.

Chroma's native Rust binding is incompatible with Python 3.14 on some
machines (see README.md's Vector Store note). Since demo mode always runs
the same reference_solution/ code regardless of what a learner has
implemented, `prepare_demo_backend()` sets `VECTOR_STORE=faiss` for this
process before returning, so demo mode never depends on Chroma actually
working. This only affects the demo-mode process; a learner's own
`streamlit run` respects whatever VECTOR_STORE is set in their own `.env`.

`prepare_demo_backend()` returns merge_root; the caller passes
`base_path=merge_root` into `KnowledgeSystem(...)` (see knowledge_portal.py's
`get_system()`) so every lookup the pipeline makes resolves inside the
merged copy.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# The 6 packages this course's activities implement; reference_solution/src
# is a partial mirror containing only these (no config/, data/, src/utils/,
# or src/llm/ of its own - see this repo's agent briefing for why).
REFERENCE_MODULES = [
    "ingestion/loaders.py",
    "ingestion/document_processor.py",
    "chunking/chunker.py",
    "indexing/vector_store.py",
    "retrieval/retriever.py",
    "generation/answer_generator.py",
    "observability/tracing.py",
]

DEMO_MODE_ENV_VAR = "KP_DEMO_MODE"


def is_demo_mode() -> bool:
    """True only if KP_DEMO_MODE=1 is explicitly set. Off by default."""
    return os.getenv(DEMO_MODE_ENV_VAR, "0") == "1"


def prepare_demo_backend() -> Path:
    """Build a fresh merged overlay directory (real src/+config/+data/ with
    reference_solution/'s own files copied on top of src/) and put it at the
    front of sys.path, so every `from src...` import in this process resolves
    through the merged directory rather than caching the learner's real src/
    package under the name "src".

    Does NOT chdir the process (see module docstring for why - it broke
    Streamlit's script resolution, then chromadb's DLL loading, across two
    separate rounds). The returned merge_root must instead be passed
    explicitly as `base_path=merge_root` into `KnowledgeSystem(...)`, so
    every cwd-relative config/knowledge-base/vector-store lookup resolves
    inside the merged copy without ever touching process cwd.

    Call this exactly once per session (see module docstring). The caller
    must stash the returned path and use `reactivate_demo_backend()` on
    every later rerun instead of calling this again.
    """
    merge_root = Path(tempfile.mkdtemp(prefix="kp_demo_merge_"))
    merge_src = merge_root / "src"
    shutil.copytree(PROJECT_ROOT / "src", merge_src)
    for name in ("config", "data"):
        shutil.copytree(PROJECT_ROOT / name, merge_root / name)
    for rel in REFERENCE_MODULES:
        shutil.copy2(PROJECT_ROOT / "reference_solution" / "src" / rel, merge_src / rel)
    sys.path.insert(0, str(merge_root))
    os.environ["VECTOR_STORE"] = "faiss"
    return merge_root


def reactivate_demo_backend(merge_root: Path | str) -> None:
    """Cheap, idempotent re-assertion for every rerun after the first in a
    demo-mode session: no mkdtemp, no copytree/copy2, so the temp-dir count
    and the work done per rerun both stay flat for the life of the session.

    Only re-asserts merge_root is at the front of sys.path, if it isn't
    already there (guards against unbounded sys.path growth across reruns).
    Does not chdir - there is nothing process-wide left to re-assert; every
    caller that needs the merge directory receives it explicitly as
    `base_path`.
    """
    merge_root = Path(merge_root)
    if str(merge_root) not in sys.path:
        sys.path.insert(0, str(merge_root))
