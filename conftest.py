"""
Makes `pytest reference_solution/tests/test_pipeline.py` test the reference
solution, not the learner's unfinished top-level src/, without ever leaking
that override into the learner's own tests/test_pipeline.py when both are
collected in one run (e.g. a bare `pytest` from the project root).

reference_solution/tests/test_pipeline.py imports plain `src.*` names (by
design - the same file is also copied by scripts/verify_reference.py into a
merged temp project where `src.*` really is the reference solution). Run
directly from this repo, those bare imports would otherwise resolve to the
real top-level src/, which still has NotImplementedError stubs, and fail in
a way that looks like the reference solution is broken rather than that the
wrong module was tested.

reference_solution/src/ only contains the seven TODO modules (shared
infrastructure like src/utils, src/llm, src/pipeline.py is intentionally not
duplicated there), so this can't be fixed with a plain sys.path insert -
`from src.utils...` inside a reference module still needs to reach the real
top-level src/utils. Instead, each reference module is registered under its
`src.*` name in sys.modules, in dependency order, right before it is about
to be imported.

pytest loads every conftest.py reachable from the session's target paths
upfront, before collection of any test file begins - so "install once, then
restore on the first file collected outside this directory" is not safe:
whichever file pytest happens to collect first (which may be the learner's
own tests/test_pipeline.py, regardless of argument order) would consume that
one-shot restore before the reference file ever gets a turn. The fix here is
per-file instead of order-dependent: pytest_collectstart fires for a Module
collector right before its own import runs, so the override is installed or
removed right there, for that one file, every time - never a global
"once" state to race against collection order.
"""
import importlib.util
import sys
from pathlib import Path

_REFERENCE_TESTS_DIR = (Path(__file__).resolve().parent / "reference_solution" / "tests")
_REFERENCE_SRC = _REFERENCE_TESTS_DIR.parent / "src"
_REFERENCE_MODULES = [
    "ingestion/loaders.py",
    "ingestion/document_processor.py",
    "chunking/chunker.py",
    "indexing/vector_store.py",
    "retrieval/retriever.py",
    "generation/answer_generator.py",
    "observability/tracing.py",
]
_MODULE_NAMES = ["src." + rel[:-3].replace("/", ".") for rel in _REFERENCE_MODULES]

_installed = False
_saved = {}


def _install():
    global _installed
    if _installed:
        return
    for rel, name in zip(_REFERENCE_MODULES, _MODULE_NAMES):
        _saved[name] = sys.modules.get(name)
        spec = importlib.util.spec_from_file_location(name, _REFERENCE_SRC / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    _installed = True


def _uninstall():
    global _installed
    if not _installed:
        return
    for name in _MODULE_NAMES:
        prior = _saved.pop(name, None)
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
    _installed = False


def pytest_collectstart(collector):
    path = getattr(collector, "path", None) or getattr(collector, "fspath", None)
    if path is None:
        return
    path = Path(str(path))
    if not path.is_file():
        return
    if _REFERENCE_TESTS_DIR in path.parents:
        _install()
    else:
        _uninstall()
