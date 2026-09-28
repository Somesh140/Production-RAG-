"""
LangSmith observability.

TODO (Activity 3.1): You implement run_query_with_tracing().
configure_langsmith() and the local-trace fallback are provided.

The engineers running a production knowledge system need to see what a
query actually did: which chunks were retrieved, what the prompt was, how
long generation took. They shouldn't have to grep logs for it. LangSmith
gives you that once you wrap one entry point in tracing.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# --- PROVIDED, COMPLETE --------------------------------------------

def configure_langsmith() -> bool:
    """Return True if LangSmith tracing is switched on and a key is present.
    Reads LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY from the environment
    (the caller loads them from .env)."""
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    has_key = bool(os.getenv("LANGCHAIN_API_KEY")) and \
        os.getenv("LANGCHAIN_API_KEY") != "your_langsmith_api_key_here"
    if enabled and not has_key:
        logger.warning("LANGCHAIN_TRACING_V2=true but no LANGCHAIN_API_KEY set - running untraced.")
        return False
    if enabled:
        os.environ.setdefault("LANGCHAIN_PROJECT", "enterprise-knowledge-system")
    return enabled and has_key


def _run_with_local_trace(run_fn: Callable, *args, **kwargs) -> Any:
    """Fallback for when LangSmith isn't configured. Run the function and
    print a short step trace to stdout, so this activity always gives you
    something visible."""
    label = getattr(run_fn, "__name__", "query")
    print(f"--- local trace: {label} ---")
    t0 = time.perf_counter()
    result = run_fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000

    if isinstance(result, dict):
        hits = result.get("retrieved_chunks", [])
        print(f"  retrieved {len(hits)} chunk(s):")
        for h in hits:
            print(f"    - {h.get('source')} (score={h.get('score'):.3f})")
        print(f"  refused: {result.get('refused')}")
        print(f"  citations: {[c.get('source') for c in result.get('citations', [])]}")
    print(f"  elapsed: {elapsed:.0f} ms")
    print("--- end trace ---")
    return result


# --- IMPLEMENT THIS (Activity 3.1) ------------------------------

def run_query_with_tracing(run_fn: Callable, *args, **kwargs) -> Any:
    """
    Run `run_fn(*args, **kwargs)` with observability. Tier: 🟠 Advanced.

    Walk through together (step 1):
    1. If configure_langsmith() is False, return
       _run_with_local_trace(run_fn, *args, **kwargs). Turning tracing off
       must never break the query.

    Your turn (step 2):
    2. Otherwise wrap run_fn with `langsmith.traceable(run_type="chain",
       name="enterprise_kb_query")` and call the wrapped function with the
       same args. Return the wrapped call's result, not the original's.
    """
    if not configure_langsmith():
        return _run_with_local_trace(run_fn, *args, **kwargs)

    import langsmith

    @langsmith.traceable(run_type="chain", name="enterprise_kb_query")
    def _traced_query(*a, **kw):
        try:
            return run_fn(*a, **kw)
        except Exception as exc:
            # A failed query still produces a useful, searchable
            # trace instead of an opaque crash - record which stage failed
            # and why on the current span before letting the exception
            # propagate. The caller (e.g. knowledge_portal.py) still sees
            # the real exception; this only enriches what LangSmith shows.
            run = langsmith.get_current_run_tree()
            if run is not None:
                run.add_tags(["error"])
                run.add_metadata({
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "failed_stage": getattr(run_fn, "__name__", "run_fn"),
                })
            logger.error(f"Traced query failed in {getattr(run_fn, '__name__', 'run_fn')}: "
                        f"{type(exc).__name__}: {exc}")
            raise

    return _traced_query(*args, **kwargs)
