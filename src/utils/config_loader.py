"""Configuration loader for YAML files. Provided, complete - not part of any activity."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def resolve_path(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a relative path against an explicit base_path instead of the
    process's current working directory.

    Every caller in src/ passes relative paths (e.g.
    "config/ingestion_config.yaml", "data/knowledge_base"), which normally
    resolve against the working directory. That's a problem for a caller
    that can't, or shouldn't, change the working directory for the whole
    process (for example, app/demo_backend.py's demo-mode merge overlay,
    which needs every lookup to resolve inside a temp merge directory
    without moving the rest of the process there). With base_path, a caller
    can point lookups at an explicit directory instead.

    The default (base_path=None) behaves exactly as before. An absolute path
    comes back as it is, and a relative path comes back as it is too, so it
    still resolves against the working directory. Only when base_path is
    given and `path` is relative does it get joined onto base_path.
    """
    p = Path(path)
    if base_path is not None and not p.is_absolute():
        return Path(base_path) / p
    return p


def load_yaml_config(config_path: str,
                     base_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    With base_path=None (the default), config_path resolves relative to the
    process's current working directory. Pass base_path to resolve it
    relative to that directory instead, so you don't need to os.chdir() the
    process. See resolve_path().
    """
    path = resolve_path(config_path, base_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
