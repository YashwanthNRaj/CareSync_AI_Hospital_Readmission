from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON safely."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project_relative(path: Path) -> str:
    """Return a readable path string."""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)
