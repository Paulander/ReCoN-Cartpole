from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(path: str, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_checkpoint(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

