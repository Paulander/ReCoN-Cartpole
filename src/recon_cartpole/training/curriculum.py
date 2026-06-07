from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .train import train_block


def run_curriculum(path: str) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    results = []
    for stage in config.get("stages", []):
        result = train_block(stage)
        result["stage"] = stage.get("name", "unnamed")
        results.append(result)
    return results


def save_curriculum_results(results: list[dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

