from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .train import train_block


@dataclass
class GroundIterationResult:
    stage: str
    report_dir: str
    metrics: dict[str, Any]
    promoted: bool


def run_ground_iteration(stage: dict[str, Any], report_root: str = "reports") -> GroundIterationResult:
    metrics = train_block(stage)
    stage_name = stage.get("name", "stage")
    report_dir = Path(report_root) / f"{stage_name}_iter_0"
    report_dir.mkdir(parents=True, exist_ok=True)
    promoted = metrics.get("mean_survival_steps", 0) >= stage.get("pass", {}).get("mean_survival_steps", float("inf"))
    return GroundIterationResult(stage_name, str(report_dir), metrics, promoted)

