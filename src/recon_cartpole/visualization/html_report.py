from __future__ import annotations

from pathlib import Path
from typing import Any

from .dashboard import render_dashboard


def write_html_report(metrics: dict[str, Any], output_dir: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dashboard.html"
    render_dashboard(metrics, str(path))
    return str(path)

