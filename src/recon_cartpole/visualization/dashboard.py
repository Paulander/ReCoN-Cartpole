from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_dashboard(metrics: dict[str, Any], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!doctype html><meta charset='utf-8'><title>ReCoN Dashboard</title>"
        "<style>body{font-family:system-ui,sans-serif;padding:20px}pre{background:#f1f5f9;padding:12px}</style>"
        "<h1>ReCoN CartPole Dashboard</h1><pre>"
        + json.dumps(metrics, indent=2)
        + "</pre>",
        encoding="utf-8",
    )

