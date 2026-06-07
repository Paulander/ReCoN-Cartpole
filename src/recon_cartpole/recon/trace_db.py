from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recon_lite import Graph


def graph_to_trace(graph: Graph) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.nid,
                "type": node.ntype.name,
                "meta": dict(node.meta),
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "src": edge.src,
                "dst": edge.dst,
                "type": edge.ltype.name,
                "weight": _weight(edge.w),
                "meta": dict(edge.meta),
            }
            for edge in graph.edges
        ],
    }


def save_trace(path: str, metadata: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": metadata, "steps": steps}, indent=2), encoding="utf-8")


def load_trace(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _weight(value: Any) -> float:
    try:
        return float(value[0]) if hasattr(value, "__len__") else float(value)
    except Exception:
        return 1.0
