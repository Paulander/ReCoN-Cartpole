from __future__ import annotations

from pathlib import Path

from recon_lite import Graph


def render_graph_html(graph: Graph, output_path: str) -> None:
    rows = []
    for edge in graph.edges:
        rows.append(f"<tr><td>{edge.src}</td><td>{edge.ltype.name}</td><td>{edge.dst}</td><td>{edge.w}</td></tr>")
    html = """<!doctype html><meta charset="utf-8"><title>ReCoN Graph</title>
<style>body{font-family:system-ui,sans-serif}td,th{padding:6px 10px;border-bottom:1px solid #ddd}</style>
<h1>ReCoN Graph</h1><table><tr><th>src</th><th>type</th><th>dst</th><th>w</th></tr>""" + "".join(rows) + "</table>"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

