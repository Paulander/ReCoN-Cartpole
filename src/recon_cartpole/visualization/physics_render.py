from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_trace_html(trace: dict[str, Any], output_path: str, title: str = "ReCoN CartPole Replay") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(trace)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; font-family: system-ui, sans-serif; background:#f8fafc; color:#0f172a; }}
main {{ display:grid; grid-template-columns: minmax(560px, 1fr) 400px; gap:16px; padding:16px; }}
canvas, svg {{ width:100%; background:white; border:1px solid #cbd5e1; border-radius:6px; }}
#graph {{ height:680px; }}
.panel {{ background:white; border:1px solid #cbd5e1; border-radius:6px; padding:12px; }}
.panel h2 {{ margin:0 0 8px; font-size:18px; }}
.legend {{ display:grid; grid-template-columns: 1fr 1fr; gap:5px 10px; margin:10px 0; font-size:12px; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; border:1px solid #64748b; }}
pre {{ white-space:pre-wrap; font-size:12px; max-height:340px; overflow:auto; background:#f8fafc; padding:8px; border-radius:4px; }}
</style>
</head>
<body>
<main>
  <section>
    <canvas id="scene" width="900" height="460"></canvas>
    <svg id="graph" viewBox="0 0 1200 760" preserveAspectRatio="xMidYMid meet"></svg>
  </section>
  <aside class="panel">
    <h2>{html.escape(title)}</h2>
    <div id="stats"></div>
    <div class="legend">
      <span><i class="swatch" style="background:#e2e8f0"></i>inactive</span>
      <span><i class="swatch" style="background:#38bdf8"></i>requested</span>
      <span><i class="swatch" style="background:#f59e0b"></i>waiting</span>
      <span><i class="swatch" style="background:#22c55e"></i>confirmed</span>
      <span><i class="swatch" style="background:#ef4444"></i>failed</span>
      <span><i class="swatch" style="background:#a78bfa"></i>active/true</span>
    </div>
    <pre id="details"></pre>
  </aside>
</main>
<script>
const trace = {data};
const steps = trace.steps || [];
const topology = (trace.metadata && trace.metadata.graph) || {{nodes: [], edges: []}};
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const graph = document.getElementById('graph');
let frame = 0;
const stateColor = {{
  INACTIVE: '#e2e8f0', REQUESTED: '#38bdf8', ACTIVE: '#a78bfa', SUPPRESSED: '#94a3b8',
  WAITING: '#f59e0b', TRUE: '#84cc16', CONFIRMED: '#22c55e', FAILED: '#ef4444'
}};
function drawScene(s) {{
  ctx.clearRect(0,0,900,460);
  ctx.fillStyle = '#f8fafc'; ctx.fillRect(0,0,900,460);
  const raw = s.raw_state || [0,0,0,0];
  const n = Math.max(1, Math.floor((raw.length - 2) / 2));
  const railY = 360, centerX = 450, scale = 120;
  const cartX = centerX + raw[0] * scale;
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(70, railY); ctx.lineTo(830, railY); ctx.stroke();
  ctx.strokeStyle = '#fca5a5'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(centerX - 2.4 * scale, railY - 52); ctx.lineTo(centerX - 2.4 * scale, railY + 24); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(centerX + 2.4 * scale, railY - 52); ctx.lineTo(centerX + 2.4 * scale, railY + 24); ctx.stroke();
  ctx.fillStyle = '#2563eb'; ctx.fillRect(cartX - 44, railY - 28, 88, 44);
  let x = cartX, y = railY - 28;
  for (let i=0; i<n; i++) {{
    const th = raw[2+i] || 0, len = 96 - Math.min(38, i * 6);
    const nx = x + Math.sin(th) * len, ny = y - Math.cos(th) * len;
    ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 8;
    ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(nx,ny); ctx.stroke();
    ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(nx,ny,8,0,Math.PI*2); ctx.fill();
    ctx.fillStyle = '#0891b2'; ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2); ctx.fill();
    x = nx; y = ny;
  }}
  const force = Number(s.force || 0);
  ctx.strokeStyle = force >= 0 ? '#16a34a' : '#dc2626';
  ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(cartX, railY+44); ctx.lineTo(cartX + Math.sign(force) * Math.min(90, Math.abs(force) * 9), railY+44); ctx.stroke();
}}
function computeLayout() {{
  const nodes = topology.nodes || [];
  const edges = topology.edges || [];
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const children = new Map();
  edges.filter(e => e.type === 'SUB').forEach(e => {{
    if (!children.has(e.src)) children.set(e.src, []);
    children.get(e.src).push(e.dst);
  }});
  const root = byId.root_balance ? 'root_balance' : (nodes[0] && nodes[0].id);
  const levels = [];
  const seen = new Set();
  function visit(id, depth) {{
    if (!id || seen.has(id)) return;
    seen.add(id);
    if (!levels[depth]) levels[depth] = [];
    levels[depth].push(id);
    (children.get(id) || []).forEach(child => visit(child, depth + 1));
  }}
  visit(root, 0);
  nodes.forEach(n => {{ if (!seen.has(n.id)) {{ if (!levels[levels.length]) levels[levels.length] = []; levels[levels.length - 1].push(n.id); }} }});
  const pos = {{}};
  const yStep = Math.max(70, 700 / Math.max(1, levels.length));
  levels.forEach((ids, depth) => {{
    const xStep = 1120 / Math.max(1, ids.length + 1);
    ids.forEach((id, i) => {{ pos[id] = {{x: 40 + xStep * (i + 1), y: 38 + depth * yStep}}; }});
  }});
  return {{pos, byId, edges}};
}}
const layout = computeLayout();
function esc(text) {{ return String(text).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
function drawArrow(x1, y1, x2, y2, color, dashed, width) {{
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx*dx + dy*dy) || 1;
  const ux = dx / len, uy = dy / len;
  const startX = x1 + ux * 31, startY = y1 + uy * 31;
  const endX = x2 - ux * 31, endY = y2 - uy * 31;
  const dash = dashed ? 'stroke-dasharray="7 7"' : '';
  return `<line x1="${{startX}}" y1="${{startY}}" x2="${{endX}}" y2="${{endY}}" stroke="${{color}}" stroke-width="${{width}}" marker-end="url(#arrow)" ${{dash}} />`;
}}
function drawNode(id, state) {{
  const p = layout.pos[id];
  const node = layout.byId[id] || {{type:'SCRIPT'}};
  if (!p) return '';
  const color = stateColor[state] || '#e2e8f0';
  const stroke = id === (steps[frame % Math.max(1, steps.length)] || {{}}).selected_regime ? '#7c3aed' : '#334155';
  const label = esc(id);
  if (node.type === 'TERMINAL') {{
    return `<polygon points="${{p.x}},${{p.y-25}} ${{p.x+31}},${{p.y}} ${{p.x}},${{p.y+25}} ${{p.x-31}},${{p.y}}" fill="${{color}}" stroke="${{stroke}}" stroke-width="2" />`
      + `<text x="${{p.x}}" y="${{p.y+45}}" text-anchor="middle" font-size="12" fill="#0f172a">${{label}}</text>`;
  }}
  return `<circle cx="${{p.x}}" cy="${{p.y}}" r="28" fill="${{color}}" stroke="${{stroke}}" stroke-width="2" />`
    + `<text x="${{p.x}}" y="${{p.y+44}}" text-anchor="middle" font-size="12" fill="#0f172a">${{label}}</text>`;
}}
function drawGraph(s) {{
  const states = s.graph_nodes || {{}};
  let out = `<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto"><polygon points="0 0, 8 3.5, 0 7" fill="#475569" /></marker></defs>`;
  layout.edges.forEach(e => {{
    const a = layout.pos[e.src], b = layout.pos[e.dst];
    if (!a || !b) return;
    const fired = (s.fired_edges || []).some(fe => fe.src === e.src && fe.dst === e.dst && fe.ltype === e.type);
    if (e.type === 'SUB') out += drawArrow(a.x, a.y, b.x, b.y, fired ? '#7c3aed' : '#475569', false, fired ? 4 : 2);
    if (e.type === 'POR') out += drawArrow(a.x, a.y, b.x, b.y, '#94a3b8', true, 1.5);
  }});
  Object.keys(layout.pos).forEach(id => {{ out += drawNode(id, states[id] || 'INACTIVE'); }});
  graph.innerHTML = out;
}}
function tick() {{
  if (!steps.length) return;
  const s = steps[frame % steps.length];
  drawScene(s); drawGraph(s);
  document.getElementById('stats').innerHTML = `step ${{s.step}} | return ${{s.return_so_far}} | regime ${{s.selected_regime}}`;
  document.getElementById('details').textContent = JSON.stringify({{goal:s.goal_vector, proposal:s.proposal, bandit:s.bandit, plasticity:s.plasticity}}, null, 2);
  frame += 1;
  setTimeout(tick, 70);
}}
tick();
</script>
</body>
</html>"""
    out.write_text(document, encoding="utf-8")
