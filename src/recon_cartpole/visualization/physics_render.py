from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


def render_trace_html(trace: dict[str, Any], output_path: str, title: str = "ReCoN CartPole Replay") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    default_asset = Path("reports/assets/nikita_boar.jpg")
    default_image = os.path.relpath(default_asset, out.parent).replace(os.sep, "/")
    trace.setdefault("metadata", {}).setdefault("floating_images", [{"src": default_image, "scale": 0.08}])
    data = json.dumps(trace)
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE_HTML__</title>
<style>
:root { color-scheme: light; }
body { margin:0; font-family: system-ui, sans-serif; background:#f8fafc; color:#0f172a; }
main { display:grid; grid-template-columns: minmax(560px, 1fr) 420px; gap:16px; padding:16px; }
canvas, svg { width:100%; background:white; border:1px solid #cbd5e1; border-radius:6px; }
#graph { height:680px; }
.panel { background:white; border:1px solid #cbd5e1; border-radius:6px; padding:12px; }
.panel h2 { margin:0 0 10px; font-size:18px; }
.controls { display:grid; gap:9px; margin:10px 0 12px; }
.control-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
button, select, input::file-selector-button { border:1px solid #94a3b8; background:#f8fafc; color:#0f172a; border-radius:6px; padding:6px 9px; font:inherit; cursor:pointer; }
button:hover, input::file-selector-button:hover { background:#e2e8f0; }
button.icon { width:34px; height:32px; display:inline-grid; place-items:center; padding:0; }
input[type="range"] { flex:1; min-width:150px; }
input[type="file"] { max-width:100%; }
label { font-size:13px; color:#334155; }
#stats { font-size:13px; line-height:1.45; }
.legend { display:grid; grid-template-columns: 1fr 1fr; gap:5px 10px; margin:10px 0; font-size:12px; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; border:1px solid #64748b; }
pre { white-space:pre-wrap; font-size:12px; max-height:340px; overflow:auto; background:#f8fafc; padding:8px; border-radius:4px; }
@media (max-width: 980px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <section>
    <canvas id="scene" width="900" height="460"></canvas>
    <svg id="graph" viewBox="0 0 1200 760" preserveAspectRatio="xMidYMid meet"></svg>
  </section>
  <aside class="panel">
    <h2 id="title">__TITLE_HTML__</h2>
    <div class="controls">
      <input id="traceFile" type="file" accept="application/json,.json">
      <div class="control-row">
        <button id="playPause" type="button">Pause</button>
        <button id="prevStep" class="icon" type="button" title="Previous environment step">|<</button>
        <button id="prevTick" class="icon" type="button" title="Previous graph tick"><</button>
        <button id="nextTick" class="icon" type="button" title="Next graph tick">></button>
        <button id="nextStep" class="icon" type="button" title="Next environment step">>|</button>
      </div>
      <div class="control-row">
        <label for="playUnit">Playback</label>
        <select id="playUnit">
          <option value="tick">graph ticks</option>
          <option value="step">env steps</option>
        </select>
      </div>
      <div class="control-row">
        <label for="speed">Speed</label>
        <input id="speed" type="range" min="10" max="500" step="10" value="90">
        <span id="speedLabel">90 ms</span>
      </div>
    </div>
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
const embeddedTrace = __TRACE_JSON__;
const defaultFloatingImage = '__DEFAULT_FLOATING_IMAGE__';
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const graph = document.getElementById('graph');
const stateColor = {
  INACTIVE: '#e2e8f0', REQUESTED: '#38bdf8', ACTIVE: '#a78bfa', SUPPRESSED: '#94a3b8',
  WAITING: '#f59e0b', TRUE: '#84cc16', CONFIRMED: '#22c55e', FAILED: '#ef4444'
};
let currentTrace = embeddedTrace;
let steps = [];
let topology = {nodes: [], edges: []};
let layout = {pos: {}, byId: {}, edges: []};
let timeline = [];
let frame = 0;
let playing = true;
let timer = null;
let floatingSprites = [];

function buildTimeline(nextSteps) {
  return nextSteps.flatMap((step, stepIndex) => {
    const ticks = Array.isArray(step.graph_ticks) && step.graph_ticks.length
      ? step.graph_ticks
      : [{
          engine_tick: 0, phase: 'settled', nodes: step.graph_nodes || {},
          fired_edges: step.fired_edges || [], selected_regime: step.selected_regime,
          force: step.force, proposal: step.proposal, action_ready: true
        }];
    return ticks.map((graphTick, tickIndex) => ({step, stepIndex, graphTick, tickIndex}));
  });
}
function computeLayout() {
  const nodes = topology.nodes || [];
  const edges = topology.edges || [];
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const children = new Map();
  edges.filter(e => e.type === 'SUB').forEach(e => {
    if (!children.has(e.src)) children.set(e.src, []);
    children.get(e.src).push(e.dst);
  });
  const root = byId.root_balance ? 'root_balance' : (nodes[0] && nodes[0].id);
  const levels = [];
  const seen = new Set();
  function visit(id, depth) {
    if (!id || seen.has(id)) return;
    seen.add(id);
    if (!levels[depth]) levels[depth] = [];
    levels[depth].push(id);
    (children.get(id) || []).forEach(child => visit(child, depth + 1));
  }
  visit(root, 0);
  nodes.forEach(n => { if (!seen.has(n.id)) { if (!levels[levels.length]) levels[levels.length] = []; levels[levels.length - 1].push(n.id); } });
  const pos = {};
  const yStep = Math.max(70, 700 / Math.max(1, levels.length));
  levels.forEach((ids, depth) => {
    const xStep = 1120 / Math.max(1, ids.length + 1);
    ids.forEach((id, i) => { pos[id] = {x: 40 + xStep * (i + 1), y: 38 + depth * yStep}; });
  });
  return {pos, byId, edges};
}
function configureFloatingImages() {
  floatingSprites = [];
  const images = (currentTrace.metadata && currentTrace.metadata.floating_images) || [{src: defaultFloatingImage, scale: 0.08}];
  images.forEach((item, index) => {
    const img = new Image();
    img.onload = () => drawCurrent();
    img.src = item.src || defaultFloatingImage;
    floatingSprites.push({
      img,
      scale: Number(item.scale || 0.08),
      alpha: Number(item.alpha || 0.28),
      phase: index * 2.13 + 0.7,
      speed: Number(item.speed || 0.018)
    });
  });
}
function setTrace(trace, label) {
  currentTrace = trace || {metadata: {}, steps: []};
  steps = currentTrace.steps || [];
  topology = (currentTrace.metadata && currentTrace.metadata.graph) || {nodes: [], edges: []};
  layout = computeLayout();
  timeline = buildTimeline(steps);
  frame = 0;
  configureFloatingImages();
  document.getElementById('title').textContent = label || currentTrace.metadata?.stage || __TITLE_JS__;
  drawCurrent();
}
function drawFloatingBackground(frameNumber) {
  floatingSprites.forEach(sprite => {
    if (!sprite.img.complete || !sprite.img.naturalWidth) return;
    const w = canvas.width * Math.max(0.05, Math.min(0.10, sprite.scale));
    const h = w * sprite.img.naturalHeight / sprite.img.naturalWidth;
    const t = frameNumber * sprite.speed + sprite.phase;
    const x = canvas.width * (0.5 + 0.38 * Math.sin(t * 0.73));
    const y = canvas.height * (0.45 + 0.28 * Math.cos(t * 1.11));
    ctx.save();
    ctx.globalAlpha = sprite.alpha;
    ctx.translate(x, y);
    ctx.rotate(0.18 * Math.sin(t));
    ctx.drawImage(sprite.img, -w / 2, -h / 2, w, h);
    ctx.restore();
  });
}
function drawScene(s) {
  ctx.clearRect(0,0,900,460);
  ctx.fillStyle = '#f8fafc'; ctx.fillRect(0,0,900,460);
  drawFloatingBackground(frame);
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
  for (let i=0; i<n; i++) {
    const th = raw[2+i] || 0, len = 96 - Math.min(38, i * 6);
    const nx = x + Math.sin(th) * len, ny = y - Math.cos(th) * len;
    ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 8;
    ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(nx,ny); ctx.stroke();
    ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(nx,ny,8,0,Math.PI*2); ctx.fill();
    ctx.fillStyle = '#0891b2'; ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2); ctx.fill();
    x = nx; y = ny;
  }
  const force = Number(s.force || 0);
  ctx.strokeStyle = force >= 0 ? '#16a34a' : '#dc2626';
  ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(cartX, railY+44); ctx.lineTo(cartX + Math.sign(force) * Math.min(90, Math.abs(force) * 9), railY+44); ctx.stroke();
}
function esc(text) { return String(text).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function drawArrow(x1, y1, x2, y2, color, dashed, width) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx*dx + dy*dy) || 1;
  const ux = dx / len, uy = dy / len;
  const startX = x1 + ux * 31, startY = y1 + uy * 31;
  const endX = x2 - ux * 31, endY = y2 - uy * 31;
  const dash = dashed ? 'stroke-dasharray="7 7"' : '';
  return `<line x1="${startX}" y1="${startY}" x2="${endX}" y2="${endY}" stroke="${color}" stroke-width="${width}" marker-end="url(#arrow)" ${dash} />`;
}
function drawNode(id, state, selectedRegime) {
  const p = layout.pos[id];
  const node = layout.byId[id] || {type:'SCRIPT'};
  if (!p) return '';
  const color = stateColor[state] || '#e2e8f0';
  const selected = id === selectedRegime || id === `${selectedRegime}_proposal`;
  const stroke = selected ? '#7c3aed' : '#334155';
  const label = esc(id);
  if (node.type === 'TERMINAL') {
    return `<polygon points="${p.x},${p.y-25} ${p.x+31},${p.y} ${p.x},${p.y+25} ${p.x-31},${p.y}" fill="${color}" stroke="${stroke}" stroke-width="2" />`
      + `<text x="${p.x}" y="${p.y+45}" text-anchor="middle" font-size="12" fill="#0f172a">${label}</text>`;
  }
  return `<circle cx="${p.x}" cy="${p.y}" r="28" fill="${color}" stroke="${stroke}" stroke-width="2" />`
    + `<text x="${p.x}" y="${p.y+44}" text-anchor="middle" font-size="12" fill="#0f172a">${label}</text>`;
}
function drawGraph(s, graphTick) {
  const states = graphTick.nodes || s.graph_nodes || {};
  const selectedRegime = graphTick.selected_regime || s.selected_regime;
  const firedEdges = graphTick.fired_edges || [];
  let out = `<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto"><polygon points="0 0, 8 3.5, 0 7" fill="#475569" /></marker></defs>`;
  layout.edges.forEach(e => {
    const a = layout.pos[e.src], b = layout.pos[e.dst];
    if (!a || !b) return;
    const fired = firedEdges.some(fe => fe.src === e.src && fe.dst === e.dst && fe.ltype === e.type);
    if (e.type === 'SUB') out += drawArrow(a.x, a.y, b.x, b.y, fired ? '#7c3aed' : '#475569', false, fired ? 4 : 2);
    if (e.type === 'POR') out += drawArrow(a.x, a.y, b.x, b.y, fired ? '#7c3aed' : '#94a3b8', true, fired ? 3 : 1.5);
  });
  Object.keys(layout.pos).forEach(id => { out += drawNode(id, states[id] || 'INACTIVE', selectedRegime); });
  graph.innerHTML = out;
}
function drawCurrent() {
  if (!timeline.length) return;
  const item = timeline[((frame % timeline.length) + timeline.length) % timeline.length];
  const s = item.step;
  const graphTick = item.graphTick;
  const tickLabel = graphTick.engine_tick ?? item.tickIndex;
  drawScene(s); drawGraph(s, graphTick);
  document.getElementById('stats').innerHTML = `env step ${s.step} / ${steps.length - 1} | graph tick ${tickLabel} (${item.tickIndex + 1}) | frame ${frame + 1} / ${timeline.length}<br>return ${s.return_so_far} | regime ${graphTick.selected_regime || s.selected_regime || ''}`;
  document.getElementById('details').textContent = JSON.stringify({
    metadata: currentTrace.metadata, graph_tick: graphTick, goal: s.goal_vector,
    proposal: graphTick.proposal || s.proposal, proposals: s.proposals || [],
    suppressed_proposals: s.suppressed_proposals || [], bandit: s.bandit, plasticity: s.plasticity,
    fast_deltas: s.fast_deltas || {}, consolidation: s.consolidation || {}
  }, null, 2);
}
function advanceTick(delta) {
  if (!timeline.length) return;
  frame = (frame + delta + timeline.length) % timeline.length;
  drawCurrent();
}
function firstFrameForStep(stepIndex) {
  const idx = timeline.findIndex(item => item.stepIndex === stepIndex);
  return idx >= 0 ? idx : 0;
}
function advanceStep(delta) {
  if (!timeline.length) return;
  const current = timeline[((frame % timeline.length) + timeline.length) % timeline.length];
  const nextStep = (current.stepIndex + delta + steps.length) % steps.length;
  frame = firstFrameForStep(nextStep);
  drawCurrent();
}
function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  const delay = Number(document.getElementById('speed').value || 90);
  timer = setTimeout(() => {
    if (document.getElementById('playUnit').value === 'step') advanceStep(1);
    else advanceTick(1);
    schedule();
  }, delay);
}
document.getElementById('playPause').addEventListener('click', () => {
  playing = !playing;
  document.getElementById('playPause').textContent = playing ? 'Pause' : 'Play';
  schedule();
});
document.getElementById('prevTick').addEventListener('click', () => { playing = false; document.getElementById('playPause').textContent = 'Play'; advanceTick(-1); schedule(); });
document.getElementById('nextTick').addEventListener('click', () => { playing = false; document.getElementById('playPause').textContent = 'Play'; advanceTick(1); schedule(); });
document.getElementById('prevStep').addEventListener('click', () => { playing = false; document.getElementById('playPause').textContent = 'Play'; advanceStep(-1); schedule(); });
document.getElementById('nextStep').addEventListener('click', () => { playing = false; document.getElementById('playPause').textContent = 'Play'; advanceStep(1); schedule(); });
document.getElementById('speed').addEventListener('input', event => {
  document.getElementById('speedLabel').textContent = `${event.target.value} ms`;
  schedule();
});
document.getElementById('playUnit').addEventListener('change', schedule);
document.getElementById('traceFile').addEventListener('change', async event => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const loaded = JSON.parse(await file.text());
  setTrace(loaded, file.name);
});
setTrace(embeddedTrace, __TITLE_JS__);
schedule();
</script>
</body>
</html>"""
    document = (
        document.replace("__TITLE_HTML__", html.escape(title))
        .replace("__TITLE_JS__", json.dumps(title))
        .replace("__TRACE_JSON__", data)
        .replace("__DEFAULT_FLOATING_IMAGE__", default_image)
    )
    out.write_text(document, encoding="utf-8")
