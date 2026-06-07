from pathlib import Path

from recon_cartpole.visualization.physics_render import render_trace_html


def test_render_trace_html(tmp_path: Path):
    trace = {
        "metadata": {"env": "test"},
        "steps": [
            {
                "step": 0,
                "raw_state": [0, 0, 0.02, 0.0],
                "force": 10.0,
                "return_so_far": 1,
                "selected_regime": "recover_worst_pole",
                "goal_vector": {},
                "proposal": {},
                "graph_nodes": {"root_balance": "CONFIRMED"},
            }
        ],
    }
    out = tmp_path / "trace.html"
    render_trace_html(trace, str(out))
    assert out.exists()
    assert "ReCoN CartPole Replay" in out.read_text(encoding="utf-8")

