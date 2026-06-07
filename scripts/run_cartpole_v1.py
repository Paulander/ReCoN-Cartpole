from __future__ import annotations

import argparse

from recon_cartpole.envs.gym_cartpole_adapter import GymCartPoleAdapter
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.trace_db import graph_to_trace, save_trace
from recon_cartpole.training.evaluate import rollout
from recon_cartpole.visualization.physics_render import render_trace_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--mode", default="static_recon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-html")
    args = parser.parse_args()
    env = GymCartPoleAdapter(render_mode="rgb_array")
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode=args.mode))
    last = None
    for idx in range(args.episodes):
        last = rollout(env, controller, seed=args.seed + idx, trace=bool(args.render_html))
        print(f"episode={idx} return={last['return']} steps={last['steps']}")
    if args.render_html and last:
        trace = {"metadata": {"env": "CartPole-v1", "mode": args.mode, "graph": graph_to_trace(controller.graph)}, "steps": last["trace"]}
        save_trace(args.render_html.replace(".html", ".json"), trace["metadata"], trace["steps"])
        render_trace_html(trace, args.render_html, "Gymnasium CartPole-v1")


if __name__ == "__main__":
    main()

