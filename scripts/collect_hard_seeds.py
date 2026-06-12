from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout
from recon_cartpole.control.policy_observation import POLICY_OBSERVATION_MODES


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=args.n_poles,
            horizon=args.horizon,
            dt=args.dt,
            dynamics_mode=args.dynamics_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
        )
    )


def classify(raw: list[float], args: argparse.Namespace) -> str:
    if not raw:
        return "unknown"
    x = float(raw[0])
    if abs(x) > 2.2:
        return "rail_left" if x < 0 else "rail_right"
    angles = raw[2 : 2 + args.n_poles]
    if angles:
        return f"pole_{int(np.argmax(np.abs(angles)))}_angle"
    return "unknown"


def make_controller(args: argparse.Namespace) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_policy_terminal",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.model_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_frame_stack=args.frame_stack,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_observation_mode,
        )
    )


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    controller = make_controller(args)
    rows: list[dict[str, Any]] = []
    hard: list[int] = []
    for index in range(args.episodes):
        seed = args.seed_start + index
        env = make_env(args)
        result = rollout(env, controller, seed=seed, horizon=args.horizon, trace=False)
        raw = np.asarray(env.raw_state, dtype=float).tolist()
        failure = "truncated" if result["success"] else classify(raw, args)
        row = {
            "seed": seed,
            "steps": float(result["steps"]),
            "return": float(result["return"]),
            "success": bool(result["success"]),
            "failure": failure,
        }
        rows.append(row)
        if (not row["success"] and row["steps"] >= args.min_steps) or row["steps"] <= args.max_hard_steps:
            hard.append(seed)
        if args.limit and len(hard) >= args.limit:
            break
    summary = summarize_steps([row["steps"] for row in rows], args.horizon)
    result = {
        "status": "completed",
        "model_path": args.model_path,
        "seed_start": args.seed_start,
        "policy_terminal_scope": args.policy_terminal_scope,
        "policy_observation_mode": args.policy_observation_mode,
        "frame_stack": args.frame_stack,
        "episodes_requested": args.episodes,
        "episodes_run": len(rows),
        "hard_seeds": hard,
        "hard_seed_count": len(hard),
        "summary": summary,
        "failure_counts": dict(Counter(row["failure"] for row in rows if not row["success"])),
        "rows": rows,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "hard_seeds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "hard_seeds.txt").write_text("\n".join(str(seed) for seed in hard) + ("\n" if hard else ""), encoding="utf-8")
    write_markdown(result, out / "hard_seeds.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    s = result["summary"]
    lines = [
        "# Hard Seed Collection",
        "",
        f"Model: `{result['model_path']}`",
        f"Episodes run: `{result['episodes_run']}`",
        f"Hard seeds: `{result['hard_seed_count']}`",
        "",
        "| mean | p10 | success | max |",
        "|---:|---:|---:|---:|",
        f"| {s['mean_survival']:.1f} | {s['p10_survival']:.1f} | {s['success_rate']:.2f} | {s['max_survival']:.1f} |",
        "",
        "Failure counts:",
    ]
    for key, value in result["failure_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument(
        "--policy-terminal-scope",
        choices=["stabilize_chain", "selected", "all"],
        default="stabilize_chain",
    )
    parser.add_argument(
        "--policy-observation-mode", choices=POLICY_OBSERVATION_MODES, default="env"
    )
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1_100_000)
    parser.add_argument("--min-steps", type=float, default=0.0)
    parser.add_argument("--max-hard-steps", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="reports/hard_seed_collection")
    args = parser.parse_args()
    result = run_collect(args)
    print(json.dumps({"out": args.out, "hard_seed_count": result["hard_seed_count"], "success_rate": result["summary"]["success_rate"]}, indent=2))


if __name__ == "__main__":
    main()
