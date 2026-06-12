from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminal
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_recurrent_terminal_ladder import terminal_config  # noqa: E402


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


def classify(raw: Any, args: argparse.Namespace) -> str:
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size == 0:
        return "unknown"
    x = float(values[0])
    if abs(x) > 2.2:
        return "rail_left" if x < 0 else "rail_right"
    angles = values[2 : 2 + int(args.n_poles)]
    if angles.size:
        return f"pole_{int(np.argmax(np.abs(angles)))}_angle"
    return "unknown"


def seed_values(args: argparse.Namespace) -> list[int]:
    return [int(args.seed_start) + idx for idx in range(int(args.episodes))]


def run_pure_mingru(args: argparse.Namespace, terminal: MinGRUTerminal, seed: int) -> tuple[dict[str, Any], Any]:
    env = make_env(args)
    obs, info = env.reset(seed=seed)
    terminal.reset()
    total = 0.0
    steps = 0
    for step in range(int(args.horizon)):
        prediction = terminal.predict(obs, info.get("raw_state"), {})
        force = 0.0 if prediction.force is None else float(prediction.force)
        action = action_from_force(force, "discrete", args.force_mag, args.discrete_action_bins)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps = step + 1
        if terminated or truncated:
            break
    return {
        "return": total,
        "steps": steps,
        "success": steps >= int(args.horizon) and total >= float(args.horizon) - 1.0,
    }, env.raw_state


def make_recon_controller(args: argparse.Namespace) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_mingru_terminal",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            mingru_terminal=terminal_config(args, args.checkpoint_path, args.hidden_size, args.sequence_length),
        )
    )


def run_recon_mingru(args: argparse.Namespace, controller: ReConCartPoleController, seed: int) -> tuple[dict[str, Any], Any]:
    env = make_env(args)
    result = rollout(env, controller, seed=seed, horizon=args.horizon, trace=False)
    return result, env.raw_state


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hard: list[int] = []

    terminal = None
    controller = None
    if args.policy_mode == "pure_mingru_policy":
        terminal = MinGRUTerminal(
            args.n_poles,
            args.force_mag,
            args.discrete_action_bins,
            terminal_config(args, args.checkpoint_path, args.hidden_size, args.sequence_length),
        )
    else:
        controller = make_recon_controller(args)

    for seed in seed_values(args):
        if terminal is not None:
            episode, raw = run_pure_mingru(args, terminal, seed)
        else:
            assert controller is not None
            episode, raw = run_recon_mingru(args, controller, seed)
        success = bool(episode["success"])
        steps = float(episode["steps"])
        failure = "success" if success else classify(raw, args)
        row = {
            "seed": int(seed),
            "steps": steps,
            "return": float(episode["return"]),
            "success": success,
            "failure": failure,
        }
        rows.append(row)
        if (not success and steps >= float(args.min_steps)) or steps <= float(args.max_hard_steps):
            hard.append(int(seed))
        if args.limit and len(hard) >= int(args.limit):
            break

    summary = summarize_steps([row["steps"] for row in rows], args.horizon)
    result = {
        "status": "completed",
        "out": str(out),
        "checkpoint_path": args.checkpoint_path,
        "policy_mode": args.policy_mode,
        "seed_start": int(args.seed_start),
        "episodes_requested": int(args.episodes),
        "episodes_run": len(rows),
        "hard_seeds": hard,
        "hard_seed_count": len(hard),
        "summary": summary,
        "failure_counts": dict(Counter(row["failure"] for row in rows if not row["success"])),
        "rows": rows,
        "config": {
            "n_poles": int(args.n_poles),
            "horizon": int(args.horizon),
            "dt": float(args.dt),
            "dynamics_mode": args.dynamics_mode,
            "discrete_action_bins": int(args.discrete_action_bins),
            "force_mag": float(args.force_mag),
            "initial_angle_range": float(args.initial_angle_range),
            "force_noise": float(args.force_noise),
            "link_coupling": float(args.link_coupling),
            "observation_mode": args.observation_mode,
            "include_prev_force": bool(args.include_prev_force),
            "include_context": bool(args.include_context),
            "include_motif_score": bool(args.include_motif_score),
            "motif_model_path": str(args.motif_model_path),
            "hidden_size": int(args.hidden_size),
            "sequence_length": int(args.sequence_length),
            "min_steps": float(args.min_steps),
            "max_hard_steps": float(args.max_hard_steps),
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "hard_seeds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    seed_text = "\n".join(str(seed) for seed in hard) + ("\n" if hard else "")
    (out / "hard_seeds.txt").write_text(seed_text, encoding="utf-8")
    write_markdown(result, out / "hard_seeds.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    s = result["summary"]
    lines = [
        "# minGRU Hard Seed Collection",
        "",
        f"Checkpoint: `{result['checkpoint_path']}`",
        f"Policy mode: `{result['policy_mode']}`",
        f"Episodes run: `{result['episodes_run']}`",
        f"Hard seeds: `{result['hard_seed_count']}`",
        "",
        "| mean | p10 | success | max |",
        "|---:|---:|---:|---:|",
        f"| {s['mean_survival']:.1f} | {s['p10_survival']:.1f} | {s['success_rate']:.3f} | {s['max_survival']:.1f} |",
        "",
        "Failure counts:",
    ]
    for key, value in sorted(result["failure_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect hard seeds for a trained minGRU recurrent terminal.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--policy-mode", choices=["pure_mingru_policy", "recon_mingru_terminal"], default="recon_mingru_terminal"
    )
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
    parser.add_argument(
        "--observation-mode",
        choices=[
            "env",
            "normalized_raw",
            "normalized_raw_prev_force",
            "normalized_raw4",
            "normalized_raw4_prev_force",
            "normalized_raw4_subchains",
            "normalized_raw4_subchains_prev_force",
        ],
        default="normalized_raw4_subchains_prev_force",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-motif-score", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motif-model-path", default="")
    parser.add_argument("--motif-score-scale", type=float, default=10.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-logit-margin-floor", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=5_200_000)
    parser.add_argument("--min-steps", type=float, default=350.0)
    parser.add_argument("--max-hard-steps", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="reports/mingru_hard_seeds")
    return parser


def main() -> None:
    result = run_collect(build_parser().parse_args())
    print(
        json.dumps(
            {
                "out": result["out"],
                "hard_seed_count": result["hard_seed_count"],
                "success_rate": result["summary"]["success_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
