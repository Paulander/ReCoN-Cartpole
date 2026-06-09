from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=args.n_poles,
            horizon=args.horizon,
            dt=args.dt,
            dynamics_mode=args.dynamics_mode,
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
        )
    )


def make_controller(args: argparse.Namespace, mode: str) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode=mode,
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.model_path if mode == "recon_policy_terminal" else "",
            policy_terminal_blend=args.policy_terminal_blend,
        )
    )


def evaluate_mode(args: argparse.Namespace, mode: str, seeds: list[int]) -> list[dict[str, Any]]:
    controller = make_controller(args, mode)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=False)
        rows.append({"seed": seed, "steps": float(result["steps"]), "return": float(result["return"])})
    return rows


def threshold(n_poles: int) -> dict[str, float]:
    if n_poles == 3:
        return {"mean_survival": 475.0, "p10_survival": 400.0, "success_rate": 0.80}
    if n_poles == 4:
        return {"mean_survival": 475.0, "p10_survival": 350.0, "success_rate": 0.70}
    return {"mean_survival": 475.0, "p10_survival": 350.0, "success_rate": 0.80}


def summarize(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    return summarize_steps([row["steps"] for row in rows], horizon)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed_start + idx for idx in range(args.episodes)]
    static_rows = evaluate_mode(args, "static_recon", seeds)
    terminal_rows = evaluate_mode(args, "recon_policy_terminal", seeds)
    oracle_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    static_wins = 0
    terminal_wins = 0
    ties = 0
    for seed, static, terminal in zip(seeds, static_rows, terminal_rows):
        static_steps = float(static["steps"])
        terminal_steps = float(terminal["steps"])
        oracle_steps = max(static_steps, terminal_steps)
        if static_steps > terminal_steps:
            static_wins += 1
            winner = "static_recon"
        elif terminal_steps > static_steps:
            terminal_wins += 1
            winner = "recon_policy_terminal"
        else:
            ties += 1
            winner = "tie"
        row = {
            "seed": seed,
            "static_recon_steps": static_steps,
            "recon_policy_terminal_steps": terminal_steps,
            "oracle_steps": oracle_steps,
            "winner": winner,
        }
        per_seed.append(row)
        oracle_rows.append({"seed": seed, "steps": oracle_steps, "return": oracle_steps})
    result = {
        "status": "completed",
        "threshold": threshold(args.n_poles),
        "env": {
            "n_poles": args.n_poles,
            "horizon": args.horizon,
            "dt": args.dt,
            "dynamics_mode": args.dynamics_mode,
            "action_mode": args.action_mode,
            "discrete_action_bins": args.discrete_action_bins,
            "force_mag": args.force_mag,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
        },
        "model_path": args.model_path,
        "selection_mode": args.selection_mode,
        "policy_terminal_blend": args.policy_terminal_blend,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "static_recon": summarize(static_rows, args.horizon),
        "recon_policy_terminal": summarize(terminal_rows, args.horizon),
        "oracle_max": summarize(oracle_rows, args.horizon),
        "winner_counts": {"static_recon": static_wins, "recon_policy_terminal": terminal_wins, "tie": ties},
        "mean_terminal_minus_static": float(np.mean([row["recon_policy_terminal_steps"] - row["static_recon_steps"] for row in per_seed])),
        "per_seed": per_seed,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "oracle_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "oracle_analysis.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    static = result["static_recon"]
    terminal = result["recon_policy_terminal"]
    oracle = result["oracle_max"]
    lines = [
        "# Policy Terminal Oracle Analysis",
        "",
        f"Episodes: `{result['episodes']}`",
        f"Model: `{result['model_path']}`",
        f"Selection mode: `{result['selection_mode']}`",
        f"Policy terminal blend: `{result['policy_terminal_blend']}`",
        "",
        "| evaluator | mean | p10 | success | max |",
        "|---|---:|---:|---:|---:|",
        f"| static_recon | {static['mean_survival']:.1f} | {static['p10_survival']:.1f} | {static['success_rate']:.2f} | {static['max_survival']:.1f} |",
        f"| recon_policy_terminal | {terminal['mean_survival']:.1f} | {terminal['p10_survival']:.1f} | {terminal['success_rate']:.2f} | {terminal['max_survival']:.1f} |",
        f"| oracle_max | {oracle['mean_survival']:.1f} | {oracle['p10_survival']:.1f} | {oracle['success_rate']:.2f} | {oracle['max_survival']:.1f} |",
        "",
        "Winner counts:",
        f"- static_recon: `{result['winner_counts']['static_recon']}`",
        f"- recon_policy_terminal: `{result['winner_counts']['recon_policy_terminal']}`",
        f"- tie: `{result['winner_counts']['tie']}`",
        "",
        "## Interpretation",
        "",
        "`oracle_max` is an upper bound for a perfect per-episode gate between static ReCoN and the learned terminal. If it is below the solve threshold, a gate alone is unlikely to solve N=4.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=980_000)
    parser.add_argument("--out", default="reports/policy_terminal_oracle")
    args = parser.parse_args()
    result = run_analysis(args)
    print(json.dumps({"out": args.out, "oracle_success": result["oracle_max"]["success_rate"], "wall_clock_seconds": result["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
