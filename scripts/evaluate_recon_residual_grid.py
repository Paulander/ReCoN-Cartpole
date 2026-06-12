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
from recon_cartpole.control.policy_observation import POLICY_OBSERVATION_MODES


def _floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def seeds(args: argparse.Namespace) -> list[int]:
    starts = args.seed_starts or [args.seed_start]
    out: list[int] = []
    for start in starts:
        out.extend(int(start) + idx for idx in range(args.episodes_per_start))
    return out


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


def tail_metrics(steps: list[float], horizon: int, cvar_fraction: float = 0.10) -> dict[str, Any]:
    summary = summarize_steps(steps, horizon)
    values = np.asarray(steps, dtype=float)
    if values.size:
        count = max(1, int(np.ceil(values.size * cvar_fraction)))
        summary["cvar_survival"] = float(np.mean(np.sort(values)[:count]))
    else:
        summary["cvar_survival"] = 0.0
    return summary


def controller_for(args: argparse.Namespace, threshold: float, max_force: float) -> ReConCartPoleController:
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
            policy_terminal_path=args.base_model_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.base_observation_mode,
            policy_terminal_normalizer_path=args.base_normalizer_path,
            residual_policy_terminal_path=args.residual_model_path,
            residual_policy_terminal_mode=args.residual_mode,
            residual_policy_terminal_action_bins=args.residual_action_bins,
            residual_policy_terminal_max_force=max_force,
            residual_policy_terminal_gate_threshold=threshold,
            residual_policy_terminal_feature_mode=args.residual_feature_mode,
            residual_policy_terminal_hold_steps=int(getattr(args, "residual_hold_steps", 1)),
        )
    )


def evaluate_candidate(args: argparse.Namespace, seed_values: list[int], threshold: float, max_force: float) -> dict[str, Any]:
    controller = controller_for(args, threshold, max_force)
    steps: list[float] = []
    returns: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed in seed_values:
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=False)
        step_count = float(result["steps"])
        steps.append(step_count)
        returns.append(float(result["return"]))
        per_seed.append({"seed": int(seed), "steps": int(step_count), "success": step_count >= args.horizon})
    summary = tail_metrics(steps, args.horizon, args.cvar_fraction)
    summary.update(
        {
            "threshold": float(threshold),
            "max_residual_force": float(max_force),
            "episodes": len(seed_values),
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "per_seed": per_seed,
        }
    )
    return summary


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# ReCoN Residual Gate Sweep",
        "",
        f"Status: `{result.get('status')}`",
        f"Base model: `{result.get('base_model_path')}`",
        f"Residual model: `{result.get('residual_model_path')}`",
        f"Residual feature mode: `{result.get('residual_feature_mode')}`",
        f"Residual hold steps: `{result.get('residual_hold_steps', 1)}`",
        "",
        "| threshold | max force | mean | p10 | cvar | success | episodes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("candidates", []):
        lines.append(
            f"| {row['threshold']:.3f} | {row['max_residual_force']:.2f} | {row['mean_survival']:.1f} | "
            f"{row['p10_survival']:.1f} | {row['cvar_survival']:.1f} | {row['success_rate']:.3f} | {row['episodes']} |"
        )
    best = result.get("best") or {}
    if best:
        lines.extend(["", f"Best candidate: threshold `{best['threshold']:.3f}`, max force `{best['max_residual_force']:.2f}`."])
    lines.extend([
        "",
        "## Claim Discipline",
        "",
        "This is a held-out/evaluation sweep over a fixed learned residual. No train-seed solve claims are made from this table.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_values = seeds(args)
    rows: list[dict[str, Any]] = []
    for threshold in _floats(args.thresholds):
        for max_force in _floats(args.max_residual_forces):
            row = evaluate_candidate(args, seed_values, threshold, max_force)
            rows.append(row)
            partial = {
                "status": "running",
                "base_model_path": args.base_model_path,
                "residual_model_path": args.residual_model_path,
                "residual_feature_mode": args.residual_feature_mode,
                "residual_hold_steps": int(getattr(args, "residual_hold_steps", 1)),
                "seed_starts": args.seed_starts or [args.seed_start],
                "episodes_per_start": args.episodes_per_start,
                "candidates": rows,
                "best": max(rows, key=lambda item: (item["success_rate"], item["p10_survival"], item["cvar_survival"], item["mean_survival"])),
            }
            (out / "summary.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
            write_markdown(partial, out / "summary.md")
    best = max(rows, key=lambda item: (item["success_rate"], item["p10_survival"], item["cvar_survival"], item["mean_survival"])) if rows else None
    result = {
        "status": "completed",
        "out": str(out),
        "base_model_path": args.base_model_path,
        "residual_model_path": args.residual_model_path,
        "residual_feature_mode": args.residual_feature_mode,
        "residual_hold_steps": int(getattr(args, "residual_hold_steps", 1)),
        "seed_starts": args.seed_starts or [args.seed_start],
        "episodes_per_start": args.episodes_per_start,
        "candidates": rows,
        "best": best,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a learned ReCoN residual terminal across gate thresholds.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--residual-model-path", required=True)
    parser.add_argument("--base-normalizer-path", default="")
    parser.add_argument("--base-observation-mode", choices=POLICY_OBSERVATION_MODES, default="normalized_raw")
    parser.add_argument("--residual-mode", choices=["force", "bin_delta"], default="bin_delta")
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics", "subchain_diagnostics"], default="proposal_diagnostics")
    parser.add_argument("--residual-action-bins", type=int, default=5)
    parser.add_argument("--residual-hold-steps", type=int, default=1)
    parser.add_argument("--thresholds", default="0.30,0.50,0.62,0.75,0.90")
    parser.add_argument("--max-residual-forces", default="4.0")
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
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--episodes-per-start", type=int, default=15)
    parser.add_argument("--cvar-fraction", type=float, default=0.10)
    parser.add_argument("--out", default="reports/recon_residual_gate_sweep")
    return parser


def main() -> None:
    result = run_sweep(build_parser().parse_args())
    best = result.get("best") or {}
    print(json.dumps({"out": result.get("out"), "status": result["status"], "best_success": best.get("success_rate")}, indent=2))


if __name__ == "__main__":
    main()
