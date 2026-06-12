from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig, SubchainBiasConfig
from recon_cartpole.training.ablations import summarize_steps


def seed_grid(starts: list[int], episodes_per_start: int) -> list[int]:
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(episodes_per_start)))
    return seeds


def candidate_configs() -> dict[str, SubchainBiasConfig]:
    base = SubchainBiasConfig(enabled=True)
    return {
        "baseline_no_subchain_bias": SubchainBiasConfig(enabled=False),
        "conservative_default": base,
        "low_blend_default": replace(base, blend=0.10, confidence_boost=0.03, urgency_boost=0.06),
        "delta_angle_focus": replace(
            base,
            blend=0.20,
            mean_angle_gain=0.0,
            mean_velocity_gain=0.0,
            delta_angle_gain=16.0,
            delta_velocity_gain=2.5,
        ),
        "mean_pair_focus": replace(
            base,
            blend=0.20,
            mean_angle_gain=18.0,
            mean_velocity_gain=4.0,
            delta_angle_gain=4.0,
            delta_velocity_gain=1.0,
        ),
        "outer_pair_weighted": replace(
            base,
            blend=0.18,
            outer_pair_weight=0.45,
            mean_angle_gain=12.0,
            mean_velocity_gain=3.0,
            delta_angle_gain=12.0,
            delta_velocity_gain=2.0,
        ),
    }


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


def make_controller(args: argparse.Namespace, config: SubchainBiasConfig) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_feedforward_terminal_frozen",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.model_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_observation_mode,
            subchain_bias=config,
        )
    )


def rollout(args: argparse.Namespace, controller: ReConCartPoleController, seed: int) -> dict[str, Any]:
    env = make_env(args)
    observation, info = env.reset(seed=seed)
    controller.start_episode()
    total_return = 0.0
    subchain_applied = 0
    subchain_action_changed = 0
    subchain_force_deltas: list[float] = []
    subchain_pressures: list[float] = []
    for step in range(args.horizon):
        raw_before = info.get("raw_state")
        action, diagnostics = controller.act(observation, raw_before)
        bias = diagnostics.get("subchain_bias", {}) or {}
        if bias.get("applied"):
            subchain_applied += 1
            base_force = float(bias.get("base_force", 0.0))
            proposal_force = float(bias.get("proposal_force", 0.0))
            subchain_force_deltas.append(abs(proposal_force - base_force))
            base_action = action_from_force(base_force, "discrete", args.force_mag, args.discrete_action_bins)
            proposal_action = action_from_force(proposal_force, "discrete", args.force_mag, args.discrete_action_bins)
            if int(np.asarray(base_action).reshape(-1)[0]) != int(np.asarray(proposal_action).reshape(-1)[0]):
                subchain_action_changed += 1
            subchain_pressures.append(float(bias.get("max_pressure", 0.0)))
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        if terminated or truncated:
            break
    steps = step + 1
    return {
        "seed": int(seed),
        "steps": int(steps),
        "return": float(total_return),
        "success": bool(steps >= args.horizon),
        "subchain_applied_ticks": int(subchain_applied),
        "subchain_action_changed_ticks": int(subchain_action_changed),
        "subchain_force_delta_mean": float(np.mean(subchain_force_deltas)) if subchain_force_deltas else 0.0,
        "subchain_pressure_mean": float(np.mean(subchain_pressures)) if subchain_pressures else 0.0,
    }


def evaluate_config(args: argparse.Namespace, name: str, config: SubchainBiasConfig, seeds: list[int]) -> dict[str, Any]:
    started = time.perf_counter()
    controller = make_controller(args, config)
    rows = [rollout(args, controller, seed) for seed in seeds]
    steps = [float(row["steps"]) for row in rows]
    return {
        "name": name,
        "config": asdict(config),
        "mechanisms": controller.learning_mechanisms(),
        "episodes": len(rows),
        "per_seed": rows,
        "subchain_applied_ticks_mean": float(np.mean([row["subchain_applied_ticks"] for row in rows])) if rows else 0.0,
        "subchain_action_changed_ticks_mean": float(np.mean([row["subchain_action_changed_ticks"] for row in rows])) if rows else 0.0,
        "subchain_force_delta_mean": float(np.mean([row["subchain_force_delta_mean"] for row in rows])) if rows else 0.0,
        "subchain_pressure_mean": float(np.mean([row["subchain_pressure_mean"] for row in rows])) if rows else 0.0,
        "wall_clock_seconds": time.perf_counter() - started,
        **summarize_steps(steps, args.horizon),
    }


def write_report(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Subchain Bias Grid Evaluation",
        "",
        f"Model: `{result['model_path']}`",
        f"Seeds: `{result['seed_starts']}` x `{result['episodes_per_start']}` episodes/start",
        f"Environment: `{result['env']}`",
        "",
        "| candidate | mean | p10 | success | max | applied ticks | action changes | force delta | pressure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['name']} | {row['mean_survival']:.2f} | {row['p10_survival']:.1f} | "
            f"{row['success_rate']:.4f} | {row['max_survival']:.1f} | "
            f"{row['subchain_applied_ticks_mean']:.1f} | {row['subchain_action_changed_ticks_mean']:.1f} | "
            f"{row['subchain_force_delta_mean']:.3f} | {row['subchain_pressure_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "This is a held-out diagnostic of a compositional ReCoN bias on top of a frozen PPO terminal. It is not training evidence and is not a solve claim.",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen PPO terminal with optional ReCoN subchain bias configs.")
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
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--policy-observation-mode", default="normalized_raw")
    parser.add_argument("--seed-starts", type=int, nargs="+", default=[980_000, 1300_000, 1500_000, 1600_000])
    parser.add_argument("--episodes-per-start", type=int, default=10)
    parser.add_argument("--out", default="reports/n4_subchain_bias_grid")
    args = parser.parse_args()

    seeds = seed_grid(args.seed_starts, args.episodes_per_start)
    candidates = [evaluate_config(args, name, cfg, seeds) for name, cfg in candidate_configs().items()]
    result = {
        "status": "completed",
        "model_path": args.model_path,
        "seed_starts": [int(seed) for seed in args.seed_starts],
        "episodes_per_start": int(args.episodes_per_start),
        "env": {
            "n_poles": args.n_poles,
            "horizon": args.horizon,
            "dt": args.dt,
            "dynamics_mode": args.dynamics_mode,
            "discrete_action_bins": args.discrete_action_bins,
            "force_mag": args.force_mag,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
        },
        "candidates": candidates,
    }
    write_report(result, Path(args.out))
    best = max(candidates, key=lambda row: (row["success_rate"], row["p10_survival"], row["mean_survival"]))
    print(json.dumps({"out": args.out, "best": best["name"], "success_rate": best["success_rate"], "p10": best["p10_survival"], "mean": best["mean_survival"]}, indent=2))


if __name__ == "__main__":
    main()
