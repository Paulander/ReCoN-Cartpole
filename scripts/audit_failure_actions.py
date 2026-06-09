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


def make_env(args: argparse.Namespace, force_noise: float | None = None) -> CartPoleNEnv:
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
            force_noise=args.force_noise if force_noise is None else force_noise,
            link_coupling=args.link_coupling,
        )
    )


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
        )
    )


def classify_final(raw: list[float], args: argparse.Namespace) -> str:
    if not raw:
        return "unknown"
    x = float(raw[0])
    if abs(x) > 2.2:
        return "rail_left" if x < 0 else "rail_right"
    angles = raw[2 : 2 + args.n_poles]
    if angles:
        idx = int(np.argmax(np.abs(angles)))
        return f"pole_{idx}_angle"
    return "unknown"


def collect_episode(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env = make_env(args)
    controller = make_controller(args)
    obs, info = env.reset(seed=seed)
    controller.start_episode()
    states: list[dict[str, Any]] = []
    total = 0.0
    for step in range(args.horizon):
        raw_before = np.asarray(info["raw_state"], dtype=float).copy()
        action, diagnostics = controller.act(obs, raw_before)
        states.append(
            {
                "step": step,
                "raw_before": raw_before.tolist(),
                "obs_before": np.asarray(obs, dtype=float).tolist(),
                "action": int(action),
                "force": float(diagnostics.get("force", 0.0)),
                "selected_regime": diagnostics.get("selected_regime", ""),
                "proposal": diagnostics.get("proposal", {}),
                "policy_terminal": diagnostics.get("policy_terminal", {}),
            }
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            return {
                "seed": seed,
                "steps": step + 1,
                "return": total,
                "success": bool(truncated and step + 1 >= args.horizon),
                "failure": classify_final(np.asarray(info.get("raw_state", [])).tolist(), args) if terminated else "truncated",
                "states": states,
            }
    return {"seed": seed, "steps": args.horizon, "return": total, "success": True, "failure": "truncated", "states": states}


def set_env_state(env: CartPoleNEnv, raw_state: list[float], step: int) -> None:
    env.state = np.asarray(raw_state, dtype=float).copy()
    env.steps = int(step)




def stability_margin(raw_state: list[float], args: argparse.Namespace) -> float:
    if not raw_state:
        return -10.0
    raw = np.asarray(raw_state, dtype=float)
    n = args.n_poles
    x = abs(float(raw[0])) / 2.4
    theta = np.abs(raw[2 : 2 + n]) / (12.0 * 2.0 * np.pi / 360.0)
    theta_dot = np.abs(raw[2 + n : 2 + 2 * n]) / 5.0
    angle_pressure = float(np.max(theta)) if theta.size else 0.0
    velocity_pressure = float(np.mean(theta_dot)) if theta_dot.size else 0.0
    return float(1.0 - angle_pressure - 0.10 * x - 0.03 * velocity_pressure)


def counterfactual_score(args: argparse.Namespace, raw_state: list[float], step: int, first_action: int) -> dict[str, Any]:
    env = make_env(args, force_noise=0.0 if args.counterfactual_no_noise else args.force_noise)
    controller = make_controller(args)
    set_env_state(env, raw_state, step)
    obs = env._get_obs()  # counterfactual analysis needs exact env state observation
    controller.start_episode()
    total = 0.0
    actions = [int(first_action)]
    obs, reward, terminated, truncated, info = env.step(first_action)
    total += float(reward)
    survived = 1
    final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
    if terminated or truncated:
        margin = stability_margin(final_raw, args)
        return {"action": int(first_action), "survived": survived, "return": total, "ended": bool(terminated or truncated), "margin": margin, "score": survived + margin, "actions": actions}
    for _ in range(args.probe_horizon - 1):
        raw = np.asarray(info["raw_state"], dtype=float).copy()
        action, _diagnostics = controller.act(obs, raw)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        survived += 1
        final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
        if terminated or truncated:
            break
    margin = stability_margin(final_raw, args)
    return {"action": int(first_action), "survived": survived, "return": total, "ended": bool(terminated or truncated), "margin": margin, "score": survived + margin, "actions": actions[:10]}


def audit_state(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    options = [counterfactual_score(args, state["raw_before"], int(state["step"]), action) for action in range(args.discrete_action_bins)]
    best_score = max(float(item["score"]) for item in options)
    best_actions = [item["action"] for item in options if abs(float(item["score"]) - best_score) <= args.score_tolerance]
    chosen = int(state["action"])
    chosen_score = next(item for item in options if item["action"] == chosen)
    return {
        "step": state["step"],
        "chosen_action": chosen,
        "chosen_force": state["force"],
        "selected_regime": state["selected_regime"],
        "policy_terminal": state.get("policy_terminal", {}),
        "best_actions": best_actions,
        "chosen_is_best": chosen in best_actions,
        "chosen_survived": chosen_score["survived"],
        "best_survived": max(item["survived"] for item in options),
        "best_score": best_score,
        "chosen_score": float(chosen_score["score"]),
        "survival_gap": max(item["survived"] for item in options) - chosen_score["survived"],
        "score_gap": best_score - float(chosen_score["score"]),
        "options": options,
        "raw_before": state["raw_before"],
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []
    for idx in range(args.episodes):
        episode = collect_episode(args, args.seed_start + idx)
        episodes.append({key: episode[key] for key in ("seed", "steps", "return", "success", "failure")})
        if episode["success"]:
            continue
        if args.failure_offsets:
            window = []
            for offset in args.failure_offsets:
                idx = len(episode["states"]) - 1 - int(offset)
                if 0 <= idx < len(episode["states"]):
                    window.append(episode["states"][idx])
            seen_steps = set()
            window = [state for state in window if not (state["step"] in seen_steps or seen_steps.add(state["step"]))]
        else:
            window = episode["states"][-args.failure_window :]
            if args.max_states_per_failure > 0:
                window = window[-args.max_states_per_failure :]
        for state in window:
            row = audit_state(args, state)
            row["seed"] = episode["seed"]
            row["episode_steps"] = episode["steps"]
            row["failure"] = episode["failure"]
            audited.append(row)
        partial = {"episodes": episodes, "audited_states": len(audited)}
        (out / "partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    steps = [episode["steps"] for episode in episodes]
    gaps = [row["survival_gap"] for row in audited]
    score_gaps = [row["score_gap"] for row in audited]
    mistake_rows = [row for row in audited if not row["chosen_is_best"]]
    result = {
        "status": "completed",
        "model_path": args.model_path,
        "env": {
            "n_poles": args.n_poles,
            "horizon": args.horizon,
            "dt": args.dt,
            "dynamics_mode": args.dynamics_mode,
            "discrete_action_bins": args.discrete_action_bins,
            "force_mag": args.force_mag,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "counterfactual_no_noise": args.counterfactual_no_noise,
        },
        "probe_horizon": args.probe_horizon,
        "failure_offsets": args.failure_offsets,
        "episodes_summary": summarize_steps(steps, args.horizon),
        "episodes": episodes,
        "failure_counts": dict(Counter(episode["failure"] for episode in episodes if not episode["success"])),
        "audited_states": len(audited),
        "mistake_states": len(mistake_rows),
        "mistake_rate": float(len(mistake_rows) / len(audited)) if audited else 0.0,
        "mean_survival_gap": float(np.mean(gaps)) if gaps else 0.0,
        "p90_survival_gap": float(np.percentile(gaps, 90)) if gaps else 0.0,
        "mean_score_gap": float(np.mean(score_gaps)) if score_gaps else 0.0,
        "p90_score_gap": float(np.percentile(score_gaps, 90)) if score_gaps else 0.0,
        "chosen_action_counts": dict(Counter(str(row["chosen_action"]) for row in audited)),
        "best_action_counts": dict(Counter(str(action) for row in audited for action in row["best_actions"])),
        "top_mistakes": sorted(mistake_rows, key=lambda row: row["survival_gap"], reverse=True)[: args.keep_examples],
        "audits": audited if args.keep_all else [],
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "failure_action_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "failure_action_audit.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["episodes_summary"]
    lines = [
        "# Failure Action Audit",
        "",
        f"Model: `{result['model_path']}`",
        f"Probe horizon: `{result['probe_horizon']}`",
        f"Failure offsets: `{result.get('failure_offsets', [])}`",
        f"Audited states: `{result['audited_states']}`",
        f"Mistake rate: `{result['mistake_rate']:.2f}`",
        f"Mean survival gap: `{result['mean_survival_gap']:.1f}`",
        f"Mean score gap: `{result.get('mean_score_gap', 0.0):.3f}`",
        "",
        "| episodes mean | p10 | success | max |",
        "|---:|---:|---:|---:|",
        f"| {summary['mean_survival']:.1f} | {summary['p10_survival']:.1f} | {summary['success_rate']:.2f} | {summary['max_survival']:.1f} |",
        "",
        "Failure counts:",
    ]
    for key, value in result["failure_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Action counts:", f"- chosen: `{result['chosen_action_counts']}`", f"- best: `{result['best_action_counts']}`"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "For each audited pre-failure state, the script forces each discrete action once, then lets the same ReCoN policy-terminal controller continue for the probe horizon. A high mistake rate or large survival gap means the terminal/action choice is locally improvable near failures.",
    ])
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
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed-start", type=int, default=980_000)
    parser.add_argument("--failure-window", type=int, default=20)
    parser.add_argument("--failure-offsets", type=int, nargs="*", default=[])
    parser.add_argument("--max-states-per-failure", type=int, default=5)
    parser.add_argument("--probe-horizon", type=int, default=80)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--counterfactual-no-noise", action="store_true")
    parser.add_argument("--keep-examples", type=int, default=20)
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument("--out", default="reports/failure_action_audit")
    args = parser.parse_args()
    result = run_audit(args)
    print(json.dumps({"out": args.out, "mistake_rate": result["mistake_rate"], "mean_survival_gap": result["mean_survival_gap"], "mean_score_gap": result["mean_score_gap"]}, indent=2))


if __name__ == "__main__":
    main()
