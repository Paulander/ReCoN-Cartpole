from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.control.rewards import reward_tick
from recon_cartpole.recon.engine_runner import ReConCartPoleController
from recon_cartpole.training.ablations import summarize_steps

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_subchain_motif_gate import (  # noqa: E402
    collect_rows,
    fit_prototypes,
    make_controller,
    make_env,
    motif_scores,
    motif_vector,
)


def motif_score_for_step(model: dict[str, Any], raw_state: Any, diagnostics: dict[str, Any], args: argparse.Namespace) -> float:
    step = {
        "raw_state": np.asarray(raw_state, dtype=float).tolist(),
        "force": float(diagnostics.get("force", 0.0) or 0.0),
        "mingru_terminal": diagnostics.get("mingru_terminal", {}) or {},
        "mingru_passthrough": diagnostics.get("mingru_passthrough", {}) or {},
    }
    vector = motif_vector(step, args).reshape(1, -1)
    return float(motif_scores(model, vector)[0])


def force_to_action(force: float, args: argparse.Namespace) -> int:
    return int(action_from_force(float(force), "discrete", args.force_mag, args.discrete_action_bins))


def gated_action(
    base_action: int,
    diagnostics: dict[str, Any],
    score: float,
    args: argparse.Namespace,
    gate_mode: str,
    threshold: float,
) -> tuple[int, dict[str, Any]]:
    info: dict[str, Any] = {
        "mode": gate_mode,
        "score": float(score),
        "threshold": float(threshold),
        "changed": False,
        "reason": "below_threshold",
    }
    if gate_mode == "baseline" or score < threshold:
        return int(base_action), info

    mingru = diagnostics.get("mingru_passthrough") or diagnostics.get("mingru_terminal") or {}
    proposal = diagnostics.get("proposal", {}) or {}
    final_force = float(diagnostics.get("force", 0.0) or 0.0)
    chosen_force = final_force

    if gate_mode == "suppress_passthrough":
        if proposal.get("source_node") != "mingru_terminal" and not bool(mingru.get("passthrough_applied", False)):
            info["reason"] = "no_passthrough_to_suppress"
            return int(base_action), info
        base_force = mingru.get("passthrough_base_force")
        if base_force is None:
            base_proposal = mingru.get("passthrough_base_proposal", {}) or {}
            base_force = base_proposal.get("force")
        if base_force is None:
            info["reason"] = "missing_base_force"
            return int(base_action), info
        chosen_force = float(base_force)
        info["reason"] = "suppressed_passthrough"
    elif gate_mode == "force_passthrough":
        terminal_force = mingru.get("passthrough_force")
        if terminal_force is None:
            terminal_force = mingru.get("force") or mingru.get("terminal_force")
        if terminal_force is None:
            info["reason"] = "missing_terminal_force"
            return int(base_action), info
        chosen_force = float(terminal_force)
        info["reason"] = "forced_passthrough"
    elif gate_mode == "center_on_risk":
        chosen_force = 0.0
        info["reason"] = "centered_force"
    else:
        raise ValueError(f"unsupported gate mode: {gate_mode}")

    action = force_to_action(chosen_force, args)
    info.update(
        {
            "changed": int(action) != int(base_action),
            "base_action": int(base_action),
            "action": int(action),
            "base_force": final_force,
            "chosen_force": float(chosen_force),
        }
    )
    return int(action), info


def rollout_gated(seed: int, model: dict[str, Any], args: argparse.Namespace, gate_mode: str, threshold: float) -> dict[str, Any]:
    env = make_env(args)
    controller = make_controller(args)
    observation, info = env.reset(seed=seed)
    controller.start_episode()
    total_return = 0.0
    last_reward_tick = 0.0
    reward_history: list[float] = []
    gate_events = 0
    changed_actions = 0
    scores: list[float] = []
    for step in range(int(args.horizon)):
        controller.observe_reward(last_reward_tick)
        raw_before = info.get("raw_state")
        base_action, diagnostics = controller.act(observation, raw_before)
        score = motif_score_for_step(model, raw_before, diagnostics, args)
        action, gate_info = gated_action(base_action, diagnostics, score, args, gate_mode, threshold)
        if score >= threshold:
            gate_events += 1
        if gate_info.get("changed"):
            changed_actions += 1
        scores.append(score)
        before_obs = np.asarray(observation, dtype=float)
        next_observation, env_reward, terminated, truncated, info = env.step(action)
        total_return += float(env_reward)
        last_reward_tick = reward_tick(
            before_obs,
            next_observation,
            raw_before,
            info.get("raw_state"),
            controller.config.n_poles,
            terminated,
        )
        reward_history.append(float(last_reward_tick))
        observation = next_observation
        if terminated or truncated:
            break
    controller.observe_reward(last_reward_tick)
    controller.end_episode(reward_history, total_return, args.horizon)
    steps = step + 1
    return {
        "seed": int(seed),
        "steps": int(steps),
        "return": float(total_return),
        "success": bool(steps >= args.horizon),
        "gate_events": int(gate_events),
        "changed_actions": int(changed_actions),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_p95": float(np.percentile(scores, 95)) if scores else 0.0,
        "score_max": float(np.max(scores)) if scores else 0.0,
    }


def evaluate_gate(model: dict[str, Any], args: argparse.Namespace, gate_mode: str, threshold: float) -> dict[str, Any]:
    per_seed = [
        rollout_gated(seed, model, args, gate_mode, threshold)
        for seed in range(int(args.heldout_seed_start), int(args.heldout_seed_start) + int(args.heldout_episodes))
    ]
    steps = [float(item["steps"]) for item in per_seed]
    summary = summarize_steps(steps, args.horizon)
    summary.update(
        {
            "gate_mode": gate_mode,
            "threshold": float(threshold),
            "episodes": len(per_seed),
            "gate_events": int(sum(item["gate_events"] for item in per_seed)),
            "changed_actions": int(sum(item["changed_actions"] for item in per_seed)),
            "per_seed": per_seed,
        }
    )
    return summary


def threshold_values(args: argparse.Namespace, model: dict[str, Any], train_data: dict[str, Any]) -> list[float]:
    explicit = str(getattr(args, "thresholds", "") or "").strip()
    if explicit:
        return [float(item.strip()) for item in explicit.replace(",", " ").split() if item.strip()]
    scores = motif_scores(model, train_data["x"])
    positive = scores[train_data["y"] == 1]
    if positive.size == 0:
        return [0.0]
    return [float(np.percentile(positive, q)) for q in args.threshold_percentiles]


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Motif-Gated Passthrough Evaluation",
        "",
        "This evaluates motif-risk as an online control gate. No solve claim is made; all rows use held-out seeds.",
        "",
        f"Prototype train seeds: `{result['train_seed_start']}` x `{result['train_episodes']}`; held-out seeds: `{result['heldout_seed_start']}` x `{result['heldout_episodes']}`",
        "",
        "| mode | threshold | mean | p10 | success | gate events | changed actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['gate_mode']} | {row['threshold']:.3f} | {row['mean_survival']:.1f} | "
            f"{row['p10_survival']:.1f} | {row['success_rate']:.3f} | {row['gate_events']} | {row['changed_actions']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: a useful gate must improve held-out success or lower-tail survival without relying on train seeds. Changed actions show whether the motif score had causal opportunity, not whether it helped.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.include_recon_diagnostics = True
    train_data = collect_rows(args, args.train_seed_start, args.train_episodes)
    model = fit_prototypes(train_data["x"], train_data["y"])
    thresholds = threshold_values(args, model, train_data)
    rows: list[dict[str, Any]] = []
    rows.append(evaluate_gate(model, args, "baseline", float("inf")))
    for mode in args.gate_modes:
        for threshold in thresholds:
            rows.append(evaluate_gate(model, args, mode, threshold))
    rows.sort(key=lambda item: (item["success_rate"], item["mean_survival"], item["p10_survival"]), reverse=True)
    result = {
        "status": "completed",
        "train_seed_start": int(args.train_seed_start),
        "train_episodes": int(args.train_episodes),
        "heldout_seed_start": int(args.heldout_seed_start),
        "heldout_episodes": int(args.heldout_episodes),
        "thresholds": thresholds,
        "prototype_positive_rows": int(model["positive_rows"]),
        "prototype_negative_rows": int(model["negative_rows"]),
        "rows": rows,
        "mechanisms": {
            "subchain_motif_prototypes": True,
            "online_motif_gate": True,
            "gain_mutation": False,
            "train_seed_solve_claim": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "report.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    from train_subchain_motif_gate import build_parser as motif_parser

    parser = motif_parser()
    parser.description = "Evaluate online motif-risk gates for minGRU passthrough."
    parser.set_defaults(out="reports/n4_motif_gated_passthrough")
    parser.add_argument("--gate-modes", nargs="+", default=["suppress_passthrough", "force_passthrough"])
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--threshold-percentiles", nargs="+", type=float, default=[50.0, 70.0, 85.0, 95.0])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    best = result["rows"][0]
    print(json.dumps({"out": args.out, "best_mode": best["gate_mode"], "best_success": best["success_rate"], "best_mean": best["mean_survival"]}, indent=2))


if __name__ == "__main__":
    main()
