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


def env_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
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
    }


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(CartPoleNConfig(**env_payload(args)))


def make_controller(model_path: str, args: argparse.Namespace) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_policy_terminal",
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=model_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_frame_stack=args.frame_stack,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_observation_mode,
            policy_terminal_recurrent=args.policy_terminal_recurrent,
            policy_terminal_normalizer_path=args.policy_terminal_normalizer_path,
        )
    )


def classify_final(raw_state: list[float], steps: int, args: argparse.Namespace) -> str:
    if steps >= args.horizon:
        return "success"
    raw = np.asarray(raw_state, dtype=float)
    if raw.size < 2 + 2 * args.n_poles:
        return "unknown"
    x = float(raw[0])
    if x <= -2.4 * 0.98:
        return "rail_left"
    if x >= 2.4 * 0.98:
        return "rail_right"
    angles = raw[2 : 2 + args.n_poles]
    if angles.size:
        return f"pole_{int(np.argmax(np.abs(angles)))}_angle"
    return "unknown"


def trim_policy_terminal(info: dict[str, Any]) -> dict[str, Any]:
    policy = info.get("policy_terminal", {}) or {}
    residual = policy.get("residual_policy_terminal", {}) or {}
    return {
        "available": bool(policy.get("available", False)),
        "action": policy.get("action", []),
        "base_force": policy.get("base_force"),
        "force": policy.get("force"),
        "residual_action": residual.get("action", []),
        "residual_delta": residual.get("residual_delta"),
        "residual_gate": residual.get("risk_gate"),
    }


def trim_diagnostics(diagnostics: dict[str, Any], raw_before: Any, step: int) -> dict[str, Any]:
    proposal = diagnostics.get("proposal", {}) or {}
    return {
        "step": int(step),
        "raw_before": np.asarray(raw_before, dtype=float).reshape(-1).tolist(),
        "action": int(diagnostics.get("action", -1)) if "action" in diagnostics else None,
        "force": float(diagnostics.get("force", 0.0)),
        "selected_regime": diagnostics.get("selected_regime", ""),
        "proposal_regime": proposal.get("regime", ""),
        "proposal_score": proposal.get("weighted_score", proposal.get("score")),
        "policy_terminal": trim_policy_terminal(diagnostics),
    }


def collect_episode(model_path: str, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env = make_env(args)
    controller = make_controller(model_path, args)
    observation, info = env.reset(seed=seed)
    controller.start_episode()
    total_return = 0.0
    actions: list[int] = []
    forces: list[float] = []
    selected: list[str] = []
    step_summaries: list[dict[str, Any]] = []
    final_raw: list[float] = np.asarray(info.get("raw_state", []), dtype=float).tolist()
    for step in range(args.horizon):
        raw_before = np.asarray(info.get("raw_state", []), dtype=float).copy()
        action, diagnostics = controller.act(observation, raw_before)
        action_int = int(np.asarray(action).reshape(-1)[0])
        diag = dict(diagnostics)
        diag["action"] = action_int
        step_summaries.append(trim_diagnostics(diag, raw_before, step))
        actions.append(action_int)
        forces.append(float(diag.get("force", 0.0)))
        selected.append(str(diag.get("selected_regime", "")))
        observation, reward, terminated, truncated, info = env.step(action_int)
        total_return += float(reward)
        final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
        if terminated or truncated:
            break
    steps = len(actions)
    if args.keep_step_summaries > 0 and len(step_summaries) > args.keep_step_summaries * 2:
        kept = step_summaries[: args.keep_step_summaries] + step_summaries[-args.keep_step_summaries :]
    else:
        kept = step_summaries
    return {
        "seed": int(seed),
        "steps": steps,
        "return": float(total_return),
        "success": bool(steps >= args.horizon),
        "failure": classify_final(final_raw, steps, args),
        "actions": actions,
        "forces": forces,
        "selected_regimes": selected,
        "step_summaries": kept,
        "final_raw_state": final_raw,
    }


def episode_step_summary(episode: dict[str, Any], step: int) -> dict[str, Any]:
    for item in episode["step_summaries"]:
        if int(item.get("step", -1)) == int(step):
            return item
    return {
        "step": int(step),
        "action": int(episode["actions"][step]),
        "force": float(episode["forces"][step]),
        "selected_regime": episode["selected_regimes"][step],
        "raw_before": [],
        "note": "step details omitted; increase --keep-step-summaries for interior divergences",
    }


def first_difference(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    limit = min(len(a["actions"]), len(b["actions"]))
    for idx in range(limit):
        if int(a["actions"][idx]) != int(b["actions"][idx]):
            return {
                "step": idx,
                "a_action": int(a["actions"][idx]),
                "b_action": int(b["actions"][idx]),
                "a_force": float(a["forces"][idx]),
                "b_force": float(b["forces"][idx]),
                "a_selected_regime": a["selected_regimes"][idx],
                "b_selected_regime": b["selected_regimes"][idx],
                "a": episode_step_summary(a, idx),
                "b": episode_step_summary(b, idx),
            }
    if len(a["actions"]) != len(b["actions"]):
        return {
            "step": limit,
            "reason": "episode_length_differs_before_action_diff",
            "a_steps": int(a["steps"]),
            "b_steps": int(b["steps"]),
        }
    return None


def compare_episode(model_a: str, model_b: str, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    episode_a = collect_episode(model_a, args, seed)
    episode_b = collect_episode(model_b, args, seed)
    limit = min(len(episode_a["actions"]), len(episode_b["actions"]))
    diff_count = sum(
        int(episode_a["actions"][idx]) != int(episode_b["actions"][idx])
        for idx in range(limit)
    )
    first_diff = first_difference(episode_a, episode_b)
    return {
        "seed": int(seed),
        "a_steps": int(episode_a["steps"]),
        "b_steps": int(episode_b["steps"]),
        "delta_steps": int(episode_b["steps"]) - int(episode_a["steps"]),
        "a_success": bool(episode_a["success"]),
        "b_success": bool(episode_b["success"]),
        "a_failure": episode_a["failure"],
        "b_failure": episode_b["failure"],
        "same_outcome": bool(
            episode_a["steps"] == episode_b["steps"]
            and episode_a["success"] == episode_b["success"]
            and episode_a["failure"] == episode_b["failure"]
        ),
        "action_compare_steps": int(limit),
        "action_diff_count": int(diff_count),
        "action_diff_fraction": float(diff_count / limit) if limit else 0.0,
        "first_action_diff": first_diff,
    }


def seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    seeds: list[int] = []
    for start in args.seed_starts:
        seeds.extend(int(start) + idx for idx in range(args.episodes_per_start))
    return seeds


def summarize_comparison(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    a_steps = [float(row["a_steps"]) for row in rows]
    b_steps = [float(row["b_steps"]) for row in rows]
    changed = [row for row in rows if not row["same_outcome"] or row["action_diff_count"] > 0]
    success_gains = [row for row in rows if (not bool(row["a_success"])) and bool(row["b_success"])]
    success_losses = [row for row in rows if bool(row["a_success"]) and (not bool(row["b_success"]))]
    first_diff_steps = [
        float(row["first_action_diff"]["step"])
        for row in rows
        if row.get("first_action_diff") and "step" in row["first_action_diff"]
    ]
    return {
        "a": summarize_steps(a_steps, horizon),
        "b": summarize_steps(b_steps, horizon),
        "episodes": len(rows),
        "mean_delta_steps": float(np.mean([row["delta_steps"] for row in rows])) if rows else 0.0,
        "changed_seed_count": len(changed),
        "action_changed_seed_count": sum(1 for row in rows if row["action_diff_count"] > 0),
        "success_gain_count": len(success_gains),
        "success_loss_count": len(success_losses),
        "first_diff_step_mean": float(np.mean(first_diff_steps)) if first_diff_steps else None,
        "first_diff_step_median": float(np.median(first_diff_steps)) if first_diff_steps else None,
        "a_failure_counts": dict(Counter(row["a_failure"] for row in rows)),
        "b_failure_counts": dict(Counter(row["b_failure"] for row in rows)),
        "success_gain_seeds": [int(row["seed"]) for row in success_gains],
        "success_loss_seeds": [int(row["seed"]) for row in success_losses],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    labels = result["labels"]
    summary = result["summary"]
    lines = [
        "# Policy Action Difference Report",
        "",
        f"A: `{labels['a']}`",
        f"B: `{labels['b']}`",
        f"Episodes: `{summary['episodes']}`",
        f"Changed seeds: `{summary['changed_seed_count']}`",
        f"Action-changed seeds: `{summary['action_changed_seed_count']}`",
        f"Success gains B over A: `{summary['success_gain_count']}`",
        f"Success losses B vs A: `{summary['success_loss_count']}`",
        "",
        "| model | mean | p10 | success | max |",
        "|---|---:|---:|---:|---:|",
        f"| {labels['a']} | {summary['a']['mean_survival']:.1f} | {summary['a']['p10_survival']:.1f} | {summary['a']['success_rate']:.3f} | {summary['a']['max_survival']:.0f} |",
        f"| {labels['b']} | {summary['b']['mean_survival']:.1f} | {summary['b']['p10_survival']:.1f} | {summary['b']['success_rate']:.3f} | {summary['b']['max_survival']:.0f} |",
        "",
        "## Outcome Deltas",
        "",
        f"Mean delta steps (B - A): `{summary['mean_delta_steps']:.2f}`",
        f"First action diff median step: `{summary['first_diff_step_median']}`",
        f"Success gain seeds: `{summary['success_gain_seeds']}`",
        f"Success loss seeds: `{summary['success_loss_seeds']}`",
        "",
        "## Changed Seeds",
        "",
        "| seed | A steps | B steps | delta | A failure | B failure | first diff | diff frac |",
        "|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    changed_rows = [row for row in result["per_seed"] if (not row["same_outcome"] or row["action_diff_count"] > 0)]
    for row in sorted(changed_rows, key=lambda item: item["seed"]):
        first = row.get("first_action_diff") or {}
        first_step = first.get("step", "")
        lines.append(
            f"| {row['seed']} | {row['a_steps']} | {row['b_steps']} | {row['delta_steps']} | "
            f"{row['a_failure']} | {row['b_failure']} | {first_step} | {row['action_diff_fraction']:.3f} |"
        )
    if result.get("examples"):
        lines.extend(["", "## First-Difference Examples", ""])
        for row in result["examples"]:
            first = row.get("first_action_diff") or {}
            lines.extend(
                [
                    f"### Seed {row['seed']}",
                    "",
                    f"Steps: `{row['a_steps']}` -> `{row['b_steps']}`; first diff step `{first.get('step')}`.",
                    f"Actions: `{first.get('a_action')}` -> `{first.get('b_action')}`; forces `{first.get('a_force')}` -> `{first.get('b_force')}`.",
                    f"Regimes: `{first.get('a_selected_regime')}` -> `{first.get('b_selected_regime')}`.",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seed_list(args)
    rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds, start=1):
        rows.append(compare_episode(args.model_a, args.model_b, args, seed))
        if idx % max(1, args.save_every) == 0:
            partial = {"completed": idx, "total": len(seeds), "per_seed": rows}
            (out / "partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    changed_rows = [row for row in rows if (not row["same_outcome"] or row["action_diff_count"] > 0)]
    examples = sorted(
        changed_rows,
        key=lambda row: (
            0 if row["b_success"] and not row["a_success"] else 1,
            -abs(int(row["delta_steps"])),
            row["seed"],
        ),
    )[: args.keep_examples]
    result = {
        "status": "completed",
        "labels": {"a": args.label_a, "b": args.label_b},
        "models": {"a": args.model_a, "b": args.model_b},
        "env": env_payload(args),
        "controller": {
            "selection_mode": args.selection_mode,
            "policy_terminal_blend": args.policy_terminal_blend,
            "policy_terminal_scope": args.policy_terminal_scope,
            "policy_observation_mode": args.policy_observation_mode,
            "frame_stack": args.frame_stack,
            "policy_terminal_recurrent": args.policy_terminal_recurrent,
            "policy_terminal_normalizer_path": args.policy_terminal_normalizer_path,
        },
        "seeds": seeds,
        "summary": summarize_comparison(rows, args.horizon),
        "per_seed": rows,
        "examples": examples,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "action_differences.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "action_differences.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two ReCoN policy-terminal checkpoints action-by-action on matched seeds.")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default="model_a")
    parser.add_argument("--label-b", default="model_b")
    parser.add_argument("--out", default="reports/policy_action_differences")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", default="serial_lagrange")
    parser.add_argument("--action-mode", default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", default="stabilize_chain")
    parser.add_argument("--policy-observation-mode", default="normalized_raw")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--policy-terminal-recurrent", action="store_true")
    parser.add_argument("--policy-terminal-normalizer-path", default="")
    parser.add_argument("--seed-starts", type=int, nargs="*", default=[900000])
    parser.add_argument("--episodes-per-start", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--keep-step-summaries", type=int, default=32)
    parser.add_argument("--keep-examples", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def main() -> None:
    result = run_comparison(build_parser().parse_args())
    summary = result["summary"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": summary["episodes"],
                "a_success": summary["a"]["success_rate"],
                "b_success": summary["b"]["success_rate"],
                "changed_seed_count": summary["changed_seed_count"],
                "success_gain_count": summary["success_gain_count"],
                "success_loss_count": summary["success_loss_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
