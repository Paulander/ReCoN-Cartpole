from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig
from recon_cartpole.training.ablations import summarize_steps


def env_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_poles": args.n_poles,
        "horizon": args.horizon,
        "dt": args.dt,
        "dynamics_mode": args.dynamics_mode,
        "action_mode": "discrete",
        "discrete_action_bins": args.discrete_action_bins,
        "force_mag": args.force_mag,
        "initial_angle_range": args.initial_angle_range,
        "force_noise": args.force_noise,
        "link_coupling": args.link_coupling,
    }


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(CartPoleNConfig(**env_payload(args)))


def terminal_config(args: argparse.Namespace) -> MinGRUTerminalConfig:
    return MinGRUTerminalConfig(
        enabled=True,
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        include_motif_score=args.include_motif_score,
        motif_model_path=args.motif_model_path,
        motif_score_scale=args.motif_score_scale,
        blend=args.blend,
        scope=args.scope,
        confidence_floor=args.confidence_floor,
        passthrough_enabled=args.passthrough_enabled,
        passthrough_confidence_floor=args.passthrough_confidence_floor,
        passthrough_logit_margin_floor=args.passthrough_logit_margin_floor,
        checkpoint_path=args.checkpoint_path,
    )


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
            reset_bandit_each_episode=False,
            mingru_terminal=terminal_config(args),
        )
    )


def make_pure_terminal(args: argparse.Namespace) -> MinGRUTerminal:
    return MinGRUTerminal(
        args.n_poles,
        args.force_mag,
        args.discrete_action_bins,
        terminal_config(args),
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


def logit_margin(logits: list[float]) -> float:
    finite = [float(value) for value in logits if np.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0
    top_two = sorted(finite, reverse=True)[:2]
    return float(top_two[0] - top_two[1])


def trim_prediction(prediction: Any, force: float, action: int) -> dict[str, Any]:
    logits = [float(value) for value in getattr(prediction, "logits", [])]
    return {
        "action": int(action),
        "force": float(force),
        "confidence": float(getattr(prediction, "confidence", 0.0)),
        "failure_probability": float(getattr(prediction, "failure_probability", 0.0)),
        "value": float(getattr(prediction, "value", 0.0)),
        "hidden_norm": float(getattr(prediction, "hidden_norm", 0.0)),
        "sequence_length": int(getattr(prediction, "sequence_length", 0)),
        "logit_margin": logit_margin(logits),
        "logits": logits,
        "valid": bool(getattr(prediction, "valid", False)),
        "reason": str(getattr(prediction, "reason", "")),
    }


def compact_proposal(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_node": item.get("source_node", ""),
        "regime": item.get("regime", ""),
        "force": item.get("force"),
        "confidence": item.get("confidence"),
        "urgency": item.get("urgency"),
        "score": item.get("score"),
        "weighted_score": item.get("weighted_score", item.get("score")),
        "selected": item.get("selected"),
        "reason": item.get("reason", ""),
    }


def trim_recon_diagnostics(
    diagnostics: dict[str, Any], action: int, step: int, args: argparse.Namespace
) -> dict[str, Any]:
    proposal = diagnostics.get("proposal", {}) or {}
    terminal = diagnostics.get("mingru_terminal", {}) or {}
    terminal_force = terminal.get("force")
    terminal_action = None
    if terminal_force is not None:
        terminal_action = int(
            action_from_force(
                float(terminal_force),
                "discrete",
                args.force_mag,
                args.discrete_action_bins,
            )
        )
    return {
        "step": int(step),
        "action": int(action),
        "force": float(diagnostics.get("force", 0.0)),
        "selected_regime": diagnostics.get("selected_regime", ""),
        "proposal": compact_proposal(proposal),
        "proposals": [compact_proposal(item) for item in diagnostics.get("proposals", [])],
        "suppressed_proposals": [
            compact_proposal(item) for item in diagnostics.get("suppressed_proposals", [])
        ],
        "mingru_terminal": {
            "available": bool(terminal.get("available", False)),
            "applied": bool(terminal.get("applied", False)),
            "action": terminal_action,
            "force": terminal_force,
            "terminal_force": terminal.get("terminal_force"),
            "proposal_force": terminal.get("proposal_force"),
            "base_force": terminal.get("base_force"),
            "confidence": terminal.get("confidence"),
            "confidence_floor": terminal.get("confidence_floor"),
            "failure_probability": terminal.get("failure_probability"),
            "value": terminal.get("value"),
            "hidden_norm": terminal.get("hidden_norm"),
            "logit_margin": logit_margin([float(value) for value in terminal.get("logits", [])]),
            "scope": terminal.get("scope"),
            "applied_regime": terminal.get("applied_regime"),
            "applied_regimes": terminal.get("applied_regimes", []),
            "reason": terminal.get("reason", ""),
        },
        "mingru_passthrough": diagnostics.get("mingru_passthrough", {}) or {},
    }


def collect_pure_episode(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env = make_env(args)
    terminal = make_pure_terminal(args)
    observation, info = env.reset(seed=seed)
    terminal.reset()
    total_return = 0.0
    actions: list[int] = []
    forces: list[float] = []
    steps: list[dict[str, Any]] = []
    final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
    for step in range(args.horizon):
        raw_before = np.asarray(info.get("raw_state", []), dtype=float).copy()
        prediction = terminal.predict(observation, raw_before, {})
        force = 0.0 if prediction.force is None else float(prediction.force)
        action = int(action_from_force(force, "discrete", args.force_mag, args.discrete_action_bins))
        actions.append(action)
        forces.append(force)
        steps.append(
            {
                "step": int(step),
                "raw_before": raw_before.tolist(),
                "prediction": trim_prediction(prediction, force, action),
            }
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
        if terminated or truncated:
            break
    return {
        "seed": int(seed),
        "steps": len(actions),
        "return": float(total_return),
        "success": bool(len(actions) >= args.horizon),
        "failure": classify_final(final_raw, len(actions), args),
        "actions": actions,
        "forces": forces,
        "step_details": steps,
        "final_raw_state": final_raw,
    }


def collect_recon_episode(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env = make_env(args)
    controller = make_recon_controller(args)
    observation, info = env.reset(seed=seed)
    controller.start_episode()
    total_return = 0.0
    actions: list[int] = []
    forces: list[float] = []
    steps: list[dict[str, Any]] = []
    final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
    for step in range(args.horizon):
        raw_before = np.asarray(info.get("raw_state", []), dtype=float).copy()
        action_raw, diagnostics = controller.act(observation, raw_before)
        action = int(np.asarray(action_raw).reshape(-1)[0])
        actions.append(action)
        forces.append(float(diagnostics.get("force", 0.0)))
        steps.append(
            {
                "step": int(step),
                "raw_before": raw_before.tolist(),
                "diagnostics": trim_recon_diagnostics(diagnostics, action, step, args),
            }
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        final_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
        if terminated or truncated:
            break
    return {
        "seed": int(seed),
        "steps": len(actions),
        "return": float(total_return),
        "success": bool(len(actions) >= args.horizon),
        "failure": classify_final(final_raw, len(actions), args),
        "actions": actions,
        "forces": forces,
        "step_details": steps,
        "final_raw_state": final_raw,
    }


def step_detail(episode: dict[str, Any], step: int) -> dict[str, Any]:
    details = episode.get("step_details", [])
    if 0 <= step < len(details):
        return details[step]
    return {"step": int(step), "note": "step outside episode"}


def first_difference(pure: dict[str, Any], recon: dict[str, Any]) -> dict[str, Any] | None:
    limit = min(len(pure["actions"]), len(recon["actions"]))
    for step in range(limit):
        if int(pure["actions"][step]) != int(recon["actions"][step]):
            return {
                "step": int(step),
                "pure_action": int(pure["actions"][step]),
                "recon_action": int(recon["actions"][step]),
                "pure_force": float(pure["forces"][step]),
                "recon_force": float(recon["forces"][step]),
                "pure": step_detail(pure, step),
                "recon": step_detail(recon, step),
            }
    if len(pure["actions"]) != len(recon["actions"]):
        return {
            "step": int(limit),
            "reason": "episode_length_differs_before_action_diff",
            "pure_steps": int(pure["steps"]),
            "recon_steps": int(recon["steps"]),
        }
    return None


def compare_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    pure = collect_pure_episode(args, seed)
    recon = collect_recon_episode(args, seed)
    limit = min(len(pure["actions"]), len(recon["actions"]))
    diff_count = sum(int(pure["actions"][idx]) != int(recon["actions"][idx]) for idx in range(limit))
    force_deltas = [
        abs(float(pure["forces"][idx]) - float(recon["forces"][idx]))
        for idx in range(limit)
    ]
    first_diff = first_difference(pure, recon)
    return {
        "seed": int(seed),
        "pure_steps": int(pure["steps"]),
        "recon_steps": int(recon["steps"]),
        "delta_steps": int(recon["steps"]) - int(pure["steps"]),
        "pure_success": bool(pure["success"]),
        "recon_success": bool(recon["success"]),
        "pure_failure": pure["failure"],
        "recon_failure": recon["failure"],
        "same_outcome": bool(
            pure["steps"] == recon["steps"]
            and pure["success"] == recon["success"]
            and pure["failure"] == recon["failure"]
        ),
        "action_compare_steps": int(limit),
        "action_diff_count": int(diff_count),
        "action_diff_fraction": float(diff_count / limit) if limit else 0.0,
        "mean_abs_force_delta": float(np.mean(force_deltas)) if force_deltas else 0.0,
        "first_action_diff": first_diff,
    }


def _seed_from_item(item: Any) -> int:
    if isinstance(item, dict):
        return int(item["seed"])
    return int(item)


def read_seed_file(path: str) -> list[int]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("hard_seeds", "seeds", "tail_seeds"):
            if isinstance(payload.get(key), list):
                return [_seed_from_item(item) for item in payload[key]]
    if isinstance(payload, list):
        return [_seed_from_item(item) for item in payload]
    seeds: list[int] = []
    for item in raw.replace(",", "\n").splitlines():
        value = item.strip()
        if value:
            seeds.append(int(value))
    return seeds


def seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    if args.seed_file:
        seeds = read_seed_file(args.seed_file)
        return seeds[: args.max_seeds] if args.max_seeds > 0 else seeds
    seeds: list[int] = []
    for start in args.seed_starts:
        seeds.extend(int(start) + idx for idx in range(args.episodes_per_start))
    return seeds


def summarize(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    pure_steps = [float(row["pure_steps"]) for row in rows]
    recon_steps = [float(row["recon_steps"]) for row in rows]
    changed = [row for row in rows if not row["same_outcome"] or row["action_diff_count"] > 0]
    gains = [row for row in rows if (not row["pure_success"]) and row["recon_success"]]
    losses = [row for row in rows if row["pure_success"] and (not row["recon_success"])]
    first_steps = [
        float(row["first_action_diff"]["step"])
        for row in rows
        if row.get("first_action_diff") and "step" in row["first_action_diff"]
    ]
    return {
        "pure_mingru_policy": summarize_steps(pure_steps, horizon),
        "recon_mingru_terminal": summarize_steps(recon_steps, horizon),
        "episodes": len(rows),
        "mean_delta_steps": float(np.mean([row["delta_steps"] for row in rows])) if rows else 0.0,
        "changed_seed_count": len(changed),
        "action_changed_seed_count": sum(1 for row in rows if row["action_diff_count"] > 0),
        "success_gain_count": len(gains),
        "success_loss_count": len(losses),
        "success_gain_seeds": [int(row["seed"]) for row in gains],
        "success_loss_seeds": [int(row["seed"]) for row in losses],
        "pure_failure_counts": dict(Counter(row["pure_failure"] for row in rows)),
        "recon_failure_counts": dict(Counter(row["recon_failure"] for row in rows)),
        "first_diff_step_mean": float(np.mean(first_steps)) if first_steps else None,
        "first_diff_step_median": float(np.median(first_steps)) if first_steps else None,
        "mean_abs_force_delta": float(np.mean([row["mean_abs_force_delta"] for row in rows])) if rows else 0.0,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    pure = summary["pure_mingru_policy"]
    recon = summary["recon_mingru_terminal"]
    lines = [
        "# minGRU Routing Comparison",
        "",
        f"Checkpoint: `{result['checkpoint_path']}`",
        f"Episodes: `{summary['episodes']}`",
        f"Changed seeds: `{summary['changed_seed_count']}`",
        f"Action-changed seeds: `{summary['action_changed_seed_count']}`",
        f"Success gains ReCoN over pure: `{summary['success_gain_count']}`",
        f"Success losses ReCoN vs pure: `{summary['success_loss_count']}`",
        "",
        "| mode | mean | p10 | success | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| pure_minGRU_policy | {pure['mean_survival']:.1f} | {pure['p10_survival']:.1f} | {pure['success_rate']:.3f} | {pure['max_survival']:.0f} |",
        f"| ReCoN_routed_minGRU | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['success_rate']:.3f} | {recon['max_survival']:.0f} |",
        "",
        "## Routing Delta",
        "",
        f"Mean delta steps (ReCoN - pure): `{summary['mean_delta_steps']:.2f}`",
        f"Mean absolute force delta: `{summary['mean_abs_force_delta']:.3f}`",
        f"First action diff median step: `{summary['first_diff_step_median']}`",
        f"Success gain seeds: `{summary['success_gain_seeds']}`",
        f"Success loss seeds: `{summary['success_loss_seeds']}`",
        "",
        "## Changed Seeds",
        "",
        "| seed | pure steps | ReCoN steps | delta | pure failure | ReCoN failure | first diff | diff frac |",
        "| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    changed = [row for row in result["per_seed"] if not row["same_outcome"] or row["action_diff_count"] > 0]
    for row in sorted(changed, key=lambda item: (item["seed"])):
        first = row.get("first_action_diff") or {}
        lines.append(
            f"| {row['seed']} | {row['pure_steps']} | {row['recon_steps']} | {row['delta_steps']} | "
            f"{row['pure_failure']} | {row['recon_failure']} | {first.get('step', '')} | {row['action_diff_fraction']:.3f} |"
        )
    if result.get("examples"):
        lines.extend(["", "## First-Difference Examples", ""])
        for row in result["examples"]:
            first = row.get("first_action_diff") or {}
            recon_diag = ((first.get("recon") or {}).get("diagnostics") or {})
            pure_pred = ((first.get("pure") or {}).get("prediction") or {})
            proposal = recon_diag.get("proposal", {})
            mingru = recon_diag.get("mingru_terminal", {})
            lines.extend(
                [
                    f"### Seed {row['seed']}",
                    "",
                    f"Steps: `{row['pure_steps']}` pure -> `{row['recon_steps']}` ReCoN.",
                    f"First diff step: `{first.get('step')}`; action `{first.get('pure_action')}` pure -> `{first.get('recon_action')}` ReCoN.",
                    f"Pure force/confidence/margin: `{pure_pred.get('force')}` / `{pure_pred.get('confidence')}` / `{pure_pred.get('logit_margin')}`.",
                    f"ReCoN force/selected/proposal: `{recon_diag.get('force')}` / `{recon_diag.get('selected_regime')}` / `{proposal.get('source_node')}`.",
                    f"ReCoN mingru applied/action/confidence/margin: `{mingru.get('applied')}` / `{mingru.get('action')}` / `{mingru.get('confidence')}` / `{mingru.get('logit_margin')}`.",
                    f"Proposal reason: `{proposal.get('reason')}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seed_list(args)
    rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds, start=1):
        rows.append(compare_seed(args, seed))
        if idx % max(1, args.save_every) == 0:
            (out / "partial.json").write_text(
                json.dumps({"completed": idx, "total": len(seeds), "per_seed": rows}, indent=2),
                encoding="utf-8",
            )
    changed = [row for row in rows if not row["same_outcome"] or row["action_diff_count"] > 0]
    examples = sorted(
        changed,
        key=lambda row: (
            0 if row["pure_success"] and not row["recon_success"] else 1,
            -abs(int(row["delta_steps"])),
            row["seed"],
        ),
    )[: args.keep_examples]
    result = {
        "status": "completed",
        "checkpoint_path": args.checkpoint_path,
        "env": env_payload(args),
        "terminal": {
            "hidden_size": args.hidden_size,
            "sequence_length": args.sequence_length,
            "observation_mode": args.observation_mode,
            "include_prev_force": args.include_prev_force,
            "include_context": args.include_context,
            "include_motif_score": args.include_motif_score,
            "motif_model_path": args.motif_model_path,
            "motif_score_scale": args.motif_score_scale,
            "blend": args.blend,
            "scope": args.scope,
            "confidence_floor": args.confidence_floor,
            "passthrough_enabled": args.passthrough_enabled,
            "passthrough_confidence_floor": args.passthrough_confidence_floor,
            "passthrough_logit_margin_floor": args.passthrough_logit_margin_floor,
        },
        "controller": {"selection_mode": args.selection_mode},
        "seeds": seeds,
        "summary": summarize(rows, args.horizon),
        "per_seed": rows,
        "examples": examples,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "mingru_routing_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "mingru_routing_comparison.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare pure minGRU control against ReCoN-routed minGRU control.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out", default="reports/mingru_routing_comparison")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", default="hard_select")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--observation-mode", default="normalized_raw4_subchains_prev_force")
    parser.add_argument("--include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-motif-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motif-model-path", default="reports/n4_subchain_motif_diag_recon_20260612_seed2420k/prototype_model.json")
    parser.add_argument("--motif-score-scale", type=float, default=10.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--scope", default="stabilize_chain")
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-logit-margin-floor", type=float, default=0.0)
    parser.add_argument("--seed-starts", type=int, nargs="*", default=[1900000, 2000000, 2100000, 2200000])
    parser.add_argument("--episodes-per-start", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--seed-file", default="")
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument("--keep-examples", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    summary = result["summary"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": summary["episodes"],
                "pure_success": summary["pure_mingru_policy"]["success_rate"],
                "recon_success": summary["recon_mingru_terminal"]["success_rate"],
                "changed_seed_count": summary["changed_seed_count"],
                "action_changed_seed_count": summary["action_changed_seed_count"],
                "success_gain_count": summary["success_gain_count"],
                "success_loss_count": summary["success_loss_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
