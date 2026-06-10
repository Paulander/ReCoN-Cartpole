from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from recon_cartpole.control.rewards import reward_tick
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RescueConfig, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_n4_pole1_robustness import (  # noqa: E402
    FEEDFORWARD_CHECKPOINT,
    classify,
    env_payload,
    pole1_diagnostics,
)

BASELINE_EVAL = "reports/n4_pole1_robustness_20260610_171635/300_seed_eval.json"
BASELINE_MODE = "recon_feedforward_terminal_frozen"


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(CartPoleNConfig(**env_payload(args)))


def controller_for(args: argparse.Namespace, rescue: RescueConfig | None = None) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_feedforward_terminal_frozen",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode="hard_select",
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.feedforward_checkpoint,
            policy_terminal_blend=1.0,
            policy_terminal_scope="stabilize_chain",
            policy_terminal_observation_mode=args.observation_mode,
            rescue=rescue or RescueConfig(),
        )
    )


def candidate_configs() -> dict[str, RescueConfig]:
    return {
        "late_episode_conservative_mode": RescueConfig(
            enabled=True,
            late_episode_conservative_mode=True,
            late_start_step=400,
            late_rail_guard=2.05,
        ),
        "terminal_force_passthrough_high_confidence": RescueConfig(
            enabled=True,
            terminal_force_passthrough_high_confidence=True,
            passthrough_start_step=400,
            passthrough_angle_threshold=0.14,
            passthrough_velocity_pressure=0.65,
        ),
        "anti_oscillation_damper": RescueConfig(
            enabled=True,
            terminal_force_passthrough_high_confidence=True,
            anti_oscillation_damper=True,
            passthrough_start_step=400,
            passthrough_angle_threshold=0.16,
            oscillation_window=12,
            oscillation_flip_threshold=8,
        ),
        "rail_vs_pole_priority_gate": RescueConfig(
            enabled=True,
            rail_vs_pole_priority_gate=True,
            rail_imminent_x=2.10,
        ),
        "pole1_emergency_guard_v2": RescueConfig(
            enabled=True,
            pole1_emergency_guard_v2=True,
            pole1_angle_threshold=0.18,
            pole1_velocity_threshold=0.75,
            pole1_force_blend=0.75,
        ),
        "late_terminal_passthrough_combo": RescueConfig(
            enabled=True,
            late_episode_conservative_mode=True,
            terminal_force_passthrough_high_confidence=True,
            late_start_step=400,
            passthrough_start_step=400,
            passthrough_angle_threshold=0.14,
        ),
        "rail_gate_terminal_passthrough_combo": RescueConfig(
            enabled=True,
            rail_vs_pole_priority_gate=True,
            terminal_force_passthrough_high_confidence=True,
            rail_imminent_x=2.10,
            passthrough_start_step=400,
            passthrough_angle_threshold=0.14,
        ),
    }


def rollout_candidate(
    args: argparse.Namespace,
    controller: ReConCartPoleController,
    seed: int,
    trace: bool = False,
) -> dict[str, Any]:
    env = make_env(args)
    observation, info = env.reset(seed=seed)
    initial_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
    controller.start_episode()
    total_return = 0.0
    traces: list[dict[str, Any]] = []
    last_reward_tick = 0.0
    final_raw: list[float] = initial_raw
    selected_counts: Counter[str] = Counter()
    rescue_events: Counter[str] = Counter()
    policy_diffs: list[float] = []
    forces: list[float] = []
    for step in range(args.horizon):
        controller.observe_reward(last_reward_tick)
        raw_before = info.get("raw_state")
        action, diagnostics = controller.act(observation, raw_before)
        before_obs = np.asarray(observation, dtype=float)
        next_observation, env_reward, terminated, truncated, info = env.step(action)
        total_return += float(env_reward)
        next_raw = np.asarray(info.get("raw_state", []), dtype=float).tolist()
        final_raw = next_raw
        force = float(diagnostics.get("force", 0.0))
        forces.append(force)
        selected = str(diagnostics.get("selected_regime", ""))
        if selected:
            selected_counts[selected] += 1
        for event in diagnostics.get("rescue", {}).get("events", []) or []:
            rescue_events[str(event)] += 1
        policy = diagnostics.get("policy_terminal", {}) or {}
        if policy.get("policy_force") is not None:
            policy_diffs.append(abs(float(policy.get("policy_force")) - force))
        last_reward_tick = reward_tick(
            before_obs,
            next_observation,
            raw_before,
            info.get("raw_state"),
            controller.config.n_poles,
            terminated,
        )
        if trace:
            traces.append(
                {
                    "step": step,
                    "raw_state": next_raw,
                    "action": int(np.asarray(action).reshape(-1)[0]),
                    "force": force,
                    "env_reward": float(env_reward),
                    "reward_tick": float(last_reward_tick),
                    "return_so_far": float(total_return),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "selected_regime": selected,
                    "proposal": diagnostics.get("proposal", {}),
                    "proposals": diagnostics.get("proposals", []),
                    "suppressed_proposals": diagnostics.get("suppressed_proposals", []),
                    "policy_terminal": policy,
                    "rescue": diagnostics.get("rescue", {}),
                }
            )
        observation = next_observation
        if terminated or truncated:
            break
    controller.observe_reward(last_reward_tick)
    steps = step + 1
    failure = classify(final_raw, steps, args)
    force_flips = sum(
        1
        for a, b in zip(forces, forces[1:])
        if abs(a) > 1e-6 and abs(b) > 1e-6 and np.sign(a) != np.sign(b)
    )
    row = {
        "seed": seed,
        "steps": steps,
        "return": float(total_return),
        "success": steps >= args.horizon,
        "failure": failure,
        "initial_raw_state": initial_raw,
        "final_raw_state": final_raw,
        "selected_regime_counts": dict(selected_counts),
        "rescue_event_counts": dict(rescue_events),
        "policy_final_force_abs_diff_mean": float(np.mean(policy_diffs)) if policy_diffs else 0.0,
        "force_sign_flips": int(force_flips),
    }
    if trace:
        row["trace"] = traces
        row.update(pole1_diagnostics(traces, args))
        row.update(tail_diagnostics(traces, args))
    return row


def tail_diagnostics(trace: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not trace:
        return {}
    tail = trace[-100:]
    raws = [np.asarray(step.get("raw_state", []), dtype=float) for step in tail]
    raws = [raw for raw in raws if raw.size >= 2 + 2 * args.n_poles]
    forces = np.asarray([float(step.get("force", 0.0) or 0.0) for step in tail], dtype=float)
    if not raws:
        return {}
    theta = np.asarray([raw[2 : 2 + args.n_poles] for raw in raws], dtype=float)
    theta_dot = np.asarray([raw[2 + args.n_poles : 2 + 2 * args.n_poles] for raw in raws], dtype=float)
    x = np.asarray([raw[0] for raw in raws], dtype=float)
    flips = sum(
        1
        for a, b in zip(forces, forces[1:])
        if abs(a) > 1e-6 and abs(b) > 1e-6 and np.sign(a) != np.sign(b)
    )
    return {
        "last100_pole1_max_abs_angle": float(np.max(np.abs(theta[:, 1]))) if args.n_poles > 1 else 0.0,
        "last100_pole1_max_abs_velocity": float(np.max(np.abs(theta_dot[:, 1]))) if args.n_poles > 1 else 0.0,
        "last100_force_oscillation_score": float(flips / max(1, len(forces) - 1)),
        "last100_max_rail_abs_x": float(np.max(np.abs(x))),
        "last100_pole_max_abs_angles": np.max(np.abs(theta), axis=0).tolist(),
        "last100_pole_max_abs_velocities": np.max(np.abs(theta_dot), axis=0).tolist(),
    }


def evaluate_candidate(
    name: str,
    rescue: RescueConfig,
    seeds: list[int],
    args: argparse.Namespace,
    trace: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    controller = controller_for(args, rescue)
    per_seed = []
    for idx, seed in enumerate(seeds, 1):
        per_seed.append(rollout_candidate(args, controller, seed, trace=trace))
        if idx % args.progress_every == 0:
            print(f"[n4-final-gap] {name}: {idx}/{len(seeds)}", flush=True)
    steps = [float(item["steps"]) for item in per_seed]
    summary = summarize_steps(steps, args.horizon)
    failures = Counter(item["failure"] for item in per_seed)
    result = {
        "candidate": name,
        "episodes": len(seeds),
        "mean_survival": summary["mean_survival"],
        "median_survival": float(np.median(steps)) if steps else 0.0,
        "p10_survival": summary["p10_survival"],
        "p90_survival": float(np.percentile(steps, 90)) if steps else 0.0,
        "success_at_500": summary["success_rate"],
        "max_survival": summary["max_survival"],
        "failure_distribution": dict(failures),
        "wall_clock_seconds": time.perf_counter() - started,
        "rescue_config": asdict(rescue),
        "per_seed": per_seed,
    }
    return result


def load_baseline(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.baseline_eval)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        baseline = data.get(BASELINE_MODE, data)
        baseline = dict(baseline)
        baseline["candidate"] = "baseline_best_frozen"
        baseline["success_at_500"] = baseline.get("success_rate", baseline.get("success_at_500", 0.0))
        return baseline
    seeds = [args.validation_seed_start + idx for idx in range(args.validation_episodes)]
    print("[n4-final-gap] baseline JSON missing; re-evaluating baseline", flush=True)
    return evaluate_candidate("baseline_best_frozen", RescueConfig(), seeds, args, trace=False)


def choose_dev_seeds(baseline: dict[str, Any], args: argparse.Namespace) -> list[int]:
    rows = list(baseline["per_seed"])
    late_failures = [item for item in rows if not item["success"] and item["steps"] >= args.late_failure_step]
    early_failures = [item for item in rows if not item["success"] and item["steps"] < args.late_failure_step]
    successes = [item for item in rows if item["success"]]
    seeds = [
        item["seed"]
        for item in sorted(late_failures, key=lambda x: x["steps"], reverse=True)[: args.dev_late_failures]
    ]
    seeds.extend(item["seed"] for item in sorted(early_failures, key=lambda x: x["steps"], reverse=True)[: args.dev_early_failures])
    seeds.extend(item["seed"] for item in successes[: args.dev_successes])
    seen = set()
    deduped = []
    for seed in seeds:
        if seed not in seen:
            deduped.append(int(seed))
            seen.add(seed)
    return deduped


def compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    base_by_seed = {int(item["seed"]): item for item in baseline["per_seed"]}
    changed = []
    fail_to_success = []
    success_to_fail = []
    improved = []
    worsened = []
    for item in candidate["per_seed"]:
        seed = int(item["seed"])
        base = base_by_seed.get(seed)
        if not base:
            continue
        delta = int(item["steps"]) - int(base["steps"])
        if delta != 0 or bool(item["success"]) != bool(base["success"]):
            changed.append({"seed": seed, "baseline": base, "candidate": item, "delta_steps": delta})
        if not base["success"] and item["success"]:
            fail_to_success.append(seed)
        if base["success"] and not item["success"]:
            success_to_fail.append(seed)
        if delta > 0:
            improved.append(seed)
        elif delta < 0:
            worsened.append(seed)
    return {
        "changed_count": len(changed),
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "improved_seeds": improved,
        "worsened_seeds": worsened,
        "net_success_change": len(fail_to_success) - len(success_to_fail),
        "changed": changed,
    }


def row_for(candidate: dict[str, Any], baseline: dict[str, Any], phase: str) -> dict[str, Any]:
    comparison = compare_to_baseline(candidate, baseline)
    return {
        "phase": phase,
        "candidate": candidate["candidate"],
        "episodes": candidate["episodes"],
        "mean_survival": candidate["mean_survival"],
        "p10_survival": candidate["p10_survival"],
        "success_at_500": candidate["success_at_500"],
        "max_survival": candidate["max_survival"],
        "changed_count": comparison["changed_count"],
        "fail_to_success": len(comparison["fail_to_success"]),
        "success_to_fail": len(comparison["success_to_fail"]),
        "net_success_change": comparison["net_success_change"],
        "failure_distribution": candidate["failure_distribution"],
    }


def analyze_baseline_failures(baseline: dict[str, Any], args: argparse.Namespace, out: Path) -> list[dict[str, Any]]:
    pure = {}
    full = json.loads(Path(args.baseline_eval).read_text(encoding="utf-8")) if Path(args.baseline_eval).exists() else {}
    if "pure_feedforward_policy_terminal" in full:
        pure = {int(item["seed"]): item for item in full["pure_feedforward_policy_terminal"]["per_seed"]}
    failures = [item for item in baseline["per_seed"] if not item["success"]]
    ranked = sorted(failures, key=lambda item: item["steps"], reverse=True)
    details = []
    for idx, item in enumerate(ranked[: args.failure_trace_limit], 1):
        seed = int(item["seed"])
        trace_row = rollout_candidate(args, controller_for(args), seed, trace=True)
        trace_row.pop("trace")
        pure_row = pure.get(seed, {})
        delta_vs_pure = int(item["steps"]) - int(pure_row.get("steps", item["steps"]))
        if delta_vs_pure > 0:
            arbitration_effect = "helped"
        elif delta_vs_pure < 0:
            arbitration_effect = "hurt"
        else:
            arbitration_effect = "neutral"
        detail = {
            **{k: v for k, v in trace_row.items() if k != "initial_raw_state"},
            "baseline_steps": item["steps"],
            "baseline_failure": item["failure"],
            "pure_ppo_steps": pure_row.get("steps"),
            "pure_ppo_failure": pure_row.get("failure"),
            "recon_arbitration_effect_vs_pure": arbitration_effect,
            "recon_minus_pure_steps": delta_vs_pure,
            "initial_raw_state": trace_row.get("initial_raw_state", []),
        }
        details.append(detail)
        if idx % args.progress_every == 0:
            print(f"[n4-final-gap] traced baseline failures: {idx}/{min(len(ranked), args.failure_trace_limit)}", flush=True)
    (out / "baseline_failure_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def write_candidate_tables(rows: list[dict[str, Any]], out: Path) -> None:
    (out / "candidate_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "phase",
        "candidate",
        "episodes",
        "mean_survival",
        "p10_survival",
        "success_at_500",
        "max_survival",
        "changed_count",
        "fail_to_success",
        "success_to_fail",
        "net_success_change",
    ]
    with (out / "candidate_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_changed_seed_analysis(best: dict[str, Any], baseline: dict[str, Any], out: Path) -> dict[str, Any]:
    comparison = compare_to_baseline(best, baseline)
    lines = ["# Changed Seed Analysis", ""]
    lines.append(f"Candidate: `{best['candidate']}`")
    lines.append(f"Changed seeds: `{comparison['changed_count']}`")
    lines.append(f"Fail -> success: `{len(comparison['fail_to_success'])}` {comparison['fail_to_success'][:50]}")
    lines.append(f"Success -> fail: `{len(comparison['success_to_fail'])}` {comparison['success_to_fail'][:50]}")
    lines.append("")
    lines.append("## Largest Improvements")
    for item in sorted(comparison["changed"], key=lambda x: x["delta_steps"], reverse=True)[:25]:
        lines.append(
            f"- `{item['seed']}`: {item['baseline']['steps']} -> {item['candidate']['steps']} "
            f"({item['delta_steps']:+d}), {item['baseline']['failure']} -> {item['candidate']['failure']}"
        )
    lines.append("")
    lines.append("## Largest Regressions")
    for item in sorted(comparison["changed"], key=lambda x: x["delta_steps"])[:25]:
        lines.append(
            f"- `{item['seed']}`: {item['baseline']['steps']} -> {item['candidate']['steps']} "
            f"({item['delta_steps']:+d}), {item['baseline']['failure']} -> {item['candidate']['failure']}"
        )
    (out / "changed_seed_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "changed_seed_analysis.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def write_rescue_patterns(details: list[dict[str, Any]], baseline: dict[str, Any], out: Path) -> None:
    failures = [item for item in baseline["per_seed"] if not item["success"]]
    by_failure = Counter(item["failure"] for item in failures)
    late = [item for item in failures if item["steps"] >= 450]
    very_late = [item for item in failures if item["steps"] >= 475]
    effect = Counter(item.get("recon_arbitration_effect_vs_pure", "unknown") for item in details)
    selected = Counter()
    for item in details:
        selected.update(item.get("selected_regime_counts", {}))
    lines = ["# Rescue Patterns", ""]
    lines.append(f"Baseline successes/failures: `{sum(item['success'] for item in baseline['per_seed'])}` / `{len(failures)}`")
    lines.append(f"Late failures (>=450): `{len(late)}`")
    lines.append(f"Very late failures (>=475): `{len(very_late)}`")
    lines.append(f"Failure distribution: `{dict(by_failure)}`")
    lines.append(f"Traced arbitration effect vs pure PPO: `{dict(effect)}`")
    lines.append(f"Selected regime counts in traced failures: `{dict(selected)}`")
    lines.append("")
    lines.append("## Easiest To Save")
    for item in sorted(failures, key=lambda x: x["steps"], reverse=True)[:30]:
        lines.append(f"- `{item['seed']}`: step `{item['steps']}`, `{item['failure']}`")
    lines.append("")
    lines.append("## Trace Diagnostics")
    if details:
        for key in [
            "last100_pole1_max_abs_angle",
            "last100_pole1_max_abs_velocity",
            "last100_force_oscillation_score",
            "last100_max_rail_abs_x",
            "policy_final_force_abs_diff_mean",
        ]:
            values = [float(item.get(key, 0.0)) for item in details]
            lines.append(f"- `{key}` avg `{np.mean(values):.3f}`, p90 `{np.percentile(values, 90):.3f}`")
    (out / "rescue_patterns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_trace_samples(best: dict[str, Any], baseline: dict[str, Any], comparison: dict[str, Any], args: argparse.Namespace, out: Path) -> None:
    sample_dir = out / "trace_samples"
    per_seed = sorted(best["per_seed"], key=lambda item: item["steps"])
    picks = []
    if per_seed:
        picks.extend([("worst", per_seed[0]["seed"]), ("median", per_seed[len(per_seed) // 2]["seed"]), ("best", per_seed[-1]["seed"])])
    picks.extend(("rescued", seed) for seed in comparison["fail_to_success"][: args.trace_sample_limit])
    picks.extend(("worsened", seed) for seed in comparison["success_to_fail"][: args.trace_sample_limit])
    seen = set()
    for label, seed in picks:
        key = (label, int(seed))
        if key in seen:
            continue
        seen.add(key)
        row = rollout_candidate(args, controller_for(args, RescueConfig(**best["rescue_config"])), int(seed), trace=True)
        trace = row.pop("trace")
        path = sample_dir / f"{label}_seed_{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": row, "steps": trace}, indent=2), encoding="utf-8")


def solved(result: dict[str, Any]) -> bool:
    return (
        result["episodes"] >= 300
        and result["mean_survival"] >= 475.0
        and result["p10_survival"] >= 350.0
        and result["success_at_500"] >= 0.70
    )


def write_summary(best: dict[str, Any], baseline: dict[str, Any], comparison: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> None:
    failures = {k: v for k, v in best["failure_distribution"].items() if k != "success"}
    dominant = max(failures.items(), key=lambda item: item[1])[0] if failures else "none"
    command = (
        "uv run python scripts/run_n4_final_success_gap.py "
        f"--out {out} --validation-episodes {args.validation_episodes} "
        f"--validation-seed-start {args.validation_seed_start}"
    )
    lines = ["# N=4 Final Success Gap Report", ""]
    lines.append(f"Status: `{'solved' if solved(best) else 'not solved'}`")
    lines.append(f"Best candidate: `{best['candidate']}`")
    lines.append(f"Best mean/p10/success: `{best['mean_survival']:.1f}` / `{best['p10_survival']:.1f}` / `{best['success_at_500']:.3f}`")
    lines.append(f"Baseline mean/p10/success: `{baseline['mean_survival']:.1f}` / `{baseline['p10_survival']:.1f}` / `{baseline['success_at_500']:.3f}`")
    lines.append("")
    lines.append("## Answers")
    lines.append(f"- Did any patch reach success >=0.70? `{'yes' if best['success_at_500'] >= 0.70 else 'no'}`.")
    lines.append(f"- Did it preserve mean/p10 thresholds? `{'yes' if best['mean_survival'] >= 475.0 and best['p10_survival'] >= 350.0 else 'no'}`.")
    lines.append(f"- Fail -> success seeds: `{len(comparison['fail_to_success'])}`.")
    lines.append(f"- Success -> fail seeds: `{len(comparison['success_to_fail'])}`.")
    mechanism = (
        f"{best['candidate']} with config in `best_candidate_config.yaml`"
        if comparison["fail_to_success"]
        else "none; no candidate produced fail->success rescues"
    )
    lines.append(f"- Mechanism: `{mechanism}`.")
    lines.append(f"- Is pole_1_angle still dominant? `{'yes' if dominant == 'pole_1_angle' else 'no'}`; dominant failure is `{dominant}`.")
    lines.append(f"- Enough to claim solved under fixed threshold? `{'yes' if solved(best) else 'no'}`.")
    lines.append("")
    lines.append("## Candidate Table")
    for row in rows:
        if row["phase"] == "full":
            lines.append(
                f"- `{row['candidate']}`: mean `{row['mean_survival']:.1f}`, p10 `{row['p10_survival']:.1f}`, "
                f"success `{row['success_at_500']:.3f}`, net success `{row['net_success_change']:+d}`"
            )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("```bash")
    lines.append(command)
    lines.append("```")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"reports/n4_final_success_gap_{timestamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8")
    baseline = load_baseline(args)
    baseline["candidate"] = "baseline_best_frozen"
    if "success_at_500" not in baseline:
        baseline["success_at_500"] = baseline.get("success_rate", 0.0)
    (out / "baseline_300_eval.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    details = analyze_baseline_failures(baseline, args, out)
    write_rescue_patterns(details, baseline, out)

    all_seeds = [args.validation_seed_start + idx for idx in range(args.validation_episodes)]
    dev_seeds = choose_dev_seeds(baseline, args)
    (out / "dev_seeds.json").write_text(json.dumps(dev_seeds, indent=2), encoding="utf-8")
    candidates = candidate_configs()
    rows: list[dict[str, Any]] = []
    dev_results: dict[str, dict[str, Any]] = {}
    for name, rescue in candidates.items():
        print(f"[n4-final-gap] dev eval {name} on {len(dev_seeds)} seeds", flush=True)
        result = evaluate_candidate(name, rescue, dev_seeds, args, trace=False)
        dev_results[name] = result
        rows.append(row_for(result, baseline, "dev"))
        write_candidate_tables(rows, out)
        print(
            f"[n4-final-gap] dev {name}: mean={result['mean_survival']:.1f} "
            f"success={result['success_at_500']:.3f}",
            flush=True,
        )

    ranked_dev = sorted(rows, key=lambda row: (row["net_success_change"], row["success_at_500"], row["mean_survival"]), reverse=True)
    if args.full_eval_all:
        full_names = list(candidates)
    else:
        full_names = [row["candidate"] for row in ranked_dev[: max(3, args.full_eval_top)]]
    full_results = []
    for name in full_names:
        print(f"[n4-final-gap] full eval {name} on {len(all_seeds)} seeds", flush=True)
        result = evaluate_candidate(name, candidates[name], all_seeds, args, trace=False)
        full_results.append(result)
        rows.append(row_for(result, baseline, "full"))
        write_candidate_tables(rows, out)
        print(
            f"[n4-final-gap] full {name}: mean={result['mean_survival']:.1f} "
            f"p10={result['p10_survival']:.1f} success={result['success_at_500']:.3f}",
            flush=True,
        )
    best = max(full_results, key=lambda item: (item["success_at_500"], item["mean_survival"], item["p10_survival"]))
    comparison = write_changed_seed_analysis(best, baseline, out)
    (out / "final_300_eval.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (out / "best_candidate_config.yaml").write_text(yaml.safe_dump(best["rescue_config"], sort_keys=False), encoding="utf-8")
    export_trace_samples(best, baseline, comparison, args, out)
    write_summary(best, baseline, comparison, rows, args, out)
    return {"out": str(out), "best": best, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--baseline-eval", default=BASELINE_EVAL)
    parser.add_argument("--feedforward-checkpoint", default=FEEDFORWARD_CHECKPOINT)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw"], default="normalized_raw")
    parser.add_argument("--validation-seed-start", type=int, default=930000)
    parser.add_argument("--validation-episodes", type=int, default=300)
    parser.add_argument("--late-failure-step", type=int, default=450)
    parser.add_argument("--dev-late-failures", type=int, default=60)
    parser.add_argument("--dev-early-failures", type=int, default=30)
    parser.add_argument("--dev-successes", type=int, default=60)
    parser.add_argument("--failure-trace-limit", type=int, default=100)
    parser.add_argument("--trace-sample-limit", type=int, default=8)
    parser.add_argument("--full-eval-top", type=int, default=4)
    parser.add_argument("--full-eval-all", action="store_true", default=False)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--x-threshold", type=float, default=2.4)
    parser.add_argument("--theta-threshold", type=float, default=12.0 * 2.0 * np.pi / 360.0)
    parser.add_argument("--velocity-failure-threshold", type=float, default=8.0)
    parser.add_argument("--rail-conflict-x", type=float, default=1.5)
    parser.add_argument("--pole1-velocity-mix", type=float, default=0.30)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()
    result = run(args)
    best = result["best"]
    print(
        json.dumps(
            {
                "out": result["out"],
                "best": best["candidate"],
                "mean": best["mean_survival"],
                "p10": best["p10_survival"],
                "success": best["success_at_500"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
