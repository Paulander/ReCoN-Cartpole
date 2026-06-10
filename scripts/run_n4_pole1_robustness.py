from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from recon_cartpole.control.policy_observation import policy_observation_from_state
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import Pole1FixConfig, ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminalConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_n4_autonomous_recurrent import config_hash  # noqa: E402

FEEDFORWARD_CHECKPOINT = (
    "reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/"
    "checkpoint_025000.zip"
)
RECURRENT_CHECKPOINT = (
    "reports/n4_autonomous_recurrent_20260610_160353/candidate_logs/"
    "b0021ba0d0bb/supervised/mingru_terminal.pt"
)


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


def force_from_action(action: Any, args: argparse.Namespace) -> float:
    bins = max(2, int(args.discrete_action_bins))
    idx = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, bins - 1))
    if bins == 2:
        return args.force_mag if idx == 1 else -args.force_mag
    return float(np.linspace(-args.force_mag, args.force_mag, bins)[idx])


def classify(raw_state: list[float], steps: int, args: argparse.Namespace) -> str:
    if steps >= args.horizon:
        return "success"
    raw = np.asarray(raw_state, dtype=float)
    if raw.size < 2 + 2 * args.n_poles:
        return "unknown"
    x = float(raw[0])
    theta = raw[2 : 2 + args.n_poles]
    theta_dot = raw[2 + args.n_poles : 2 + 2 * args.n_poles]
    if x <= -args.x_threshold * 0.98:
        return "rail_left"
    if x >= args.x_threshold * 0.98:
        return "rail_right"
    worst_angle = int(np.argmax(np.abs(theta)))
    if abs(float(theta[worst_angle])) >= args.theta_threshold * 0.98:
        return f"pole_{worst_angle}_angle"
    worst_vel = int(np.argmax(np.abs(theta_dot)))
    if abs(float(theta_dot[worst_vel])) > args.velocity_failure_threshold:
        return f"pole_{worst_vel}_velocity"
    return "undercorrection"


def pole1_diagnostics(trace: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not trace:
        return {}
    raws = [np.asarray(step.get("raw_state", []), dtype=float) for step in trace]
    raws = [raw for raw in raws if raw.size >= 2 + 2 * args.n_poles]
    if not raws or args.n_poles < 2:
        return {}
    theta1 = np.asarray([raw[3] for raw in raws], dtype=float)
    theta1_dot = np.asarray([raw[3 + args.n_poles] for raw in raws], dtype=float)
    x = np.asarray([raw[0] for raw in raws], dtype=float)
    forces = np.asarray([float(step.get("force", 0.0) or 0.0) for step in trace], dtype=float)
    desired = np.where(theta1 + args.pole1_velocity_mix * theta1_dot >= 0.0, 1.0, -1.0)
    active = np.abs(forces) > 1e-6
    sign_correct = float(np.mean(np.sign(forces[active]) == desired[active])) if np.any(active) else 0.0
    flips = int(
        sum(
            1
            for a, b in zip(forces, forces[1:])
            if abs(a) > 1e-6 and abs(b) > 1e-6 and np.sign(a) != np.sign(b)
        )
    )
    selected = Counter(str(step.get("selected_regime", "")) for step in trace if step.get("selected_regime"))
    policy_overrides = 0
    policy_count = 0
    mingru_low_conf = 0
    mingru_high_wrong = 0
    for step in trace:
        final_force = float(step.get("force", 0.0) or 0.0)
        policy = step.get("policy_terminal", {}) or {}
        if policy.get("policy_force") is not None:
            policy_count += 1
            policy_force = float(policy.get("policy_force"))
            if abs(policy_force) > 1e-6 and abs(final_force) > 1e-6:
                if np.sign(policy_force) != np.sign(final_force):
                    policy_overrides += 1
        mingru = step.get("mingru_terminal", {}) or {}
        if mingru:
            conf = float(mingru.get("confidence", 0.0) or 0.0)
            if conf < args.low_confidence_threshold:
                mingru_low_conf += 1
            if conf > args.high_confidence_threshold:
                mingru_high_wrong += 1
    if forces.size >= 30:
        tail_force_std = float(np.std(forces[-30:]))
    else:
        tail_force_std = float(np.std(forces)) if forces.size else 0.0
    return {
        "pole1_max_abs_angle": float(np.max(np.abs(theta1))),
        "pole1_max_abs_velocity": float(np.max(np.abs(theta1_dot))),
        "pole1_final_angle": float(theta1[-1]),
        "pole1_final_velocity": float(theta1_dot[-1]),
        "force_sign_correct_rate": sign_correct,
        "force_sign_flips": flips,
        "tail_force_std": tail_force_std,
        "rail_conflict_steps": int(np.sum(np.abs(x) > args.rail_conflict_x)),
        "selected_regime_counts": dict(selected),
        "policy_override_sign_mismatches": policy_overrides,
        "policy_terminal_steps": policy_count,
        "policy_override_rate": float(policy_overrides / policy_count) if policy_count else 0.0,
        "mingru_low_confidence_steps": mingru_low_conf,
        "mingru_high_confidence_steps": mingru_high_wrong,
    }


def eval_controller(
    label: str,
    controller: ReConCartPoleController,
    seeds: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    steps: list[float] = []
    returns: list[float] = []
    failures: Counter[str] = Counter()
    for seed in seeds:
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=True)
        trace = result["trace"]
        final_raw = trace[-1].get("raw_state", []) if trace else []
        step_count = int(result["steps"])
        failure = classify(final_raw, step_count, args)
        failures[failure] += 1
        steps.append(float(step_count))
        returns.append(float(result["return"]))
        per_seed.append(
            {
                "seed": seed,
                "mode": label,
                "steps": step_count,
                "return": float(result["return"]),
                "success": step_count >= args.horizon,
                "failure": failure,
                **pole1_diagnostics(trace, args),
            }
        )
    summary = summarize_steps(steps, args.horizon)
    values = np.asarray(steps, dtype=float)
    summary.update(
        {
            "mode": label,
            "episodes": len(seeds),
            "median_survival": float(np.median(values)) if values.size else 0.0,
            "p90_survival": float(np.percentile(values, 90)) if values.size else 0.0,
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "failure_distribution": dict(failures),
            "wall_clock_seconds": time.perf_counter() - started,
            "per_seed": per_seed,
            "mechanisms": controller.learning_mechanisms(),
        }
    )
    return summary


def eval_pure_feedforward(label: str, seeds: list[int], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from stable_baselines3 import PPO
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pure feedforward evaluation requires stable-baselines3") from exc
    model = PPO.load(args.feedforward_checkpoint, device="cpu")
    per_seed: list[dict[str, Any]] = []
    steps: list[float] = []
    returns: list[float] = []
    failures: Counter[str] = Counter()
    for seed in seeds:
        env = make_env(args)
        obs, info = env.reset(seed=seed)
        trace: list[dict[str, Any]] = []
        total = 0.0
        for step in range(args.horizon):
            policy_obs = policy_observation_from_state(
                obs, info.get("raw_state"), args.n_poles, args.observation_mode
            )
            action, _state = model.predict(policy_obs, deterministic=True)
            force = force_from_action(action, args)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            trace.append(
                {
                    "step": step,
                    "raw_state": np.asarray(info.get("raw_state", []), dtype=float).tolist(),
                    "action": int(np.asarray(action).reshape(-1)[0]),
                    "force": force,
                    "env_reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "policy_terminal": {"policy_force": force, "force": force},
                }
            )
            if terminated or truncated:
                break
        step_count = len(trace)
        final_raw = trace[-1].get("raw_state", []) if trace else []
        failure = classify(final_raw, step_count, args)
        failures[failure] += 1
        steps.append(float(step_count))
        returns.append(float(total))
        per_seed.append(
            {
                "seed": seed,
                "mode": label,
                "steps": step_count,
                "return": float(total),
                "success": step_count >= args.horizon,
                "failure": failure,
                **pole1_diagnostics(trace, args),
            }
        )
    summary = summarize_steps(steps, args.horizon)
    values = np.asarray(steps, dtype=float)
    summary.update(
        {
            "mode": label,
            "episodes": len(seeds),
            "median_survival": float(np.median(values)) if values.size else 0.0,
            "p90_survival": float(np.percentile(values, 90)) if values.size else 0.0,
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "failure_distribution": dict(failures),
            "wall_clock_seconds": time.perf_counter() - started,
            "per_seed": per_seed,
            "mechanisms": {
                "feedforward_policy_terminal": True,
                "pure_policy_baseline": True,
                "ReCoN_arbitration": False,
                "edge_plasticity": False,
                "bandit": False,
                "slow_consolidation": False,
                "pole1_fix": False,
            },
        }
    )
    return summary


def controller_for(mode: str, args: argparse.Namespace) -> ReConCartPoleController:
    pole_fix = Pole1FixConfig(
        enabled=mode == "recon_feedforward_terminal_with_pole1_fix",
        angle_threshold=args.pole1_angle_threshold,
        velocity_threshold=args.pole1_velocity_threshold,
        urgency_boost=args.pole1_urgency_boost,
        confidence_boost=args.pole1_confidence_boost,
        force_blend=args.pole1_force_blend,
        rail_guard=args.pole1_rail_guard,
        velocity_mix=args.pole1_velocity_mix,
    )
    mingru = MinGRUTerminalConfig(
        enabled=mode == "recon_mingru_terminal_frozen",
        hidden_size=args.mingru_hidden_size,
        sequence_length=args.mingru_sequence_length,
        observation_mode=args.observation_mode,
        checkpoint_path=args.recurrent_checkpoint,
        scope=args.policy_terminal_scope,
    )
    controller_mode = "recon_mingru_terminal" if mode == "recon_mingru_terminal_frozen" else mode
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode=controller_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=mode.endswith("plus_recon_learning"),
            reset_bandit_each_episode=False,
            policy_terminal_path=args.feedforward_checkpoint,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.observation_mode,
            mingru_terminal=mingru,
            pole1_fix=pole_fix,
        )
    )


def train_plus_controller(controller: ReConCartPoleController, args: argparse.Namespace) -> int:
    train_steps = 0
    for idx in range(args.plus_train_episodes):
        result = rollout(
            make_env(args),
            controller,
            seed=args.plus_train_seed_start + idx,
            horizon=args.horizon,
            trace=False,
        )
        train_steps += int(result["steps"])
    controller.config.learn = False
    return train_steps


def summary_row(result: dict[str, Any], checkpoint: str = "", train_steps: int = 0) -> dict[str, Any]:
    return {
        "mode": result["mode"],
        "config_hash": config_hash({"mode": result["mode"], "checkpoint": checkpoint, "train_steps": train_steps}),
        "checkpoint_path": checkpoint,
        "train_env_steps": train_steps,
        "eval_episodes": result["episodes"],
        "mean_survival": result["mean_survival"],
        "median_survival": result["median_survival"],
        "p10_survival": result["p10_survival"],
        "p90_survival": result["p90_survival"],
        "success_at_500": result["success_rate"],
        "max_survival": result["max_survival"],
        "failure_distribution": result["failure_distribution"],
        "mechanisms": result["mechanisms"],
        "wall_clock_seconds": result["wall_clock_seconds"],
    }


def write_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rerun_controller_trace(mode: str, seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if mode == "pure_feedforward_policy_terminal":
        try:
            from stable_baselines3 import PPO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pure feedforward trace requires stable-baselines3") from exc
        model = PPO.load(args.feedforward_checkpoint, device="cpu")
        env = make_env(args)
        obs, info = env.reset(seed=seed)
        trace = []
        for step in range(args.horizon):
            policy_obs = policy_observation_from_state(
                obs, info.get("raw_state"), args.n_poles, args.observation_mode
            )
            action, _state = model.predict(policy_obs, deterministic=True)
            force = force_from_action(action, args)
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append(
                {
                    "step": step,
                    "raw_state": np.asarray(info.get("raw_state", []), dtype=float).tolist(),
                    "action": int(np.asarray(action).reshape(-1)[0]),
                    "force": force,
                    "env_reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "policy_terminal": {"policy_force": force, "force": force},
                }
            )
            if terminated or truncated:
                break
        return trace
    controller = controller_for(mode, args)
    result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=True)
    return result["trace"]


def export_samples(results: dict[str, dict[str, Any]], out: Path, args: argparse.Namespace) -> None:
    sample_dir = out / "trace_samples"
    for mode, result in results.items():
        ordered = sorted(result["per_seed"], key=lambda item: item["steps"])
        if not ordered:
            continue
        picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        labels = ["worst", "median", "best"]
        pole1 = [item for item in ordered if item["failure"] == "pole_1_angle"][: args.pole1_trace_samples]
        picks.extend(pole1)
        labels.extend([f"pole1_failure_{idx:02d}" for idx in range(len(pole1))])
        for label, item in zip(labels, picks):
            trace = rerun_controller_trace(mode, int(item["seed"]), args)
            write_trace(
                sample_dir / mode / f"{label}_seed_{item['seed']}.json",
                {"metadata": item, "steps": trace},
            )


def compare_outcomes(ff: dict[str, Any], recurrent: dict[str, Any]) -> dict[str, Any]:
    ff_by_seed = {item["seed"]: item for item in ff["per_seed"]}
    rec_by_seed = {item["seed"]: item for item in recurrent["per_seed"]}
    groups = {
        "solved_by_feedforward_failed_by_mingru": [],
        "solved_by_mingru_failed_by_feedforward": [],
        "failed_by_both": [],
        "solved_by_both": [],
    }
    for seed, frow in ff_by_seed.items():
        rrow = rec_by_seed.get(seed)
        if not rrow:
            continue
        if frow["success"] and not rrow["success"]:
            groups["solved_by_feedforward_failed_by_mingru"].append(seed)
        elif rrow["success"] and not frow["success"]:
            groups["solved_by_mingru_failed_by_feedforward"].append(seed)
        elif frow["success"] and rrow["success"]:
            groups["solved_by_both"].append(seed)
        else:
            groups["failed_by_both"].append(seed)
    return {key: {"count": len(value), "seeds": value[:50]} for key, value in groups.items()}


def write_partial_results(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], out: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["mean_survival"], reverse=True)
    (out / "partial_leaderboard.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    partial = {
        mode: {key: value for key, value in result.items() if key != "per_seed"}
        for mode, result in results.items()
    }
    (out / "partial_results.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")


def write_reports(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    out: Path,
    args: argparse.Namespace,
) -> None:
    rows = sorted(rows, key=lambda item: item["mean_survival"], reverse=True)
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "300_seed_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "mode",
            "mean_survival",
            "median_survival",
            "p10_survival",
            "p90_survival",
            "success_at_500",
            "max_survival",
            "eval_episodes",
            "train_env_steps",
            "checkpoint_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    taxonomy = ["# Failure Taxonomy", ""]
    for row in rows:
        taxonomy.append(f"## {row['mode']}")
        for key, value in sorted(row["failure_distribution"].items(), key=lambda kv: (-kv[1], kv[0])):
            taxonomy.append(f"- `{key}`: {value}")
        taxonomy.append("")
    (out / "failure_taxonomy.md").write_text("\n".join(taxonomy), encoding="utf-8")

    trace_lines = ["# Trace Comparison", "", "## Outcome Groups", "", json.dumps(comparison, indent=2), ""]
    ff = results.get("recon_feedforward_terminal_frozen", {})
    rec = results.get("recon_mingru_terminal_frozen", {})
    if ff and rec:
        ff_fail = [item for item in ff["per_seed"] if not item["success"]]
        rec_fail = [item for item in rec["per_seed"] if not item["success"]]
        trace_lines.extend(
            [
                "## Diagnostics",
                "",
                f"Feedforward failed episodes avg force sign correctness: `{np.mean([i.get('force_sign_correct_rate', 0.0) for i in ff_fail]) if ff_fail else 0.0:.3f}`",
                f"Recurrent failed episodes avg force sign correctness: `{np.mean([i.get('force_sign_correct_rate', 0.0) for i in rec_fail]) if rec_fail else 0.0:.3f}`",
                f"Feedforward failed episodes avg policy override rate: `{np.mean([i.get('policy_override_rate', 0.0) for i in ff_fail]) if ff_fail else 0.0:.3f}`",
                f"Recurrent failed episodes avg force sign flips: `{np.mean([i.get('force_sign_flips', 0.0) for i in rec_fail]) if rec_fail else 0.0:.1f}`",
                "",
                "Sample traces are under `trace_samples/`, including best/median/worst and pole_1_angle failures where available.",
            ]
        )
    (out / "trace_comparison.md").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    best = rows[0]
    solved = (
        best["eval_episodes"] >= 300
        and best["mean_survival"] >= 475
        and best["p10_survival"] >= 350
        and best["success_at_500"] >= 0.70
    )
    ff_row = next((row for row in rows if row["mode"] == "recon_feedforward_terminal_frozen"), None)
    fix_row = next((row for row in rows if row["mode"] == "recon_feedforward_terminal_with_pole1_fix"), None)
    plus_row = next((row for row in rows if row["mode"] == "recon_feedforward_terminal_plus_recon_learning"), None)
    non_success = {k: v for k, v in best["failure_distribution"].items() if k != "success"}
    dominant = max(non_success.items(), key=lambda kv: kv[1])[0] if non_success else "none"
    summary = [
        "# N=4 Pole_1 Robustness Report",
        "",
        f"Status: `{'solved' if solved else 'not solved'}`",
        f"Report directory: `{out}`",
        f"Best candidate: `{best['mode']}`",
        f"Best mean/p10/success: `{best['mean_survival']:.1f}` / `{best['p10_survival']:.1f}` / `{best['success_at_500']:.2f}`",
        "",
        "## Answers",
        "",
        f"- Does the best feedforward policy terminal solve N=4 on 300 held-out seeds? `{'yes' if solved and best['mode'] in ('pure_feedforward_policy_terminal', 'recon_feedforward_terminal_frozen') else 'no'}`.",
        f"- Is pole_1_angle still dominant? `{'yes' if dominant == 'pole_1_angle' else 'no'}`; dominant non-success failure for the best row is `{dominant}`.",
        f"- Does ReCoN arbitration help feedforward? `{delta_text(ff_row, results.get('pure_feedforward_policy_terminal'))}`.",
        f"- Does ReCoN learning around feedforward help? `{delta_text(plus_row, ff_row)}`.",
        f"- Did the pole_1 fix improve robustness? `{delta_text(fix_row, ff_row)}`.",
        "- Remaining problem: compare `trace_comparison.md`; if feedforward is solved, remaining recurrent gap is policy learning. If not, inspect pole_1 timing and environment/control edge cases before N=5.",
        "",
        "## Key Rows",
        "",
    ]
    for row in rows:
        summary.append(
            f"- `{row['mode']}`: mean `{row['mean_survival']:.1f}`, p10 `{row['p10_survival']:.1f}`, success `{row['success_at_500']:.2f}`"
        )
    summary.extend(
        [
            "",
            "## Reproduce Best Eval",
            "",
            "```bash",
            reproduce_command(args, out),
            "```",
            "",
            "## Resume/Extend",
            "",
            "```bash",
            reproduce_command(args, out.with_name(out.name + "_extended")) + " --plus-train-episodes 120",
            "```",
        ]
    )
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def delta_text(row: dict[str, Any] | None, baseline: dict[str, Any] | None) -> str:
    if not row or not baseline:
        return "insufficient evidence"
    base_mean = baseline.get("mean_survival", 0.0)
    base_success = baseline.get("success_at_500", baseline.get("success_rate", 0.0))
    if "per_seed" in baseline:
        steps = [item["steps"] for item in baseline["per_seed"]]
        base_mean = float(np.mean(steps)) if steps else 0.0
        base_success = float(np.mean([bool(item["success"]) for item in baseline["per_seed"]])) if steps else 0.0
    delta_mean = row["mean_survival"] - float(base_mean)
    delta_success = row.get("success_at_500", row.get("success_rate", 0.0)) - float(base_success)
    verdict = "neutral"
    if abs(delta_success) >= 0.001:
        verdict = "helped" if delta_success > 0.0 else "hurt"
    elif abs(delta_mean) >= 2.0:
        verdict = "helped" if delta_mean > 0.0 else "hurt"
    return f"{verdict} ({delta_mean:+.1f} mean steps, {delta_success:+.3f} success)"


def reproduce_command(args: argparse.Namespace, out: Path) -> str:
    return (
        "uv run python scripts/run_n4_pole1_robustness.py "
        f"--out {out} --feedforward-checkpoint {args.feedforward_checkpoint} "
        f"--recurrent-checkpoint {args.recurrent_checkpoint} --validation-episodes {args.validation_episodes} "
        f"--validation-seed-start {args.validation_seed_start}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"reports/n4_pole1_robustness_{timestamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump({"env": env_payload(args), "args": vars(args)}, sort_keys=False),
        encoding="utf-8",
    )
    seeds = [args.validation_seed_start + idx for idx in range(args.validation_episodes)]
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    if args.include_baselines:
        for mode in ["baseline_heuristic", "static_recon"]:
            print(f"[n4-pole1] evaluating {mode} on {len(seeds)} seeds", flush=True)
            controller = ReConCartPoleController(
                RunnerConfig(
                    n_poles=args.n_poles,
                    mode=mode,
                    action_mode="discrete",
                    discrete_action_bins=args.discrete_action_bins,
                    force_mag=args.force_mag,
                    selection_mode=args.selection_mode,
                    learn=False,
                )
            )
            result = eval_controller(mode, controller, seeds, args)
            results[mode] = result
            rows.append(summary_row(result))
            write_partial_results(rows, results, out)
            print(f"[n4-pole1] finished {mode}: mean={result['mean_survival']:.1f} success={result['success_rate']:.3f}", flush=True)

    print(f"[n4-pole1] evaluating pure_feedforward_policy_terminal on {len(seeds)} seeds", flush=True)
    pure = eval_pure_feedforward("pure_feedforward_policy_terminal", seeds, args)
    results[pure["mode"]] = pure
    rows.append(summary_row(pure, args.feedforward_checkpoint))
    write_partial_results(rows, results, out)
    print(f"[n4-pole1] finished pure_feedforward_policy_terminal: mean={pure['mean_survival']:.1f} success={pure['success_rate']:.3f}", flush=True)

    for mode in [
        "recon_feedforward_terminal_frozen",
        "recon_feedforward_terminal_plus_recon_learning",
        "recon_feedforward_terminal_with_pole1_fix",
        "recon_mingru_terminal_frozen",
    ]:
        print(f"[n4-pole1] evaluating {mode} on {len(seeds)} seeds", flush=True)
        controller = controller_for(mode, args)
        train_steps = 0
        if mode == "recon_feedforward_terminal_plus_recon_learning":
            print(f"[n4-pole1] training {mode} for {args.plus_train_episodes} episodes", flush=True)
            train_steps = train_plus_controller(controller, args)
            controller.save_checkpoint(str(out / "recon_feedforward_plus_learning_checkpoint.json"))
        result = eval_controller(mode, controller, seeds, args)
        results[mode] = result
        checkpoint = args.recurrent_checkpoint if "mingru" in mode else args.feedforward_checkpoint
        rows.append(summary_row(result, checkpoint, train_steps))
        write_partial_results(rows, results, out)
        print(f"[n4-pole1] finished {mode}: mean={result['mean_survival']:.1f} success={result['success_rate']:.3f}", flush=True)

    comparison = compare_outcomes(
        results["recon_feedforward_terminal_frozen"],
        results["recon_mingru_terminal_frozen"],
    )
    write_reports(rows, results, comparison, out, args)
    export_samples(
        {
            key: results[key]
            for key in [
                "recon_feedforward_terminal_frozen",
                "recon_feedforward_terminal_with_pole1_fix",
                "recon_mingru_terminal_frozen",
            ]
            if key in results
        },
        out,
        args,
    )
    return {"out": str(out), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--feedforward-checkpoint", default=FEEDFORWARD_CHECKPOINT)
    parser.add_argument("--recurrent-checkpoint", default=RECURRENT_CHECKPOINT)
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
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw"], default="normalized_raw")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--mingru-hidden-size", type=int, default=32)
    parser.add_argument("--mingru-sequence-length", type=int, default=16)
    parser.add_argument("--validation-seed-start", type=int, default=930000)
    parser.add_argument("--validation-episodes", type=int, default=300)
    parser.add_argument("--plus-train-episodes", type=int, default=40)
    parser.add_argument("--plus-train-seed-start", type=int, default=760000)
    parser.add_argument("--include-baselines", action="store_true", default=True)
    parser.add_argument("--skip-baselines", dest="include_baselines", action="store_false")
    parser.add_argument("--pole1-angle-threshold", type=float, default=0.14)
    parser.add_argument("--pole1-velocity-threshold", type=float, default=1.2)
    parser.add_argument("--pole1-urgency-boost", type=float, default=0.45)
    parser.add_argument("--pole1-confidence-boost", type=float, default=0.20)
    parser.add_argument("--pole1-force-blend", type=float, default=0.35)
    parser.add_argument("--pole1-rail-guard", type=float, default=2.05)
    parser.add_argument("--pole1-velocity-mix", type=float, default=0.30)
    parser.add_argument("--x-threshold", type=float, default=2.4)
    parser.add_argument("--theta-threshold", type=float, default=12.0 * 2.0 * np.pi / 360.0)
    parser.add_argument("--velocity-failure-threshold", type=float, default=8.0)
    parser.add_argument("--rail-conflict-x", type=float, default=1.5)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--pole1-trace-samples", type=int, default=5)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"out": result["out"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
