from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_policy_dataset import collect as collect_dataset  # noqa: E402
from train_mingru_supervised import train as train_mingru_supervised  # noqa: E402
from train_recurrent_terminal_ladder import config_hash  # noqa: E402

CANONICAL_TEACHER = (
    "reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/"
    "checkpoint_025000.zip"
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


def terminal_config(
    args: argparse.Namespace,
    checkpoint_path: str,
    hidden_size: int,
    sequence_length: int,
    blend: float,
    confidence_floor: float,
) -> MinGRUTerminalConfig:
    return MinGRUTerminalConfig(
        enabled=True,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        blend=blend,
        scope=args.scope,
        confidence_floor=confidence_floor,
        checkpoint_path=checkpoint_path,
    )


def mechanism_flags(mode: str) -> dict[str, bool]:
    return {
        "edge_plasticity": mode == "recon_mingru_terminal_plus_recon_learning",
        "bandit": mode == "recon_mingru_terminal_plus_recon_learning",
        "slow_consolidation": mode == "recon_mingru_terminal_plus_recon_learning",
        "node_param_learning": False,
        "gain_mutation": False,
        "feedforward_policy_terminal": mode == "recon_policy_terminal",
        "minGRU_terminal": mode in {
            "pure_mingru_policy",
            "recon_mingru_terminal_frozen",
            "recon_mingru_terminal_plus_recon_learning",
        },
        "pure_policy_baseline": mode == "pure_mingru_policy",
        "ReCoN_arbitration": mode in {
            "static_recon",
            "recon_policy_terminal",
            "recon_mingru_terminal_frozen",
            "recon_mingru_terminal_plus_recon_learning",
        },
    }


def classify_trace(trace: list[dict[str, Any]], args: argparse.Namespace, mode: str) -> str:
    if not trace:
        return "unknown"
    last = trace[-1]
    if len(trace) >= args.horizon and last.get("truncated"):
        return "success"
    raw = np.asarray(last.get("raw_state", []), dtype=float)
    n = args.n_poles
    if raw.size >= 2 + 2 * n:
        x = float(raw[0])
        theta = raw[2 : 2 + n]
        theta_dot = raw[2 + n : 2 + 2 * n]
        if x <= -args.x_threshold * 0.98:
            return "rail_left"
        if x >= args.x_threshold * 0.98:
            return "rail_right"
        worst_angle = int(np.argmax(np.abs(theta))) if theta.size else 0
        if theta.size and abs(float(theta[worst_angle])) >= args.theta_threshold * 0.98:
            return f"pole_{worst_angle}_angle"
        worst_vel = int(np.argmax(np.abs(theta_dot))) if theta_dot.size else 0
        if theta_dot.size and abs(float(theta_dot[worst_vel])) > args.velocity_failure_threshold:
            return f"pole_{worst_vel}_velocity"
    mingru = last.get("mingru_terminal", {}) or {}
    if mingru:
        confidence = float(mingru.get("confidence", 0.0) or 0.0)
        if confidence < args.low_confidence_threshold:
            return "minGRU_low_confidence"
        if confidence > args.high_confidence_threshold:
            return "minGRU_high_confidence_wrong"
        terminal_force = mingru.get("terminal_force", mingru.get("force"))
        final_force = float(last.get("force", 0.0) or 0.0)
        if terminal_force is not None:
            terminal_force = float(terminal_force)
            if abs(terminal_force) > 1e-6 and abs(final_force) > 1e-6:
                if np.sign(terminal_force) != np.sign(final_force):
                    return "ReCoN_arbitration_overrode_good_policy"
                if mode.startswith("recon_mingru"):
                    return "ReCoN_arbitration_followed_bad_policy"
    force = float(last.get("force", 0.0) or 0.0)
    prev_forces = [float(item.get("force", 0.0) or 0.0) for item in trace[-min(len(trace), 20) :]]
    sign_flips = sum(
        1
        for a, b in zip(prev_forces, prev_forces[1:])
        if abs(a) > 1e-6 and abs(b) > 1e-6 and np.sign(a) != np.sign(b)
    )
    if sign_flips >= 8:
        return "force_oscillation"
    if abs(force) >= args.force_mag * 0.95:
        return "overcorrection"
    return "undercorrection"


def write_trace(path: Path, trace: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "steps": trace}, indent=2), encoding="utf-8")


def evaluate_controller(
    mode: str,
    args: argparse.Namespace,
    seeds: list[int],
    controller: ReConCartPoleController,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    steps: list[float] = []
    returns: list[float] = []
    failures: Counter[str] = Counter()
    traces: list[tuple[int, float, list[dict[str, Any]]]] = []
    started = time.perf_counter()
    for seed in seeds:
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=True)
        step_count = float(result["steps"])
        steps.append(step_count)
        returns.append(float(result["return"]))
        failure = classify_trace(result["trace"], args, mode)
        failures[failure] += 1
        traces.append((seed, step_count, result["trace"]))
    summary = summarize_steps(steps, args.horizon)
    values = np.asarray(steps, dtype=float)
    summary.update(
        {
            "median_survival": float(np.median(values)) if values.size else 0.0,
            "p90_survival": float(np.percentile(values, 90)) if values.size else 0.0,
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "episodes": len(seeds),
            "failure_distribution": dict(failures),
            "wall_clock_seconds": time.perf_counter() - started,
        }
    )
    if candidate_dir and traces:
        sorted_traces = sorted(traces, key=lambda item: item[1])
        median_idx = len(sorted_traces) // 2
        for label, item in [
            ("worst", sorted_traces[0]),
            ("median", sorted_traces[median_idx]),
            ("best", sorted_traces[-1]),
        ]:
            seed, step_count, trace = item
            write_trace(
                candidate_dir / f"{label}_trace.json",
                trace,
                {"mode": mode, "seed": seed, "steps": step_count},
            )
    return summary


def evaluate_pure_mingru(
    args: argparse.Namespace,
    seeds: list[int],
    checkpoint_path: str,
    config: MinGRUTerminalConfig,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    terminal = MinGRUTerminal(args.n_poles, args.force_mag, args.discrete_action_bins, config)
    steps: list[float] = []
    returns: list[float] = []
    failures: Counter[str] = Counter()
    traces: list[tuple[int, float, list[dict[str, Any]]]] = []
    started = time.perf_counter()
    for seed in seeds:
        env = make_env(args)
        obs, info = env.reset(seed=seed)
        terminal.reset()
        total = 0.0
        trace: list[dict[str, Any]] = []
        for step in range(args.horizon):
            prediction = terminal.predict(obs, info.get("raw_state"), {})
            force = 0.0 if prediction.force is None else float(prediction.force)
            action = action_from_force(force, "discrete", args.force_mag, args.discrete_action_bins)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            trace.append(
                {
                    "step": step,
                    "raw_state": np.asarray(info.get("raw_state", []), dtype=float).tolist(),
                    "action": int(action),
                    "force": force,
                    "env_reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "mingru_terminal": {
                        "available": bool(prediction.valid and prediction.force is not None),
                        "force": prediction.force,
                        "terminal_force": prediction.force,
                        "proposal_force": force,
                        "confidence": prediction.confidence,
                        "value": prediction.value,
                        "failure_probability": prediction.failure_probability,
                        "hidden_norm": prediction.hidden_norm,
                        "sequence_length": prediction.sequence_length,
                        "logits": prediction.logits,
                        "checkpoint_path": checkpoint_path,
                    },
                }
            )
            if terminated or truncated:
                break
        step_count = float(len(trace))
        steps.append(step_count)
        returns.append(total)
        failure = classify_trace(trace, args, "pure_mingru_policy")
        failures[failure] += 1
        traces.append((seed, step_count, trace))
    summary = summarize_steps(steps, args.horizon)
    values = np.asarray(steps, dtype=float)
    summary.update(
        {
            "median_survival": float(np.median(values)) if values.size else 0.0,
            "p90_survival": float(np.percentile(values, 90)) if values.size else 0.0,
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "episodes": len(seeds),
            "failure_distribution": dict(failures),
            "wall_clock_seconds": time.perf_counter() - started,
        }
    )
    if candidate_dir and traces:
        sorted_traces = sorted(traces, key=lambda item: item[1])
        median_idx = len(sorted_traces) // 2
        for label, item in [
            ("pure_worst", sorted_traces[0]),
            ("pure_median", sorted_traces[median_idx]),
            ("pure_best", sorted_traces[-1]),
        ]:
            seed, step_count, trace = item
            write_trace(
                candidate_dir / f"{label}_trace.json",
                trace,
                {"mode": "pure_mingru_policy", "seed": seed, "steps": step_count},
            )
    return summary


def train_recon_learning(
    args: argparse.Namespace,
    checkpoint_path: str,
    config: MinGRUTerminalConfig,
    candidate_dir: Path,
) -> ReConCartPoleController:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_mingru_terminal_plus_recon_learning",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=True,
            reset_bandit_each_episode=False,
            mingru_terminal=config,
        )
    )
    train_steps = 0
    for idx in range(args.recon_learning_train_episodes):
        result = rollout(
            make_env(args),
            controller,
            seed=args.recon_learning_seed_start + idx,
            horizon=args.horizon,
            trace=False,
        )
        train_steps += int(result["steps"])
    controller.config.learn = False
    checkpoint_out = candidate_dir / "recon_learning_checkpoint.json"
    controller.save_checkpoint(str(checkpoint_out))
    return controller


def row_from_result(
    mode: str,
    candidate_id: str,
    checkpoint_path: str,
    config: dict[str, Any],
    train_env_steps: int,
    supervised_samples: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "candidate_id": candidate_id,
        "config_hash": candidate_id,
        "checkpoint_path": checkpoint_path,
        "train_env_steps": train_env_steps,
        "supervised_samples": supervised_samples,
        "wall_clock_seconds": result.get("wall_clock_seconds", 0.0),
        "validation_episodes": result.get("episodes", 0),
        "mean_survival": result.get("mean_survival", 0.0),
        "median_survival": result.get("median_survival", 0.0),
        "p10_survival": result.get("p10_survival", 0.0),
        "p90_survival": result.get("p90_survival", 0.0),
        "success_at_500": result.get("success_rate", 0.0),
        "max_survival": result.get("max_survival", 0.0),
        "failure_distribution": result.get("failure_distribution", {}),
        "mechanisms": mechanism_flags(mode),
        "config": config,
    }


def train_candidate(
    args: argparse.Namespace,
    dataset_path: Path,
    candidate_dir: Path,
    hidden_size: int,
    sequence_length: int,
    blend: float,
    confidence_floor: float,
) -> tuple[str, int]:
    report = train_mingru_supervised(
        SimpleNamespace(
            dataset=str(dataset_path),
            out=str(candidate_dir / "supervised"),
            n_poles=args.n_poles,
            horizon=args.horizon,
            force_mag=args.force_mag,
            discrete_action_bins=args.discrete_action_bins,
            observation_mode=args.observation_mode,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            include_prev_force=args.include_prev_force,
            include_context=args.include_context,
            scope=args.scope,
            blend=blend,
            confidence_floor=confidence_floor,
            epochs=args.supervised_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=0.1,
            value_weight=0.05,
            failure_weight=0.10,
            confidence_weight=0.05,
            max_grad_norm=1.0,
            seed=args.train_seed,
        )
    )
    return str(report["checkpoint_path"]), int(report["samples"])


def build_or_reuse_dataset(args: argparse.Namespace, out: Path) -> Path:
    if args.dataset_path:
        return Path(args.dataset_path)
    dataset_path = out / "teacher_dataset" / "dataset.npz"
    if dataset_path.exists():
        return dataset_path
    dataset_args = SimpleNamespace(
        teacher=args.teacher,
        n_poles=args.n_poles,
        horizon=args.horizon,
        episodes=args.dataset_episodes,
        seed_start=args.train_seed_start,
        dt=args.dt,
        dynamics_mode=args.dynamics_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        selection_mode=args.selection_mode,
        observation_mode=args.observation_mode,
        policy_terminal_path=args.policy_terminal_path,
        policy_terminal_blend=args.policy_terminal_blend,
        policy_terminal_scope=args.policy_terminal_scope,
        failure_window=args.failure_window,
    )
    data = collect_dataset(dataset_args)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **data)
    (dataset_path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "teacher_checkpoint": args.policy_terminal_path,
                "episodes": args.dataset_episodes,
                "samples": int(data["observations"].shape[0]),
                "seed_start": args.train_seed_start,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path


def evaluate_baselines(args: argparse.Namespace, seeds: list[int], out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ["baseline_heuristic", "static_recon"]:
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
        result = evaluate_controller(mode, args, seeds, controller, out / "candidate_logs" / mode)
        rows.append(row_from_result(mode, mode, "", {}, 0, 0, result))
    teacher_path = Path(args.policy_terminal_path)
    if args.include_policy_terminal and args.n_poles == 4 and teacher_path.exists():
        controller = ReConCartPoleController(
            RunnerConfig(
                n_poles=args.n_poles,
                mode="recon_policy_terminal",
                action_mode="discrete",
                discrete_action_bins=args.discrete_action_bins,
                force_mag=args.force_mag,
                selection_mode=args.selection_mode,
                learn=False,
                policy_terminal_path=str(teacher_path),
                policy_terminal_blend=args.policy_terminal_blend,
                policy_terminal_scope=args.policy_terminal_scope,
                policy_terminal_observation_mode=args.observation_mode,
            )
        )
        result = evaluate_controller(
            "recon_policy_terminal",
            args,
            seeds,
            controller,
            out / "candidate_logs" / "recon_policy_terminal",
        )
        rows.append(
            row_from_result(
                "recon_policy_terminal",
                "recon_policy_terminal",
                str(teacher_path),
                {"policy_terminal_blend": args.policy_terminal_blend},
                0,
                0,
                result,
            )
        )
    return rows


def candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants = [(args.blend, args.confidence_floor), (args.aggressive_blend, args.aggressive_confidence_floor)]
    specs = []
    for hidden_size in args.hidden_sizes:
        for sequence_length in args.sequence_lengths:
            for blend, floor in variants:
                payload = {
                    "hidden_size": hidden_size,
                    "sequence_length": sequence_length,
                    "blend": blend,
                    "confidence_floor": floor,
                    "observation_mode": args.observation_mode,
                    "scope": args.scope,
                }
                specs.append({**payload, "candidate_id": config_hash(payload)})
    unique: dict[str, dict[str, Any]] = {}
    for spec in specs:
        unique.setdefault(spec["candidate_id"], spec)
    return list(unique.values())[: args.max_candidates]


def write_leaderboards(rows: list[dict[str, Any]], out: Path) -> None:
    rows_sorted = sorted(rows, key=lambda row: row["mean_survival"], reverse=True)
    (out / "leaderboard.json").write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
    fieldnames = [
        "mode",
        "candidate_id",
        "config_hash",
        "train_env_steps",
        "supervised_samples",
        "wall_clock_seconds",
        "validation_episodes",
        "mean_survival",
        "median_survival",
        "p10_survival",
        "p90_survival",
        "success_at_500",
        "max_survival",
        "checkpoint_path",
    ]
    with (out / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_failure_taxonomy(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Failure Taxonomy", ""]
    for row in sorted(rows, key=lambda item: item["mean_survival"], reverse=True):
        lines.append(f"## {row['mode']} / {row['candidate_id']}")
        failures = row.get("failure_distribution", {})
        if not failures:
            lines.append("- No failures recorded.")
        for key, value in sorted(failures.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    (out / "failure_taxonomy.md").write_text("\n".join(lines), encoding="utf-8")


def recommended_next(rows: list[dict[str, Any]]) -> str:
    recurrent = [row for row in rows if "mingru" in row["mode"]]
    recurrent_best = max((row["mean_survival"] for row in recurrent), default=0.0)
    if recurrent_best < 100:
        return "reward/obs fix or environment diagnosis"
    if recurrent_best < 475:
        return "better recurrent config or recurrent fine-tuning before more arbitration work"
    if any(row["mode"] == "recon_mingru_terminal_plus_recon_learning" for row in rows):
        return "stabilize ReCoN learning around the recurrent terminal"
    return "longer recurrent training"


def write_summary(rows: list[dict[str, Any]], out: Path, args: argparse.Namespace) -> None:
    rows_sorted = sorted(rows, key=lambda row: row["mean_survival"], reverse=True)
    best = rows_sorted[0] if rows_sorted else {}
    solved = bool(
        best
        and best.get("validation_episodes", 0) >= 300
        and best.get("mean_survival", 0.0) >= 475.0
        and best.get("p10_survival", 0.0) >= 350.0
        and best.get("success_at_500", 0.0) >= 0.70
    )
    pure = [row for row in rows_sorted if row["mode"] == "pure_mingru_policy"]
    recon = [row for row in rows_sorted if row["mode"] == "recon_mingru_terminal_frozen"]
    plus = [row for row in rows_sorted if row["mode"] == "recon_mingru_terminal_plus_recon_learning"]
    lines = [
        "# N=4 Autonomous Recurrent Experiment",
        "",
        f"Status: `{'solved' if solved else 'not solved'}`",
        f"Report directory: `{out}`",
        f"Best candidate: `{best.get('mode', 'none')}` / `{best.get('candidate_id', 'none')}`",
        f"Best checkpoint: `{best.get('checkpoint_path', '')}`",
        f"Best mean/p10/success: `{best.get('mean_survival', 0.0):.1f}` / `{best.get('p10_survival', 0.0):.1f}` / `{best.get('success_at_500', 0.0):.2f}`",
        "",
        "## Answers",
        "",
        f"1. Best N=4 candidate: `{best.get('mode', 'none')}` with candidate `{best.get('candidate_id', 'none')}`.",
        f"2. Pure minGRU learned meaningful behavior: `{'yes' if pure and max(row['mean_survival'] for row in pure) > 100 else 'not yet'}`.",
        f"3. ReCoN-routed minGRU helped or hurt: `{compare_family(pure, recon)}`.",
        f"4. ReCoN learning around minGRU helped: `{compare_family(recon, plus)}`.",
        f"5. Most common failure: `{most_common_failure(best)}`.",
        f"6. Next best move: `{recommended_next(rows_sorted)}`.",
        f"7. Continue toward N=4 solve: `{'yes, but not solved' if best.get('mean_survival', 0.0) >= 250 else 'needs bottleneck fix first'}`.",
        "",
        "## Claim Discipline",
        "",
        "No N=4 solved claim is made unless the held-out threshold is met: >=300 episodes, mean >=475, p10 >=350, success@500 >=0.70.",
        "",
        "## Resume Commands",
        "",
        "```bash",
        f"uv run python scripts/run_n4_autonomous_recurrent.py --dataset-path {out / 'teacher_dataset' / 'dataset.npz'} --out {out}_resume --max-candidates {args.max_candidates}",
        "```",
        "",
        "## Re-evaluate Best Checkpoint",
        "",
        "```bash",
        f"uv run python scripts/train_recurrent_terminal_ladder.py --checkpoints {best.get('checkpoint_path', '')} --n-poles 4 --dynamics-mode {args.dynamics_mode} --dt {args.dt} --discrete-action-bins {args.discrete_action_bins} --validation-episodes 300 --out {out / 'reeval_best'}",
        "```",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_family(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> str:
    if not left or not right:
        return "insufficient evidence"
    lbest = max(row["mean_survival"] for row in left)
    rbest = max(row["mean_survival"] for row in right)
    delta = rbest - lbest
    if abs(delta) < 2:
        return f"neutral ({delta:+.1f} mean steps)"
    return f"helped ({delta:+.1f} mean steps)" if delta > 0 else f"hurt ({delta:+.1f} mean steps)"


def most_common_failure(row: dict[str, Any]) -> str:
    failures = {
        key: value for key, value in row.get("failure_distribution", {}).items() if key != "success"
    }
    if not failures:
        return "none"
    return max(failures.items(), key=lambda item: item[1])[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"reports/n4_autonomous_recurrent_{timestamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_logs").mkdir(exist_ok=True)
    (out / "best_checkpoints").mkdir(exist_ok=True)
    resolved = {"env": env_payload(args), "args": vars(args)}
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    validation_seeds = [args.validation_seed_start + i for i in range(args.validation_episodes)]
    rows = evaluate_baselines(args, validation_seeds, out)
    dataset_path = build_or_reuse_dataset(args, out)

    for spec in candidate_specs(args):
        candidate_dir = out / "candidate_logs" / spec["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path, supervised_samples = train_candidate(
            args,
            dataset_path,
            candidate_dir,
            int(spec["hidden_size"]),
            int(spec["sequence_length"]),
            float(spec["blend"]),
            float(spec["confidence_floor"]),
        )
        config = terminal_config(
            args,
            checkpoint_path,
            int(spec["hidden_size"]),
            int(spec["sequence_length"]),
            float(spec["blend"]),
            float(spec["confidence_floor"]),
        )
        pure = evaluate_pure_mingru(args, validation_seeds, checkpoint_path, config, candidate_dir)
        rows.append(
            row_from_result(
                "pure_mingru_policy",
                spec["candidate_id"],
                checkpoint_path,
                spec,
                supervised_samples,
                supervised_samples,
                pure,
            )
        )
        frozen_controller = ReConCartPoleController(
            RunnerConfig(
                n_poles=args.n_poles,
                mode="recon_mingru_terminal",
                action_mode="discrete",
                discrete_action_bins=args.discrete_action_bins,
                force_mag=args.force_mag,
                selection_mode=args.selection_mode,
                learn=False,
                mingru_terminal=config,
            )
        )
        frozen = evaluate_controller(
            "recon_mingru_terminal_frozen",
            args,
            validation_seeds,
            frozen_controller,
            candidate_dir / "recon_frozen",
        )
        rows.append(
            row_from_result(
                "recon_mingru_terminal_frozen",
                spec["candidate_id"],
                checkpoint_path,
                spec,
                supervised_samples,
                supervised_samples,
                frozen,
            )
        )
        plus_controller = train_recon_learning(args, checkpoint_path, config, candidate_dir)
        plus = evaluate_controller(
            "recon_mingru_terminal_plus_recon_learning",
            args,
            validation_seeds,
            plus_controller,
            candidate_dir / "recon_plus_learning",
        )
        rows.append(
            row_from_result(
                "recon_mingru_terminal_plus_recon_learning",
                spec["candidate_id"],
                checkpoint_path,
                spec,
                supervised_samples + args.recon_learning_train_episodes * args.horizon,
                supervised_samples,
                plus,
            )
        )
        best_checkpoint_dir = out / "best_checkpoints"
        try:
            shutil.copy2(checkpoint_path, best_checkpoint_dir / f"{spec['candidate_id']}_mingru_terminal.pt")
        except OSError:
            pass
        write_leaderboards(rows, out)
        write_failure_taxonomy(rows, out)
        write_summary(rows, out, args)

    write_leaderboards(rows, out)
    write_failure_taxonomy(rows, out)
    write_summary(rows, out, args)
    return {"out": str(out), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--teacher", choices=["heuristic", "static_recon", "recon_policy_terminal"], default="recon_policy_terminal")
    parser.add_argument("--policy-terminal-path", default=CANONICAL_TEACHER)
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--include-policy-terminal", action="store_true", default=True)
    parser.add_argument("--skip-policy-terminal", dest="include_policy_terminal", action="store_false")
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
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw")
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--aggressive-blend", type=float, default=1.0)
    parser.add_argument("--aggressive-confidence-floor", type=float, default=0.01)
    parser.add_argument("--include-prev-force", action="store_true", default=True)
    parser.add_argument("--no-prev-force", dest="include_prev_force", action="store_false")
    parser.add_argument("--include-context", action="store_true", default=True)
    parser.add_argument("--no-context", dest="include_context", action="store_false")
    parser.add_argument("--dataset-episodes", type=int, default=160)
    parser.add_argument("--supervised-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-seed", type=int, default=9117)
    parser.add_argument("--train-seed-start", type=int, default=710000)
    parser.add_argument("--validation-seed-start", type=int, default=820000)
    parser.add_argument("--validation-episodes", type=int, default=80)
    parser.add_argument("--recon-learning-train-episodes", type=int, default=40)
    parser.add_argument("--recon-learning-seed-start", type=int, default=760000)
    parser.add_argument("--failure-window", type=int, default=50)
    parser.add_argument("--x-threshold", type=float, default=2.4)
    parser.add_argument("--theta-threshold", type=float, default=12.0 * 2.0 * np.pi / 360.0)
    parser.add_argument("--velocity-failure-threshold", type=float, default=8.0)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"out": result["out"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
