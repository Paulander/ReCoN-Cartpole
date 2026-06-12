from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_policy_dataset import collect as collect_policy_dataset  # noqa: E402
from train_mingru_supervised import train as train_mingru  # noqa: E402
from train_recurrent_terminal_ladder import evaluate_pure_mingru, evaluate_recon_mingru  # noqa: E402


ARRAY_KEYS = [
    "observations",
    "prev_forces",
    "teacher_forces",
    "teacher_actions",
    "returns_to_go",
    "failure_within_k",
    "seeds",
    "sources",
    "rollout_sources",
    "rollout_forces",
    "rollout_actions",
    "motif_scores",
    "episodes",
    "step_indices",
    "sample_weights",
]


def default_stages(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "name": "n3_stable",
            "teacher": "static_recon",
            "n_poles": 3,
            "episodes": args.n3_episodes,
            "seed_start": args.n3_seed_start,
            "initial_angle_range": min(float(args.initial_angle_range), 0.03),
            "force_noise": min(float(args.force_noise), 0.01),
            "rollout_policy": "teacher",
            "label_source": "teacher",
            "sample_weight": args.n3_sample_weight,
        },
        {
            "name": "n4_low_angle_no_noise",
            "teacher": "recon_policy_terminal",
            "n_poles": 4,
            "episodes": args.low_angle_episodes,
            "seed_start": args.low_angle_seed_start,
            "initial_angle_range": 0.02,
            "force_noise": 0.0,
            "rollout_policy": "teacher",
            "label_source": "teacher",
            "sample_weight": args.low_angle_sample_weight,
        },
        {
            "name": "n4_current",
            "teacher": "recon_policy_terminal",
            "n_poles": 4,
            "episodes": args.current_episodes,
            "seed_start": args.current_seed_start,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "rollout_policy": "teacher",
            "label_source": "teacher",
            "sample_weight": args.current_sample_weight,
        },
        {
            "name": "n4_hard_tail",
            "teacher": "recon_policy_terminal",
            "n_poles": 4,
            "episodes": args.tail_episodes,
            "seed_start": args.tail_seed_start,
            "seed_list": args.tail_seed_list,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "rollout_policy": "mingru_terminal" if args.behavior_checkpoint_path else "teacher",
            "label_source": "teacher",
            "sample_weight": args.tail_sample_weight,
        },
    ]


def dataset_args(args: argparse.Namespace, stage: dict[str, Any], out_path: Path) -> Namespace:
    return Namespace(
        teacher=stage["teacher"],
        n_poles=int(stage["n_poles"]),
        horizon=args.horizon,
        episodes=int(stage["episodes"]),
        seed_start=int(stage.get("seed_start", 0)),
        seed_list=str(stage.get("seed_list", "") or ""),
        dt=args.dt,
        dynamics_mode=args.dynamics_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=float(stage["initial_angle_range"]),
        force_noise=float(stage["force_noise"]),
        link_coupling=args.link_coupling,
        selection_mode=args.selection_mode,
        observation_mode=args.observation_mode,
        teacher_observation_mode=args.teacher_observation_mode,
        policy_terminal_path=args.policy_terminal_path if stage["teacher"] == "recon_policy_terminal" else "",
        policy_terminal_blend=args.policy_terminal_blend,
        policy_terminal_scope=args.policy_terminal_scope,
        rollout_policy=stage.get("rollout_policy", "teacher"),
        label_source=stage.get("label_source", "teacher"),
        behavior_checkpoint_path=args.behavior_checkpoint_path,
        behavior_hidden_size=args.behavior_hidden_size,
        behavior_sequence_length=args.behavior_sequence_length,
        behavior_observation_mode=args.behavior_observation_mode,
        behavior_include_prev_force=args.behavior_include_prev_force,
        behavior_include_context=args.behavior_include_context,
        behavior_confidence_floor=args.behavior_confidence_floor,
        failure_window=args.failure_window,
        motif_model_path=args.motif_model_path,
        out=str(out_path),
    )


def collect_stage(args: argparse.Namespace, stage: dict[str, Any], out_dir: Path, index: int) -> dict[str, Any]:
    stage_dir = out_dir / f"{index:02d}_{stage['name']}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    data = collect_policy_dataset(dataset_args(args, stage, stage_dir / "dataset.npz"))
    sample_weight = max(0.0, float(stage.get("sample_weight", 1.0)))
    data = {**data, "sample_weights": np.full(data["teacher_actions"].shape[0], sample_weight, dtype=np.float32)}
    np.savez_compressed(stage_dir / "dataset.npz", **data)
    report = {
        "name": stage["name"],
        "n_poles": int(stage["n_poles"]),
        "teacher": stage["teacher"],
        "rollout_policy": stage.get("rollout_policy", "teacher"),
        "label_source": stage.get("label_source", "teacher"),
        "episodes": int(stage["episodes"]),
        "seed_start": int(stage.get("seed_start", 0)),
        "seed_list": str(stage.get("seed_list", "") or ""),
        "initial_angle_range": float(stage["initial_angle_range"]),
        "force_noise": float(stage["force_noise"]),
        "samples": int(data["observations"].shape[0]),
        "sample_weight": sample_weight,
        "dataset": str(stage_dir / "dataset.npz"),
    }
    (stage_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"metadata": report, "data": data}


def aggregate_stage_data(stages: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not stages:
        raise ValueError("no stages to aggregate")
    result: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    episode_offset = 0
    for item in stages:
        data = item["data"]
        stage_episodes = np.asarray(data["episodes"], dtype=np.int64)
        for key in ARRAY_KEYS:
            if key == "motif_scores" and key not in data:
                arr = np.zeros(np.asarray(data["observations"]).shape[0], dtype=np.float32)
            else:
                arr = np.asarray(data[key])
            if key == "episodes":
                arr = arr.astype(np.int64) + episode_offset
            result[key].append(arr)
        if stage_episodes.size:
            episode_offset += int(np.max(stage_episodes)) + 1
    return {key: np.concatenate(parts, axis=0) for key, parts in result.items()}


def eval_seeds(args: argparse.Namespace) -> list[int]:
    seeds: list[int] = []
    for start in args.final_seed_starts:
        seeds.extend(int(start) + idx for idx in range(int(args.final_eval_episodes)))
    return seeds


def eval_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        n_poles=4,
        horizon=args.horizon,
        dt=args.dt,
        dynamics_mode=args.dynamics_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        blend=args.blend,
        scope=args.policy_terminal_scope,
        confidence_floor=args.confidence_floor,
        include_motif_score=args.include_motif_score,
        motif_model_path=args.motif_model_path,
        motif_score_scale=args.motif_score_scale,
        passthrough_enabled=args.passthrough_enabled,
        passthrough_confidence_floor=args.passthrough_confidence_floor,
        passthrough_logit_margin_floor=args.passthrough_logit_margin_floor,
        selection_mode=args.selection_mode,
    )


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# minGRU Recurrent Curriculum",
        "",
        f"Status: `{result['status']}`",
        f"Observation mode: `{result['observation_mode']}`",
        f"Sequence length: `{result['sequence_length']}`",
        "",
        "| stage | n | teacher | rollout | angle | noise | weight | samples |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for stage in result["stages"]:
        lines.append(
            f"| {stage['name']} | {stage['n_poles']} | {stage['teacher']} | {stage['rollout_policy']} | "
            f"{stage['initial_angle_range']:.3f} | {stage['force_noise']:.3f} | {stage.get('sample_weight', 1.0):.3f} | {stage['samples']} |"
        )
    pure = result.get("pure_mingru_policy", {})
    recon = result.get("recon_mingru_terminal", {})
    eval_status = result.get("eval_status", "completed")
    lines.extend([
        "",
        "## Held-Out N=4 Eval",
        "",
        f"Eval status: `{eval_status}`",
        "",
        "| evaluator | mean | p10 | success | episodes |",
        "|---|---:|---:|---:|---:|",
        f"| pure_mingru_policy | {pure.get('mean_survival', 0.0):.1f} | {pure.get('p10_survival', 0.0):.1f} | {pure.get('success_rate', 0.0):.3f} | {pure.get('episodes', 0)} |",
        f"| recon_mingru_terminal | {recon.get('mean_survival', 0.0):.1f} | {recon.get('p10_survival', 0.0):.1f} | {recon.get('success_rate', 0.0):.3f} | {recon.get('episodes', 0)} |",
        "",
        "## Claim Discipline",
        "",
        "This is a recurrent curriculum experiment. N=3 and low-angle stages are training curriculum only; solve claims require held-out N=4 metrics.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_payloads = [collect_stage(args, stage, out, idx) for idx, stage in enumerate(default_stages(args))]
    aggregated = aggregate_stage_data(stage_payloads)
    dataset_path = out / "curriculum_dataset.npz"
    np.savez_compressed(dataset_path, **aggregated)
    train_report = train_mingru(
        Namespace(
            dataset=str(dataset_path),
            resume_checkpoint=args.resume_checkpoint,
            resume_partial_input=args.resume_partial_input,
            out=str(out / "supervised_mingru"),
            n_poles=4,
            horizon=args.horizon,
            force_mag=args.force_mag,
            discrete_action_bins=args.discrete_action_bins,
            observation_mode=args.observation_mode,
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            include_prev_force=args.include_prev_force,
            include_context=args.include_context,
            include_motif_score=args.include_motif_score,
            motif_model_path=args.motif_model_path,
            motif_score_scale=args.motif_score_scale,
            scope=args.policy_terminal_scope,
            blend=args.blend,
            confidence_floor=args.confidence_floor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            value_weight=args.value_weight,
            failure_weight=args.failure_weight,
            confidence_weight=args.confidence_weight,
            min_sample_episode_survival=0.0,
            max_sample_episode_survival=0.0,
            failure_sample_weight=args.failure_sample_weight,
            late_sample_weight=args.late_sample_weight,
            low_return_sample_weight=args.low_return_sample_weight,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            seed=args.train_seed,
        )
    )
    seeds = eval_seeds(args)
    ladder_args = eval_args(args)
    checkpoint = str(train_report["checkpoint_path"])
    if seeds:
        pure = evaluate_pure_mingru(checkpoint, ladder_args, seeds, args.hidden_size, args.sequence_length)
        recon = evaluate_recon_mingru(checkpoint, ladder_args, seeds, args.hidden_size, args.sequence_length)
        eval_status = "completed"
    else:
        pure = {"episodes": 0, "mean_survival": 0.0, "p10_survival": 0.0, "success_rate": 0.0, "max_survival": 0.0}
        recon = dict(pure)
        eval_status = "skipped_no_eval_seeds"
    result = {
        "status": "completed",
        "out": str(out),
        "dataset": str(dataset_path),
        "samples": int(aggregated["observations"].shape[0]),
        "checkpoint_path": checkpoint,
        "observation_mode": args.observation_mode,
        "sequence_length": int(args.sequence_length),
        "hidden_size": int(args.hidden_size),
        "stages": [item["metadata"] for item in stage_payloads],
        "train_report": train_report,
        "eval_seeds": seeds,
        "eval_status": eval_status,
        "pure_mingru_policy": pure,
        "recon_mingru_terminal": recon,
        "mechanisms": {
            "minGRU_terminal": True,
            "curriculum_data": True,
            "n3_to_n4_curriculum": True,
            "previous_force_observation": "prev_force" in args.observation_mode or bool(args.include_prev_force),
            "motif_score_observation": bool(args.include_motif_score),
            "motif_model_path": str(args.motif_model_path),
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minGRU terminal from an explicit N3->N4 curriculum dataset.")
    parser.add_argument("--policy-terminal-path", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-partial-input", action="store_true", default=False)
    parser.add_argument("--behavior-checkpoint-path", default="")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--n3-episodes", type=int, default=40)
    parser.add_argument("--low-angle-episodes", type=int, default=40)
    parser.add_argument("--current-episodes", type=int, default=60)
    parser.add_argument("--tail-episodes", type=int, default=60)
    parser.add_argument("--n3-seed-start", type=int, default=2_810_000)
    parser.add_argument("--low-angle-seed-start", type=int, default=2_910_000)
    parser.add_argument("--current-seed-start", type=int, default=3_010_000)
    parser.add_argument("--tail-seed-start", type=int, default=3_110_000)
    parser.add_argument("--tail-seed-list", default="")
    parser.add_argument("--n3-sample-weight", type=float, default=1.0)
    parser.add_argument("--low-angle-sample-weight", type=float, default=1.0)
    parser.add_argument("--current-sample-weight", type=float, default=1.0)
    parser.add_argument("--tail-sample-weight", type=float, default=1.0)
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw4_prev_force")
    parser.add_argument("--teacher-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--behavior-hidden-size", type=int, default=256)
    parser.add_argument("--behavior-sequence-length", type=int, default=32)
    parser.add_argument("--behavior-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw4_prev_force")
    parser.add_argument("--behavior-include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--behavior-include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--behavior-confidence-floor", type=float, default=0.05)
    parser.add_argument("--failure-window", type=int, default=80)
    parser.add_argument("--include-motif-score", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motif-model-path", default="")
    parser.add_argument("--motif-score-scale", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-logit-margin-floor", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.05)
    parser.add_argument("--failure-weight", type=float, default=0.10)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--failure-sample-weight", type=float, default=0.5)
    parser.add_argument("--late-sample-weight", type=float, default=0.5)
    parser.add_argument("--low-return-sample-weight", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=2810)
    parser.add_argument("--final-seed-starts", type=int, nargs="+", default=[1_900_000, 2_000_000, 2_100_000, 2_200_000])
    parser.add_argument("--final-eval-episodes", type=int, default=60)
    parser.add_argument("--out", default="reports/mingru_curriculum")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"out": result["out"], "checkpoint_path": result["checkpoint_path"], "success": result["recon_mingru_terminal"].get("success_rate", 0.0)}, indent=2))


if __name__ == "__main__":
    main()
