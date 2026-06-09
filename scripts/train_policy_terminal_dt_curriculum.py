from __future__ import annotations

import argparse
import json
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from train_policy_terminal import train_policy_terminal


def solve_threshold(n_poles: int) -> dict[str, float]:
    if n_poles == 3:
        return {"mean_survival": 475.0, "p10_survival": 400.0, "success_rate": 0.80, "episodes": 300}
    if n_poles == 4:
        return {"mean_survival": 475.0, "p10_survival": 350.0, "success_rate": 0.70, "episodes": 300}
    return {"mean_survival": 475.0, "p10_survival": 350.0, "success_rate": 0.80, "episodes": 300}


def passes(summary: dict[str, Any], threshold: dict[str, float]) -> bool:
    return (
        int(summary.get("episodes", 0)) >= int(threshold["episodes"])
        and float(summary.get("mean_survival", 0.0)) >= threshold["mean_survival"]
        and float(summary.get("p10_survival", 0.0)) >= threshold["p10_survival"]
        and float(summary.get("success_rate", 0.0)) >= threshold["success_rate"]
    )


def stage_args(
    args: argparse.Namespace,
    dt: float,
    index: int,
    out: Path,
    resume_model_path: str = "",
    model_path: str = "",
    eval_episodes: int | None = None,
    eval_seed_start: int | None = None,
) -> Namespace:
    return Namespace(
        n_poles=args.n_poles,
        horizon=args.horizon,
        dt=dt,
        dynamics_mode=args.dynamics_mode,
        action_mode=args.action_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        timesteps=args.timesteps,
        model_path=model_path,
        resume_model_path=resume_model_path,
        train_seed=args.train_seed + index * args.seed_stride,
        hard_train_seeds=args.hard_train_seeds,
        hard_train_seed_probability=args.hard_train_seed_probability,
        eval_seed_start=args.validation_seed_start + index * args.seed_stride if eval_seed_start is None else eval_seed_start,
        eval_episodes=args.validation_episodes if eval_episodes is None else eval_episodes,
        n_envs=args.n_envs,
        device=args.device,
        policy=args.policy,
        net_arch=args.net_arch,
        activation=args.activation,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        reward_mode=args.reward_mode,
        selection_mode=args.selection_mode,
        policy_terminal_blend=args.policy_terminal_blend,
        verbose=args.verbose,
        out=str(out),
    )


def run_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = solve_threshold(args.n_poles)
    stages: list[dict[str, Any]] = []
    resume_model_path = args.resume_model_path

    for index, dt in enumerate(args.dt_values):
        stage_out = out / f"{index:02d}_dt_{dt:g}"
        report = train_policy_terminal(stage_args(args, dt, index, stage_out, resume_model_path=resume_model_path))
        model_path = report["model_path"]
        resume_model_path = model_path
        row = {
            "index": index,
            "dt": dt,
            "status": report["status"],
            "model_path": model_path,
            "report_dir": str(stage_out),
            "pure_ppo_eval": report["pure_ppo_eval"],
            "recon_policy_terminal_eval": report["recon_policy_terminal_eval"],
            "train_timesteps": report["train_timesteps"],
            "train_seed": report["train_seed"],
        }
        stages.append(row)
        partial = {"status": "running", "threshold": threshold, "stages": stages, "current_model_path": resume_model_path}
        (out / "summary.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        write_markdown(partial, out / "summary.md")
        if args.stop_on_unsolved and not passes(report["recon_policy_terminal_eval"], {**threshold, "episodes": args.validation_episodes}):
            break

    final_report = None
    if resume_model_path and args.final_eval_episodes > 0:
        target_dt = args.dt_values[-1]
        final_out = out / "final_eval"
        final_report = train_policy_terminal(
            stage_args(
                args,
                target_dt,
                len(args.dt_values),
                final_out,
                model_path=resume_model_path,
                eval_episodes=args.final_eval_episodes,
                eval_seed_start=args.final_seed_start,
            )
        )
        shutil.copy2(resume_model_path, out / "final_policy_terminal.zip")

    final_recon = final_report["recon_policy_terminal_eval"] if final_report else None
    result = {
        "status": "solved" if final_recon and passes(final_recon, threshold) else "completed_not_solved",
        "threshold": threshold,
        "dt_values": args.dt_values,
        "target_dt": args.dt_values[-1],
        "reward_mode": args.reward_mode,
        "selection_mode": args.selection_mode,
        "policy_terminal_blend": args.policy_terminal_blend,
        "timesteps_per_stage": args.timesteps,
        "validation_episodes": args.validation_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "stages": stages,
        "final_report": final_report,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Policy Terminal dt Curriculum",
        "",
        f"Status: `{result.get('status', 'running')}`",
        f"Reward mode: `{result.get('reward_mode', '')}`",
        f"Selection mode: `{result.get('selection_mode', '')}`",
        f"Policy terminal blend: `{result.get('policy_terminal_blend', '')}`",
        "",
        "| stage | dt | mean | p10 | success | pure PPO mean | model |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result.get("stages", []):
        recon = row["recon_policy_terminal_eval"]
        ppo = row["pure_ppo_eval"]
        href = Path(row["report_dir"]).name + "/report.md"
        lines.append(
            f"| {row['index']} | {row['dt']:g} | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['success_rate']:.2f} | {ppo['mean_survival']:.1f} | [{Path(row['model_path']).name}]({href}) |"
        )
    final = result.get("final_report")
    if final:
        recon = final["recon_policy_terminal_eval"]
        ppo = final["pure_ppo_eval"]
        lines.extend(
            [
                "",
                "## Final Target-dt Eval",
                "",
                "| evaluator | mean | p10 | success | episodes |",
                "|---|---:|---:|---:|---:|",
                f"| pure_ppo | {ppo['mean_survival']:.1f} | {ppo['p10_survival']:.1f} | {ppo['success_rate']:.2f} | {ppo['episodes']} |",
                f"| recon_policy_terminal | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['success_rate']:.2f} | {recon['episodes']} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "This is a curriculum artifact for a learned PPO terminal inside ReCoN. A harder dt is not solved merely because an easier stage is solved; only the final target-dt held-out block supports a solve claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-values", type=float, nargs="+", required=True)
    parser.add_argument("--resume-model-path", default="")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--train-seed", type=int, default=710_000)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=1.0)
    parser.add_argument("--seed-stride", type=int, default=100_000)
    parser.add_argument("--validation-seed-start", type=int, default=720_000)
    parser.add_argument("--validation-episodes", type=int, default=80)
    parser.add_argument("--final-seed-start", type=int, default=980_000)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-mode", choices=["survival", "upright_shaping"], default="upright_shaping")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--stop-on-unsolved", action="store_true")
    parser.add_argument("--out", default="reports/policy_terminal_dt_curriculum")
    args = parser.parse_args()
    result = run_curriculum(args)
    print(json.dumps({"out": args.out, "status": result["status"], "wall_clock_seconds": result["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
