from __future__ import annotations

import argparse
import json
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from train_policy_terminal import train_policy_terminal


def n4_solve_threshold(args: argparse.Namespace) -> dict[str, float]:
    if args.n_poles == 3:
        return {
            "mean_survival": 475.0,
            "p10_survival": 400.0,
            "success_rate": 0.80,
            "episodes": 300,
        }
    if args.n_poles == 4:
        return {
            "mean_survival": 475.0,
            "p10_survival": 350.0,
            "success_rate": 0.70,
            "episodes": 300,
        }
    return {"mean_survival": 475.0, "p10_survival": 350.0, "success_rate": 0.80, "episodes": 300}


def score(summary: dict[str, Any]) -> float:
    return (
        float(summary.get("mean_survival", 0.0))
        + 0.25 * float(summary.get("p10_survival", 0.0))
        + 50.0 * float(summary.get("success_rate", 0.0))
    )


def passes(summary: dict[str, Any], threshold: dict[str, float]) -> bool:
    return (
        int(summary.get("episodes", 0)) >= int(threshold["episodes"])
        and float(summary.get("mean_survival", 0.0)) >= threshold["mean_survival"]
        and float(summary.get("p10_survival", 0.0)) >= threshold["p10_survival"]
        and float(summary.get("success_rate", 0.0)) >= threshold["success_rate"]
    )


def candidate_args(
    args: argparse.Namespace, seed: int, out: Path, model_path: str = ""
) -> Namespace:
    return Namespace(
        n_poles=args.n_poles,
        horizon=args.horizon,
        dt=args.dt,
        dynamics_mode=args.dynamics_mode,
        action_mode=args.action_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        timesteps=args.timesteps,
        model_path=model_path,
        resume_model_path="",
        train_seed=seed,
        hard_train_seeds=args.hard_train_seeds,
        hard_train_seed_probability=args.hard_train_seed_probability,
        eval_seed_start=args.validation_seed_start,
        eval_episodes=args.validation_episodes,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
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
        policy_terminal_scope=args.policy_terminal_scope,
        frame_stack=args.frame_stack,
        policy_observation_mode=args.policy_observation_mode,
        verbose=args.verbose,
        out=str(out),
    )


def final_eval_args(args: argparse.Namespace, model_path: str, out: Path) -> Namespace:
    ns = candidate_args(args, args.train_seeds[0], out, model_path=model_path)
    ns.timesteps = 0
    ns.eval_seed_start = args.final_seed_start
    ns.eval_episodes = args.final_eval_episodes
    return ns


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = n4_solve_threshold(args)
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = float("-inf")

    for index, seed in enumerate(args.train_seeds):
        c_out = out / f"candidate_{index:02d}_seed_{seed}"
        report = train_policy_terminal(candidate_args(args, seed, c_out))
        recon = report["recon_policy_terminal_eval"]
        row = {
            "index": index,
            "train_seed": seed,
            "out": str(c_out),
            "model_path": report["model_path"],
            "validation": recon,
            "pure_ppo_validation": report["pure_ppo_eval"],
            "score": score(recon),
            "wall_clock_seconds": report["wall_clock_seconds"],
        }
        candidates.append(row)
        if row["score"] > best_score:
            best_score = float(row["score"])
            best = row
        (out / "summary.json").write_text(
            json.dumps({"candidates": candidates, "best": best}, indent=2), encoding="utf-8"
        )
        write_markdown(
            {"status": "running", "threshold": threshold, "candidates": candidates, "best": best},
            out / "summary.md",
        )

    final_report = None
    if best is not None and args.final_eval_episodes > 0:
        final_out = out / "final_eval"
        final_report = train_policy_terminal(final_eval_args(args, best["model_path"], final_out))
        best_model_src = Path(best["model_path"])
        if best_model_src.exists():
            shutil.copy2(best_model_src, out / "best_policy_terminal.zip")

    final_recon = final_report["recon_policy_terminal_eval"] if final_report else None
    result = {
        "status": "solved"
        if final_recon and passes(final_recon, threshold)
        else "completed_not_solved",
        "threshold": threshold,
        "env": {
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
        },
        "reward_mode": args.reward_mode,
        "selection_mode": args.selection_mode,
        "policy_terminal_blend": args.policy_terminal_blend,
        "policy_terminal_scope": args.policy_terminal_scope,
        "frame_stack": args.frame_stack,
        "policy_observation_mode": args.policy_observation_mode,
        "ppo_config": {
            "policy": args.policy,
            "net_arch": args.net_arch,
            "activation": args.activation,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "frame_stack": args.frame_stack,
            "policy_observation_mode": args.policy_observation_mode,
            "vec_env": args.vec_env,
        },
        "timesteps_per_candidate": args.timesteps,
        "validation_episodes": args.validation_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "candidates": candidates,
        "best": best,
        "final_report": final_report,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Policy Terminal Sweep",
        "",
        f"Status: `{result.get('status', 'running')}`",
        f"Reward mode: `{result.get('reward_mode', '')}`",
        f"Selection mode: `{result.get('selection_mode', '')}`",
        f"Policy terminal blend: `{result.get('policy_terminal_blend', '')}`",
        f"Policy terminal scope: `{result.get('policy_terminal_scope', 'stabilize_chain')}`",
        f"Frame stack: `{result.get('frame_stack', 1)}`",
        f"Policy observation mode: `{result.get('policy_observation_mode', 'env')}`",
        "",
        "| candidate | train seed | score | mean | p10 | success | pure PPO mean | report |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result.get("candidates", []):
        recon = row["validation"]
        ppo = row["pure_ppo_validation"]
        href = Path(row["out"]).name + "/report.md"
        lines.append(
            f"| {row['index']} | {row['train_seed']} | {row['score']:.1f} | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['success_rate']:.2f} | {ppo['mean_survival']:.1f} | [{href}]({href}) |"
        )
    best = result.get("best")
    if best:
        lines.extend(["", f"Best validation model: `{best['model_path']}`"])
    final = result.get("final_report")
    if final:
        recon = final["recon_policy_terminal_eval"]
        ppo = final["pure_ppo_eval"]
        lines.extend(
            [
                "",
                "## Final Held-Out Eval",
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
            "This is a model-selection artifact for a learned PPO terminal inside ReCoN. It is not pure symbolic ReCoN, and N=4 is solved only if the final held-out block meets the configured threshold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=1.0)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel"
    )
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--validation-seed-start", type=int, default=970_000)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--final-seed-start", type=int, default=980_000)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
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
    parser.add_argument(
        "--reward-mode", choices=["survival", "upright_shaping"], default="upright_shaping"
    )
    parser.add_argument(
        "--selection-mode", choices=["soft_select", "hard_select"], default="hard_select"
    )
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument(
        "--policy-terminal-scope",
        choices=["stabilize_chain", "selected", "all"],
        default="stabilize_chain",
        help="Which ReCoN proposals can be force-blended with the PPO terminal.",
    )
    parser.add_argument(
        "--policy-observation-mode", choices=["env", "normalized_raw"], default="env"
    )
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/policy_terminal_sweep")
    args = parser.parse_args()
    result = run_sweep(args)
    print(
        json.dumps(
            {
                "out": args.out,
                "status": result["status"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
