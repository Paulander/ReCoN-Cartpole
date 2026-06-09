from __future__ import annotations

import argparse
import json
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from train_policy_terminal import (
    evaluate_model,
    evaluate_recon_terminal,
    hard_train_seeds,
    make_env,
    ppo_kwargs,
)


def solve_threshold(n_poles: int) -> dict[str, float]:
    if n_poles == 3:
        return {
            "mean_survival": 475.0,
            "p10_survival": 400.0,
            "success_rate": 0.80,
            "episodes": 300,
        }
    if n_poles == 4:
        return {
            "mean_survival": 475.0,
            "p10_survival": 350.0,
            "success_rate": 0.70,
            "episodes": 300,
        }
    return {"mean_survival": 475.0, "p10_survival": 350.0, "success_rate": 0.80, "episodes": 300}


def score(
    summary: dict[str, Any],
    *,
    mean_weight: float = 1.0,
    p10_weight: float = 0.25,
    success_weight: float = 50.0,
) -> float:
    return (
        mean_weight * float(summary.get("mean_survival", 0.0))
        + p10_weight * float(summary.get("p10_survival", 0.0))
        + success_weight * float(summary.get("success_rate", 0.0))
    )


def passes(summary: dict[str, Any], threshold: dict[str, float]) -> bool:
    return (
        int(summary.get("episodes", 0)) >= int(threshold["episodes"])
        and float(summary.get("mean_survival", 0.0)) >= threshold["mean_survival"]
        and float(summary.get("p10_survival", 0.0)) >= threshold["p10_survival"]
        and float(summary.get("success_rate", 0.0)) >= threshold["success_rate"]
    )


def eval_args(args: argparse.Namespace, eval_seed_start: int, eval_episodes: int) -> Namespace:
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
        selection_mode=args.selection_mode,
        policy_terminal_blend=args.policy_terminal_blend,
        policy_terminal_scope=args.policy_terminal_scope,
        frame_stack=args.frame_stack,
        policy_observation_mode=args.policy_observation_mode,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        reward_mode=args.reward_mode,
        eval_seed_start=eval_seed_start,
        eval_episodes=eval_episodes,
    )


def validation_seeds(args: argparse.Namespace) -> list[int]:
    starts = args.validation_seed_starts or [args.validation_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(start + idx for idx in range(args.validation_episodes))
    return seeds


def final_seeds(args: argparse.Namespace) -> list[int]:
    return [args.final_seed_start + idx for idx in range(args.final_eval_episodes)]


def make_training_env(args: argparse.Namespace):
    return make_env(args, reward_mode=args.reward_mode, use_hard_seeds=True)


def record_checkpoint(
    args: argparse.Namespace,
    out: Path,
    checkpoint_path: Path,
    total_timesteps: int,
    label: str,
) -> dict[str, Any]:
    v_args = eval_args(args, args.validation_seed_start, args.validation_episodes)
    seeds = validation_seeds(args)
    summary = evaluate_recon_terminal(checkpoint_path, v_args, seeds)
    row = {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "total_timesteps": total_timesteps,
        "validation": summary,
        "score": score(
            summary,
            mean_weight=args.score_mean_weight,
            p10_weight=args.score_p10_weight,
            success_weight=args.score_success_weight,
        ),
        "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
        "validation_seed_count": len(seeds),
    }
    (out / "latest_validation.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Iterative Policy Terminal Training",
        "",
        f"Status: `{result.get('status', 'running')}`",
        f"Reward mode: `{result.get('reward_mode', '')}`",
        f"Selection mode: `{result.get('selection_mode', '')}`",
        f"Policy terminal blend: `{result.get('policy_terminal_blend', '')}`",
        f"Policy terminal scope: `{result.get('policy_terminal_scope', 'stabilize_chain')}`",
        f"Frame stack: `{result.get('frame_stack', 1)}`",
        f"Policy observation mode: `{result.get('policy_observation_mode', 'env')}`",
        f"Success bonus: `{result.get('success_bonus', 0.0)}`",
        f"Failure penalty: `{result.get('failure_penalty', 0.0)}`",
        f"Hard train seeds: `{result.get('hard_train_seed_count', 0)}` at probability `{result.get('hard_train_seed_probability', 1.0)}`",
        f"Validation seed starts: `{', '.join(str(seed) for seed in result.get('validation_seed_starts', []))}`",
        f"Validation seed count per start: `{result.get('validation_episodes', '')}`",
        f"Score weights: mean `{result.get('score_weights', {}).get('mean_survival', '')}`, p10 `{result.get('score_weights', {}).get('p10_survival', '')}`, success `{result.get('score_weights', {}).get('success_rate', '')}`",
        "",
        "| checkpoint | timesteps | score | mean | p10 | success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("history", []):
        val = row["validation"]
        lines.append(
            f"| {row['label']} | {row['total_timesteps']} | {row['score']:.1f} | {val['mean_survival']:.1f} | {val['p10_survival']:.1f} | {val['success_rate']:.2f} |"
        )
    best = result.get("best")
    if best:
        lines.extend(["", f"Best validation checkpoint: `{best['checkpoint']}`"])
    final = result.get("final_eval")
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
            "This runner promotes checkpoints by ReCoN-routed validation survival. It is still a learned PPO terminal inside ReCoN, not pure symbolic ReCoN. N=4 is solved only if the final held-out block meets the configured threshold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(out: Path, result: dict[str, Any]) -> None:
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")


def run_iterative(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = solve_threshold(args.n_poles)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Install RL extras with `uv sync --extra rl` to train policy terminals"
        ) from exc

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else None
    train_env = make_vec_env(
        lambda: make_training_env(args),
        n_envs=args.n_envs,
        seed=args.train_seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs={"start_method": "fork"} if vec_env_cls is SubprocVecEnv else None,
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    total_timesteps = 0

    if args.start_model_path:
        model = PPO.load(str(args.start_model_path), env=train_env, device=args.device)
        model.set_random_seed(args.train_seed)
        start_path = out / "checkpoint_000000_start.zip"
        shutil.copy2(args.start_model_path, start_path)
        row = record_checkpoint(args, out, start_path, total_timesteps, "start")
        history.append(row)
        best = row
        best_score = float(row["score"])
    else:
        model = PPO(
            args.policy,
            train_env,
            seed=args.train_seed,
            verbose=args.verbose,
            device=args.device,
            **ppo_kwargs(args),
        )

    result: dict[str, Any] = {
        "status": "running",
        "threshold": threshold,
        "reward_mode": args.reward_mode,
        "selection_mode": args.selection_mode,
        "policy_terminal_blend": args.policy_terminal_blend,
        "policy_terminal_scope": args.policy_terminal_scope,
        "frame_stack": args.frame_stack,
        "policy_observation_mode": args.policy_observation_mode,
        "success_bonus": args.success_bonus,
        "failure_penalty": args.failure_penalty,
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
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "vec_env": args.vec_env,
        },
        "chunk_timesteps": args.chunk_timesteps,
        "chunks": args.chunks,
        "hard_train_seed_count": len(hard_train_seeds(args)),
        "hard_train_seed_probability": args.hard_train_seed_probability,
        "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
        "validation_episodes": args.validation_episodes,
        "validation_seed_count": len(validation_seeds(args)),
        "final_eval_episodes": args.final_eval_episodes,
        "score_weights": {
            "mean_survival": args.score_mean_weight,
            "p10_survival": args.score_p10_weight,
            "success_rate": args.score_success_weight,
        },
        "history": history,
        "best": best,
    }
    save_summary(out, result)

    for chunk in range(1, args.chunks + 1):
        model.learn(
            total_timesteps=args.chunk_timesteps,
            reset_num_timesteps=(chunk == 1 and not args.start_model_path),
        )
        total_timesteps += args.chunk_timesteps
        checkpoint = out / f"checkpoint_{total_timesteps:06d}.zip"
        model.save(str(checkpoint))
        row = record_checkpoint(args, out, checkpoint, total_timesteps, f"chunk_{chunk}")
        history.append(row)
        if row["score"] > best_score:
            best_score = float(row["score"])
            best = row
            shutil.copy2(checkpoint, out / "best_policy_terminal.zip")
        result.update({"history": history, "best": best})
        save_summary(out, result)

    final_eval = None
    if best is not None and args.final_eval_episodes > 0:
        best_path = Path(best["checkpoint"])
        final_args = eval_args(args, args.final_seed_start, args.final_eval_episodes)
        model = PPO.load(str(best_path), device=args.device)
        seeds = final_seeds(args)
        final_eval = {
            "checkpoint": str(best_path),
            "pure_ppo_eval": evaluate_model(model, final_args, seeds),
            "recon_policy_terminal_eval": evaluate_recon_terminal(
                best_path,
                final_args,
                seeds,
                trace_seed=args.final_seed_start + 999_999,
                out_dir=out,
            ),
        }

    final_recon = final_eval["recon_policy_terminal_eval"] if final_eval else None
    result.update(
        {
            "status": "solved"
            if final_recon and passes(final_recon, threshold)
            else "completed_not_solved",
            "history": history,
            "best": best,
            "final_eval": final_eval,
            "wall_clock_seconds": time.perf_counter() - started,
        }
    )
    save_summary(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-model-path", default="")
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
    parser.add_argument("--chunk-timesteps", type=int, default=25_000)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=610_000)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=1.0)
    parser.add_argument("--validation-seed-start", type=int, default=970_000)
    parser.add_argument(
        "--validation-seed-starts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of validation seed block starts. Each start contributes "
            "--validation-episodes consecutive seeds; useful for less lucky checkpoint selection."
        ),
    )
    parser.add_argument("--success-bonus", type=float, default=0.0)
    parser.add_argument("--failure-penalty", type=float, default=0.0)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--score-mean-weight", type=float, default=1.0)
    parser.add_argument("--score-p10-weight", type=float, default=0.25)
    parser.add_argument("--score-success-weight", type=float, default=50.0)
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
    parser.add_argument("--out", default="reports/policy_terminal_iterative")
    args = parser.parse_args()
    result = run_iterative(args)
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
