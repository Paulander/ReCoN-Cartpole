from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_policy_terminal import (  # noqa: E402
    copy_vec_normalize_stats,
    evaluate_model,
    export_vec_normalize_stats,
    hard_train_seeds,
    make_env,
    ppo_kwargs,
)
from train_policy_terminal_iterative import eval_args, final_seeds, passes, solve_threshold  # noqa: E402


def tail_metrics(steps: list[float], horizon: int, cvar_fraction: float = 0.10) -> dict[str, float]:
    summary = summarize_steps(steps, horizon)
    values = np.asarray(steps, dtype=float)
    if values.size == 0:
        summary.update({"cvar_survival": 0.0, "bottom_count": 0.0, "median_survival": 0.0})
        return summary
    count = max(1, int(np.ceil(values.size * max(0.0, min(1.0, cvar_fraction)))))
    tail = np.sort(values)[:count]
    summary.update(
        {
            "cvar_survival": float(np.mean(tail)),
            "bottom_count": float(count),
            "median_survival": float(np.median(values)),
        }
    )
    return summary


def tail_score(
    summary: dict[str, Any],
    *,
    mean_weight: float = 0.35,
    p10_weight: float = 0.75,
    cvar_weight: float = 0.75,
    success_weight: float = 130.0,
) -> float:
    return (
        mean_weight * float(summary.get("mean_survival", 0.0))
        + p10_weight * float(summary.get("p10_survival", 0.0))
        + cvar_weight * float(summary.get("cvar_survival", 0.0))
        + success_weight * float(summary.get("success_rate", 0.0))
    )


def validation_seeds(args: argparse.Namespace) -> list[int]:
    starts = args.validation_seed_starts or [args.validation_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(start + idx for idx in range(args.validation_episodes))
    return seeds


def evaluate_recon_terminal_tail(
    model_path: Path,
    args: argparse.Namespace,
    seeds: list[int],
    *,
    cvar_fraction: float,
) -> dict[str, Any]:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_policy_terminal",
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=str(model_path),
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_frame_stack=int(getattr(args, "frame_stack", 1)),
            policy_terminal_scope=str(getattr(args, "policy_terminal_scope", "stabilize_chain")),
            policy_terminal_observation_mode=str(getattr(args, "policy_observation_mode", "env")),
            policy_terminal_recurrent=bool(getattr(args, "policy_terminal_recurrent", False)),
            policy_terminal_normalizer_path=str(getattr(args, "policy_terminal_normalizer_path", "")),
        )
    )
    steps: list[float] = []
    returns: list[float] = []
    per_seed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        result = rollout(
            make_env(
                args,
                reward_mode="survival",
                use_frame_stack=False,
                use_success_bonus=False,
                use_failure_penalty=False,
            ),
            controller,
            seed=seed,
            horizon=args.horizon,
            trace=False,
        )
        step_count = float(result["steps"])
        total_return = float(result["return"])
        steps.append(step_count)
        returns.append(total_return)
        per_seed.append(
            {
                "seed": int(seed),
                "steps": int(step_count),
                "return": total_return,
                "success": step_count >= args.horizon,
            }
        )
    summary = tail_metrics(steps, args.horizon, cvar_fraction)
    summary.update(
        {
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "episodes": len(seeds),
            "wall_clock_seconds": time.perf_counter() - started,
            "per_seed": per_seed,
        }
    )
    return summary


def tail_seed_pool(summary: dict[str, Any], limit: int, min_steps: int) -> list[int]:
    rows = sorted(
        (
            row
            for row in summary.get("per_seed", [])
            if int(row.get("steps", 0)) >= min_steps and not bool(row.get("success", False))
        ),
        key=lambda row: int(row.get("steps", 0)),
        reverse=True,
    )
    return [int(row["seed"]) for row in rows[: max(0, int(limit))]]


def write_seed_file(path: Path, seeds: list[int]) -> None:
    path.write_text(json.dumps({"hard_seeds": [int(seed) for seed in seeds]}, indent=2), encoding="utf-8")


def merge_seed_pool(base: list[int], extra: list[int], limit: int) -> list[int]:
    seen: set[int] = set()
    merged: list[int] = []
    for seed in list(extra) + list(base):
        seed = int(seed)
        if seed not in seen:
            seen.add(seed)
            merged.append(seed)
    return merged[: max(0, int(limit))]


def make_train_env(args: argparse.Namespace, seed_file: Path):
    train_args = Namespace(**vars(args))
    train_args.hard_train_seeds = str(seed_file)
    return make_env(train_args, reward_mode=args.reward_mode, use_hard_seeds=True)


def record_checkpoint(
    args: argparse.Namespace,
    out: Path,
    checkpoint_path: Path,
    total_timesteps: int,
    label: str,
    normalizer_path: str = "",
) -> dict[str, Any]:
    seeds = validation_seeds(args)
    eval_config = eval_args(args, args.validation_seed_start, args.validation_episodes)
    eval_config.policy_terminal_normalizer_path = normalizer_path
    summary = evaluate_recon_terminal_tail(
        checkpoint_path,
        eval_config,
        seeds,
        cvar_fraction=args.cvar_fraction,
    )
    row = {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "normalizer_path": str(normalizer_path),
        "total_timesteps": int(total_timesteps),
        "validation": summary,
        "score": tail_score(
            summary,
            mean_weight=args.score_mean_weight,
            p10_weight=args.score_p10_weight,
            cvar_weight=args.score_cvar_weight,
            success_weight=args.score_success_weight,
        ),
        "tail_seed_candidates": tail_seed_pool(
            summary, args.tail_seed_refresh_count, args.tail_seed_min_steps
        ),
        "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
        "validation_seed_count": len(seeds),
    }
    (out / "latest_validation.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def should_promote(row: dict[str, Any], best: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if best is None:
        return True
    validation = row["validation"]
    best_validation = best["validation"]
    if float(validation["success_rate"]) + args.max_success_regression < float(
        best_validation["success_rate"]
    ):
        return False
    if float(validation["p10_survival"]) + args.max_p10_regression < float(
        best_validation["p10_survival"]
    ):
        return False
    return float(row["score"]) > float(best["score"])


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tail-First Policy Terminal Curriculum",
        "",
        f"Status: `{result.get('status', 'running')}`",
        f"Reward mode: `{result.get('reward_mode', '')}`",
        f"Selection mode: `{result.get('selection_mode', '')}`",
        f"Policy observation mode: `{result.get('policy_observation_mode', 'env')}`",
        f"Hard seed probability: `{result.get('hard_train_seed_probability', 0.0)}`",
        f"Adaptive tail seed refresh: `{result.get('tail_seed_refresh_count', 0)}` seeds/chunk",
        f"Validation seed starts: `{', '.join(str(seed) for seed in result.get('validation_seed_starts', []))}`",
        f"Validation episodes per start: `{result.get('validation_episodes', '')}`",
        f"Score weights: mean `{result.get('score_weights', {}).get('mean_survival', '')}`, p10 `{result.get('score_weights', {}).get('p10_survival', '')}`, CVaR `{result.get('score_weights', {}).get('cvar_survival', '')}`, success `{result.get('score_weights', {}).get('success_rate', '')}`",
        "",
        "| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("history", []):
        val = row["validation"]
        lines.append(
            f"| {row['label']} | {row['total_timesteps']} | {row['score']:.1f} | "
            f"{val['mean_survival']:.1f} | {val['p10_survival']:.1f} | "
            f"{val['cvar_survival']:.1f} | {val['success_rate']:.3f} | "
            f"{len(row.get('tail_seed_candidates', []))} | {row.get('promoted', False)} |"
        )
    best = result.get("best")
    if best:
        lines.extend(["", f"Best validation checkpoint: `{best['checkpoint']}`"])
    final = result.get("final_eval")
    if final:
        recon = final["recon_policy_terminal_eval"]
        ppo = final.get("pure_ppo_eval") or final.get("pure_recurrent_ppo_eval")
        ppo_label = "pure_recurrent_ppo" if "pure_recurrent_ppo_eval" in final else "pure_ppo"
        lines.extend(
            [
                "",
                "## Final Held-Out Eval",
                "",
                "| evaluator | mean | p10 | cvar | success | episodes |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        if ppo:
            ppo_cvar = ppo.get("cvar_survival")
            ppo_cvar_text = "n/a" if ppo_cvar is None else f"{ppo_cvar:.1f}"
            lines.append(
                f"| {ppo_label} | {ppo['mean_survival']:.1f} | {ppo['p10_survival']:.1f} | {ppo_cvar_text} | {ppo['success_rate']:.3f} | {ppo['episodes']} |"
            )
        lines.append(
            f"| recon_policy_terminal | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['cvar_survival']:.1f} | {recon['success_rate']:.3f} | {recon['episodes']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(out: Path, result: dict[str, Any]) -> None:
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")


def run_tail_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = solve_threshold(args.n_poles)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install RL extras with `uv sync --extra rl` to train policy terminals") from exc

    base_hard_seeds = hard_train_seeds(args)
    active_seed_file = out / "active_hard_seeds.json"
    write_seed_file(active_seed_file, base_hard_seeds)
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else None

    def build_vec_env(source_stats: Any | None = None):
        env = make_vec_env(
            lambda: make_train_env(args, active_seed_file),
            n_envs=args.n_envs,
            seed=args.train_seed,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs={"start_method": "fork"} if vec_env_cls is SubprocVecEnv else None,
        )
        if args.vec_normalize:
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=bool(args.vec_normalize_reward),
                clip_obs=float(args.vec_normalize_clip_obs),
            )
            if source_stats is not None:
                copy_vec_normalize_stats(source_stats, env)
        return env

    train_env = build_vec_env()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total_timesteps = 0

    if args.start_model_path:
        model = PPO.load(str(args.start_model_path), env=train_env, device=args.device)
        model.set_random_seed(args.train_seed)
        start_path = out / "checkpoint_000000_start.zip"
        shutil.copy2(args.start_model_path, start_path)
        start_normalizer = export_vec_normalize_stats(train_env, out / "normalizer_000000_start.json") if args.vec_normalize else ""
        row = record_checkpoint(args, out, start_path, total_timesteps, "start", start_normalizer)
        row["promoted"] = True
        history.append(row)
        best = row
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
        "hard_train_seed_count": len(base_hard_seeds),
        "hard_train_seed_probability": args.hard_train_seed_probability,
        "late_survival_bonus": args.late_survival_bonus,
        "late_survival_start_fraction": args.late_survival_start_fraction,
        "vec_normalize": bool(args.vec_normalize),
        "vec_normalize_reward": bool(args.vec_normalize_reward),
        "vec_normalize_clip_obs": float(args.vec_normalize_clip_obs),
        "tail_seed_refresh_count": args.tail_seed_refresh_count,
        "tail_seed_pool_limit": args.tail_seed_pool_limit,
        "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
        "validation_episodes": args.validation_episodes,
        "validation_seed_count": len(validation_seeds(args)),
        "final_eval_episodes": args.final_eval_episodes,
        "score_weights": {
            "mean_survival": args.score_mean_weight,
            "p10_survival": args.score_p10_weight,
            "cvar_survival": args.score_cvar_weight,
            "success_rate": args.score_success_weight,
        },
        "promotion_gates": {
            "max_success_regression": args.max_success_regression,
            "max_p10_regression": args.max_p10_regression,
        },
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
            "late_survival_bonus": args.late_survival_bonus,
            "late_survival_start_fraction": args.late_survival_start_fraction,
            "vec_normalize": bool(args.vec_normalize),
            "vec_normalize_reward": bool(args.vec_normalize_reward),
            "vec_normalize_clip_obs": float(args.vec_normalize_clip_obs),
            "vec_env": args.vec_env,
        },
        "history": history,
        "best": best,
    }
    save_summary(out, result)

    active_seeds = list(base_hard_seeds)
    for chunk in range(1, args.chunks + 1):
        model.learn(
            total_timesteps=args.chunk_timesteps,
            reset_num_timesteps=(chunk == 1 and not args.start_model_path),
        )
        total_timesteps += args.chunk_timesteps
        checkpoint = out / f"checkpoint_{total_timesteps:06d}.zip"
        model.save(str(checkpoint))
        normalizer_path = (
            export_vec_normalize_stats(train_env, out / f"normalizer_{total_timesteps:06d}.json")
            if args.vec_normalize
            else ""
        )
        row = record_checkpoint(args, out, checkpoint, total_timesteps, f"chunk_{chunk}", normalizer_path)
        row["promoted"] = should_promote(row, best, args)
        history.append(row)
        if row["promoted"]:
            best = row
            shutil.copy2(checkpoint, out / "best_policy_terminal.zip")

        active_seeds = merge_seed_pool(
            active_seeds,
            row.get("tail_seed_candidates", []),
            args.tail_seed_pool_limit,
        )
        write_seed_file(active_seed_file, active_seeds)
        (out / "tail_seed_pool.json").write_text(
            json.dumps({"hard_seeds": active_seeds, "count": len(active_seeds)}, indent=2),
            encoding="utf-8",
        )
        if args.rebuild_env_each_chunk and chunk < args.chunks:
            old_env = train_env
            train_env = build_vec_env(old_env if args.vec_normalize else None)
            old_env.close()
            model.set_env(train_env)
        result.update({"history": history, "best": best, "active_tail_seed_count": len(active_seeds)})
        save_summary(out, result)

    final_eval = None
    if best is not None and args.final_eval_episodes > 0:
        best_path = Path(best["checkpoint"])
        final_args = eval_args(args, args.final_seed_start, args.final_eval_episodes)
        final_args.policy_terminal_normalizer_path = str(best.get("normalizer_path", ""))
        model_for_eval = PPO.load(str(best_path), device=args.device)
        seeds = final_seeds(args)
        final_eval = {
            "checkpoint": str(best_path),
            "pure_ppo_eval": evaluate_model(model_for_eval, final_args, seeds),
            "recon_policy_terminal_eval": evaluate_recon_terminal_tail(
                best_path,
                final_args,
                seeds,
                cvar_fraction=args.cvar_fraction,
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
    train_env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-model-path", default="")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--chunk-timesteps", type=int, default=25_000)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=2_010_000)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=0.55)
    parser.add_argument("--tail-seed-refresh-count", type=int, default=40)
    parser.add_argument("--tail-seed-min-steps", type=int, default=300)
    parser.add_argument("--tail-seed-pool-limit", type=int, default=800)
    parser.add_argument("--rebuild-env-each-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-seed-start", type=int, default=1_010_000)
    parser.add_argument("--validation-seed-starts", type=int, nargs="+", default=None)
    parser.add_argument("--validation-episodes", type=int, default=80)
    parser.add_argument("--cvar-fraction", type=float, default=0.10)
    parser.add_argument("--score-mean-weight", type=float, default=0.35)
    parser.add_argument("--score-p10-weight", type=float, default=0.75)
    parser.add_argument("--score-cvar-weight", type=float, default=0.75)
    parser.add_argument("--score-success-weight", type=float, default=130.0)
    parser.add_argument("--max-success-regression", type=float, default=0.01)
    parser.add_argument("--max-p10-regression", type=float, default=6.0)
    parser.add_argument("--final-seed-start", type=int, default=1_040_000)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--vec-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vec-normalize-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vec-normalize-clip-obs", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.025)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=25.0)
    parser.add_argument("--failure-penalty", type=float, default=2.0)
    parser.add_argument("--late-survival-bonus", type=float, default=0.0)
    parser.add_argument("--late-survival-start-fraction", type=float, default=0.80)
    parser.add_argument("--reward-mode", choices=["survival", "upright_shaping"], default="upright_shaping")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument(
        "--policy-terminal-scope",
        choices=["stabilize_chain", "selected", "all"],
        default="stabilize_chain",
    )
    parser.add_argument("--policy-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/policy_terminal_tail_curriculum")
    args = parser.parse_args()
    result = run_tail_curriculum(args)
    print(
        json.dumps(
            {
                "out": args.out,
                "status": result["status"],
                "best": result.get("best", {}).get("checkpoint") if result.get("best") else None,
                "wall_clock_seconds": result["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
