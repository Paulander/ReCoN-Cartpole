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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_policy_terminal import hard_train_seeds, make_env, ppo_kwargs  # noqa: E402
from train_policy_terminal_iterative import final_seeds, passes, solve_threshold  # noqa: E402
from train_policy_terminal_tail_curriculum import (  # noqa: E402
    evaluate_recon_terminal_tail,
    merge_seed_pool,
    tail_metrics,
    tail_score,
    tail_seed_pool,
    validation_seeds,
    write_seed_file,
    save_summary,
)


def recurrent_eval_args(args: argparse.Namespace, seed_start: int, episodes: int) -> Namespace:
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
        policy_terminal_recurrent=True,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        reward_mode=args.reward_mode,
        eval_seed_start=seed_start,
        eval_episodes=episodes,
    )


def recurrent_policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = dict(ppo_kwargs(args).get("policy_kwargs", {}))
    kwargs["lstm_hidden_size"] = int(args.lstm_hidden_size)
    kwargs["n_lstm_layers"] = int(args.n_lstm_layers)
    kwargs["shared_lstm"] = bool(args.shared_lstm)
    kwargs["enable_critic_lstm"] = bool(args.enable_critic_lstm)
    return kwargs


def recurrent_ppo_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = ppo_kwargs(args)
    kwargs["policy_kwargs"] = recurrent_policy_kwargs(args)
    return kwargs


def evaluate_recurrent_model(model: Any, args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    steps: list[float] = []
    returns: list[float] = []
    for seed in seeds:
        env = make_env(
            args,
            reward_mode="survival",
            use_frame_stack=True,
            use_success_bonus=False,
            use_failure_penalty=False,
        )
        obs, _info = env.reset(seed=seed)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        total = 0.0
        for step in range(args.horizon):
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                steps.append(float(step + 1))
                returns.append(total)
                break
        else:
            steps.append(float(args.horizon))
            returns.append(total)
    summary = tail_metrics(steps, args.horizon, getattr(args, "cvar_fraction", 0.10))
    summary.update(
        {"returns_mean": float(np.mean(returns)) if returns else 0.0, "episodes": len(seeds)}
    )
    return summary


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
) -> dict[str, Any]:
    seeds = validation_seeds(args)
    eval_config = recurrent_eval_args(args, args.validation_seed_start, args.validation_episodes)
    eval_config.cvar_fraction = args.cvar_fraction
    summary = evaluate_recon_terminal_tail(
        checkpoint_path,
        eval_config,
        seeds,
        cvar_fraction=args.cvar_fraction,
    )
    row = {
        "label": label,
        "checkpoint": str(checkpoint_path),
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
    if float(validation["cvar_survival"]) + args.max_cvar_regression < float(
        best_validation["cvar_survival"]
    ):
        return False
    return float(row["score"]) > float(best["score"])


def run_recurrent_tail_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = solve_threshold(args.n_poles)
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install RL extras with `uv sync --extra rl` for RecurrentPPO") from exc

    base_hard_seeds = hard_train_seeds(args)
    active_seed_file = out / "active_hard_seeds.json"
    write_seed_file(active_seed_file, base_hard_seeds)
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else None

    def build_vec_env():
        return make_vec_env(
            lambda: make_train_env(args, active_seed_file),
            n_envs=args.n_envs,
            seed=args.train_seed,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs={"start_method": "fork"} if vec_env_cls is SubprocVecEnv else None,
        )

    train_env = build_vec_env()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total_timesteps = 0

    if args.start_model_path:
        model = RecurrentPPO.load(str(args.start_model_path), env=train_env, device=args.device)
        model.set_random_seed(args.train_seed)
        start_path = out / "checkpoint_000000_start.zip"
        shutil.copy2(args.start_model_path, start_path)
        row = record_checkpoint(args, out, start_path, total_timesteps, "start")
        row["promoted"] = True
        history.append(row)
        best = row
    else:
        model = RecurrentPPO(
            args.policy,
            train_env,
            seed=args.train_seed,
            verbose=args.verbose,
            device=args.device,
            **recurrent_ppo_kwargs(args),
        )

    result: dict[str, Any] = {
        "status": "running",
        "threshold": threshold,
        "mechanism": "RecurrentPPO policy terminal inside ReCoN",
        "reward_mode": args.reward_mode,
        "selection_mode": args.selection_mode,
        "policy_terminal_blend": args.policy_terminal_blend,
        "policy_terminal_scope": args.policy_terminal_scope,
        "frame_stack": args.frame_stack,
        "policy_observation_mode": args.policy_observation_mode,
        "hard_train_seed_count": len(base_hard_seeds),
        "hard_train_seed_probability": args.hard_train_seed_probability,
        "tail_seed_refresh_count": args.tail_seed_refresh_count,
        "tail_seed_pool_limit": args.tail_seed_pool_limit,
        "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
        "validation_episodes": args.validation_episodes,
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
            "max_cvar_regression": args.max_cvar_regression,
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
            "lstm_hidden_size": args.lstm_hidden_size,
            "n_lstm_layers": args.n_lstm_layers,
            "shared_lstm": args.shared_lstm,
            "enable_critic_lstm": args.enable_critic_lstm,
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
        row = record_checkpoint(args, out, checkpoint, total_timesteps, f"chunk_{chunk}")
        row["promoted"] = should_promote(row, best, args)
        history.append(row)
        if row["promoted"]:
            best = row
            shutil.copy2(checkpoint, out / "best_recurrent_policy_terminal.zip")
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
            train_env.close()
            train_env = build_vec_env()
            model.set_env(train_env)
        result.update({"history": history, "best": best, "active_tail_seed_count": len(active_seeds)})
        save_summary(out, result)

    final_eval = None
    if best is not None and args.final_eval_episodes > 0:
        best_path = Path(best["checkpoint"])
        final_args = recurrent_eval_args(args, args.final_seed_start, args.final_eval_episodes)
        final_args.cvar_fraction = args.cvar_fraction
        eval_model = RecurrentPPO.load(str(best_path), device=args.device)
        seeds = final_seeds(args)
        final_eval = {
            "checkpoint": str(best_path),
            "pure_recurrent_ppo_eval": evaluate_recurrent_model(eval_model, final_args, seeds),
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
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--train-seed", type=int, default=2_110_000)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=0.40)
    parser.add_argument("--tail-seed-refresh-count", type=int, default=40)
    parser.add_argument("--tail-seed-min-steps", type=int, default=300)
    parser.add_argument("--tail-seed-pool-limit", type=int, default=900)
    parser.add_argument("--rebuild-env-each-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-seed-start", type=int, default=1_110_000)
    parser.add_argument("--validation-seed-starts", type=int, nargs="+", default=None)
    parser.add_argument("--validation-episodes", type=int, default=40)
    parser.add_argument("--cvar-fraction", type=float, default=0.10)
    parser.add_argument("--score-mean-weight", type=float, default=0.25)
    parser.add_argument("--score-p10-weight", type=float, default=0.85)
    parser.add_argument("--score-cvar-weight", type=float, default=0.85)
    parser.add_argument("--score-success-weight", type=float, default=140.0)
    parser.add_argument("--max-success-regression", type=float, default=0.01)
    parser.add_argument("--max-p10-regression", type=float, default=6.0)
    parser.add_argument("--max-cvar-regression", type=float, default=8.0)
    parser.add_argument("--final-seed-start", type=int, default=1_140_000)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="MlpLstmPolicy")
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--n-lstm-layers", type=int, default=1)
    parser.add_argument("--shared-lstm", action="store_true")
    parser.add_argument("--enable-critic-lstm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.08)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=25.0)
    parser.add_argument("--failure-penalty", type=float, default=2.0)
    parser.add_argument("--reward-mode", choices=["survival", "upright_shaping"], default="upright_shaping")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--policy-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force"], default="normalized_raw")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/recurrent_policy_terminal_tail_curriculum")
    args = parser.parse_args()
    result = run_recurrent_tail_curriculum(args)
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
