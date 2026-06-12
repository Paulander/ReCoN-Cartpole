from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from train_policy_terminal_iterative import passes, solve_threshold
from train_recurrent_policy_terminal_tail_curriculum import run_recurrent_tail_curriculum


def stage_args(args: argparse.Namespace, stage: dict[str, Any], start_model: str, index: int, out: Path) -> Namespace:
    return Namespace(
        start_model_path=start_model,
        n_poles=int(stage["n_poles"]),
        horizon=args.horizon,
        dt=float(stage.get("dt", args.dt)),
        dynamics_mode=args.dynamics_mode,
        action_mode=args.action_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=float(stage.get("initial_angle_range", args.initial_angle_range)),
        force_noise=float(stage.get("force_noise", args.force_noise)),
        link_coupling=args.link_coupling,
        chunk_timesteps=int(stage.get("chunk_timesteps", args.chunk_timesteps)),
        chunks=int(stage.get("chunks", args.chunks)),
        train_seed=args.train_seed + index * args.train_seed_stride,
        hard_train_seeds=str(stage.get("hard_train_seeds", args.hard_train_seeds)),
        hard_train_seed_probability=float(stage.get("hard_train_seed_probability", args.hard_train_seed_probability)),
        tail_seed_refresh_count=args.tail_seed_refresh_count,
        tail_seed_min_steps=args.tail_seed_min_steps,
        tail_seed_pool_limit=args.tail_seed_pool_limit,
        rebuild_env_each_chunk=args.rebuild_env_each_chunk,
        validation_seed_start=int(stage.get("validation_seed_start", args.validation_seed_start)),
        validation_seed_starts=stage.get("validation_seed_starts", args.validation_seed_starts),
        validation_episodes=int(stage.get("validation_episodes", args.validation_episodes)),
        cvar_fraction=args.cvar_fraction,
        score_mean_weight=args.score_mean_weight,
        score_p10_weight=args.score_p10_weight,
        score_cvar_weight=args.score_cvar_weight,
        score_success_weight=args.score_success_weight,
        max_success_regression=args.max_success_regression,
        max_p10_regression=args.max_p10_regression,
        max_cvar_regression=args.max_cvar_regression,
        final_seed_start=int(stage.get("final_seed_start", args.final_seed_start)),
        final_seed_starts=stage.get("final_seed_starts", getattr(args, "final_seed_starts", None)),
        final_eval_episodes=int(stage.get("final_eval_episodes", 0 if not stage.get("final", False) else args.final_eval_episodes)),
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        device=args.device,
        policy=args.policy,
        net_arch=args.net_arch,
        activation=args.activation,
        lstm_hidden_size=args.lstm_hidden_size,
        n_lstm_layers=args.n_lstm_layers,
        shared_lstm=args.shared_lstm,
        enable_critic_lstm=args.enable_critic_lstm,
        learning_rate=float(stage.get("learning_rate", args.learning_rate)),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        reward_mode=args.reward_mode,
        selection_mode=args.selection_mode,
        policy_terminal_blend=args.policy_terminal_blend,
        policy_terminal_scope=args.policy_terminal_scope,
        policy_observation_mode=args.policy_observation_mode,
        frame_stack=args.frame_stack,
        verbose=args.verbose,
        out=str(out),
    )


def default_stages(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "name": "n3_stable",
            "n_poles": 3,
            "initial_angle_range": min(args.initial_angle_range, 0.03),
            "force_noise": min(args.force_noise, 0.01),
            "chunks": args.n3_chunks,
            "chunk_timesteps": args.n3_chunk_timesteps,
            "validation_seed_start": 810_000,
            "validation_episodes": args.stage_validation_episodes,
        },
        {
            "name": "n4_low_angle_no_noise",
            "n_poles": 4,
            "initial_angle_range": 0.02,
            "force_noise": 0.0,
            "chunks": args.low_angle_chunks,
            "chunk_timesteps": args.low_angle_chunk_timesteps,
            "validation_seed_start": 910_000,
            "validation_episodes": args.stage_validation_episodes,
        },
        {
            "name": "n4_current",
            "n_poles": 4,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "chunks": args.current_chunks,
            "chunk_timesteps": args.current_chunk_timesteps,
            "validation_seed_start": 1_010_000,
            "validation_seed_starts": args.validation_seed_starts,
            "validation_episodes": args.validation_episodes,
        },
        {
            "name": "n4_tail",
            "n_poles": 4,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "hard_train_seeds": args.hard_train_seeds,
            "hard_train_seed_probability": args.tail_hard_seed_probability,
            "chunks": args.tail_chunks,
            "chunk_timesteps": args.tail_chunk_timesteps,
            "validation_seed_start": 1_010_000,
            "validation_seed_starts": args.validation_seed_starts,
            "validation_episodes": args.validation_episodes,
            "final_eval_episodes": args.final_eval_episodes,
            "final_seed_starts": getattr(args, "final_seed_starts", None),
            "final": True,
        },
    ]


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Recurrent Policy Terminal Curriculum",
        "",
        f"Status: `{result.get('status', 'running')}`",
        f"Observation mode: `{result.get('policy_observation_mode')}`",
        f"Frame stack: `{result.get('frame_stack')}`",
        "",
        "| stage | n | status | best mean | best p10 | best cvar | best success | best checkpoint |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in result.get("stages", []):
        best = row.get("best") or {}
        val = best.get("validation") or {}
        lines.append(
            f"| {row.get('name')} | {row.get('n_poles')} | {row.get('status')} | "
            f"{val.get('mean_survival', 0.0):.1f} | {val.get('p10_survival', 0.0):.1f} | "
            f"{val.get('cvar_survival', 0.0):.1f} | {val.get('success_rate', 0.0):.3f} | `{best.get('checkpoint', '')}` |"
        )
    final = result.get("final_eval") or {}
    if final:
        recon = final.get("recon_policy_terminal_eval") or {}
        lines.extend([
            "",
            "## Final Held-Out N=4 Eval",
            "",
            f"Mean `{recon.get('mean_survival', 0.0):.1f}`, p10 `{recon.get('p10_survival', 0.0):.1f}`, CVaR `{recon.get('cvar_survival', 0.0):.1f}`, success `{recon.get('success_rate', 0.0):.3f}`.",
        ])
    lines.extend([
        "",
        "## Claim Discipline",
        "",
        "The recurrent policy is trained through an easier-to-harder curriculum. N=4 is solved only if the final held-out N=4 block clears the configured threshold; N=3 warmup metrics are not used as N=4 solve evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = default_stages(args)
    current_model = args.start_model_path
    rows: list[dict[str, Any]] = []
    final_eval = None
    threshold = solve_threshold(4)
    for index, stage in enumerate(stages):
        stage_out = out / f"{index:02d}_{stage['name']}"
        run_args = stage_args(args, stage, current_model, index, stage_out)
        result = run_recurrent_tail_curriculum(run_args)
        best = result.get("best") or {}
        if best.get("checkpoint"):
            current_model = str(best["checkpoint"])
        row = {
            "name": stage["name"],
            "n_poles": stage["n_poles"],
            "status": result.get("status"),
            "out": str(stage_out),
            "best": best,
            "final_eval": result.get("final_eval"),
            "wall_clock_seconds": result.get("wall_clock_seconds", 0.0),
        }
        rows.append(row)
        if stage.get("final"):
            final_eval = result.get("final_eval")
        partial = {
            "status": "running",
            "policy_observation_mode": args.policy_observation_mode,
            "frame_stack": args.frame_stack,
            "stages": rows,
            "final_eval": final_eval,
        }
        (out / "summary.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        write_markdown(partial, out / "summary.md")
        final_recon = (final_eval or {}).get("recon_policy_terminal_eval") if final_eval else None
        if final_recon and passes(final_recon, threshold):
            break
    final_recon = (final_eval or {}).get("recon_policy_terminal_eval") if final_eval else None
    result = {
        "status": "solved" if final_recon and passes(final_recon, threshold) else "completed_not_solved",
        "threshold": threshold,
        "policy_observation_mode": args.policy_observation_mode,
        "frame_stack": args.frame_stack,
        "stages": rows,
        "final_eval": final_eval,
        "mechanisms": {
            "recurrent_policy_gradient": True,
            "policy_terminal": True,
            "n3_to_n4_curriculum": True,
            "previous_force_observation": "prev_force" in args.policy_observation_mode,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-model-path", default="")
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
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--n3-chunks", type=int, default=2)
    parser.add_argument("--n3-chunk-timesteps", type=int, default=25_000)
    parser.add_argument("--low-angle-chunks", type=int, default=2)
    parser.add_argument("--low-angle-chunk-timesteps", type=int, default=25_000)
    parser.add_argument("--current-chunks", type=int, default=3)
    parser.add_argument("--current-chunk-timesteps", type=int, default=25_000)
    parser.add_argument("--tail-chunks", type=int, default=4)
    parser.add_argument("--tail-chunk-timesteps", type=int, default=25_000)
    parser.add_argument("--train-seed", type=int, default=2_500_000)
    parser.add_argument("--train-seed-stride", type=int, default=10_000)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=0.25)
    parser.add_argument("--tail-hard-seed-probability", type=float, default=0.55)
    parser.add_argument("--tail-seed-refresh-count", type=int, default=40)
    parser.add_argument("--tail-seed-min-steps", type=int, default=300)
    parser.add_argument("--tail-seed-pool-limit", type=int, default=900)
    parser.add_argument("--rebuild-env-each-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-seed-start", type=int, default=1_010_000)
    parser.add_argument("--validation-seed-starts", type=int, nargs="+", default=[900_000, 930_000, 970_000, 1_010_000])
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--stage-validation-episodes", type=int, default=24)
    parser.add_argument("--cvar-fraction", type=float, default=0.10)
    parser.add_argument("--score-mean-weight", type=float, default=0.25)
    parser.add_argument("--score-p10-weight", type=float, default=0.85)
    parser.add_argument("--score-cvar-weight", type=float, default=0.85)
    parser.add_argument("--score-success-weight", type=float, default=140.0)
    parser.add_argument("--max-success-regression", type=float, default=0.01)
    parser.add_argument("--max-p10-regression", type=float, default=6.0)
    parser.add_argument("--max-cvar-regression", type=float, default=8.0)
    parser.add_argument("--final-seed-start", type=int, default=1_040_000)
    parser.add_argument("--final-seed-starts", type=int, nargs="+", default=None)
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
    parser.add_argument("--policy-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw4_prev_force")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/recurrent_policy_terminal_curriculum")
    args = parser.parse_args()
    result = run_curriculum(args)
    print(json.dumps({"out": args.out, "status": result["status"], "wall_clock_seconds": result["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
