from __future__ import annotations

import argparse
import itertools
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from train_policy_terminal_iterative import passes, solve_threshold
from train_policy_terminal_tail_curriculum import run_tail_curriculum, tail_score
from recon_cartpole.control.policy_observation import POLICY_OBSERVATION_MODES


def _floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _ints(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def _texts(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _archs(text: str) -> list[str]:
    sep = ";" if ";" in str(text) else "|"
    return [item.strip() for item in str(text).split(sep) if item.strip()]


def _bools(text: str) -> list[bool]:
    out: list[bool] = []
    for item in _texts(text):
        out.append(item.lower() in ("1", "true", "yes", "on"))
    return out


def _indices(text: str) -> list[int]:
    return [int(item) for item in str(text or "").split(",") if item.strip()]


def candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = itertools.product(
        _floats(args.learning_rates),
        _floats(args.clip_ranges),
        _ints(args.n_steps_values),
        _ints(args.n_epochs_values),
        _floats(args.gae_lambdas),
        _floats(args.ent_coefs),
        _archs(args.net_arch_values),
        _bools(args.vec_normalize_values),
        _floats(args.late_survival_bonus_values),
    )
    rows = [
        {
            "grid_index": int(grid_index),
            "learning_rate": lr,
            "clip_range": clip,
            "n_steps": n_steps,
            "n_epochs": n_epochs,
            "gae_lambda": gae,
            "ent_coef": ent,
            "net_arch": arch,
            "vec_normalize": vec_norm,
            "late_survival_bonus": late_bonus,
        }
        for grid_index, (lr, clip, n_steps, n_epochs, gae, ent, arch, vec_norm, late_bonus) in enumerate(raw)
    ]
    selected_indices = _indices(getattr(args, "candidate_indices", ""))
    if selected_indices:
        wanted = set(selected_indices)
        rows = [row for row in rows if int(row["grid_index"]) in wanted]
        rows.sort(key=lambda row: selected_indices.index(int(row["grid_index"])))
    elif args.shuffle_stride > 1 and rows:
        stride = max(1, int(args.shuffle_stride))
        rows = rows[::stride] + [row for idx, row in enumerate(rows) if idx % stride != 0]
    start = max(0, int(args.candidate_offset))
    end = None if args.max_candidates <= 0 else start + int(args.max_candidates)
    return rows[start:end]


def curriculum_args(args: argparse.Namespace, candidate: dict[str, Any], index: int, out: Path) -> Namespace:
    return Namespace(
        start_model_path=args.start_model_path,
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
        chunk_timesteps=args.chunk_timesteps,
        chunks=args.chunks,
        train_seed=args.train_seed + index * args.train_seed_stride,
        hard_train_seeds=args.hard_train_seeds,
        hard_train_seed_probability=args.hard_train_seed_probability,
        tail_seed_refresh_count=args.tail_seed_refresh_count,
        tail_seed_min_steps=args.tail_seed_min_steps,
        tail_seed_pool_limit=args.tail_seed_pool_limit,
        rebuild_env_each_chunk=args.rebuild_env_each_chunk,
        validation_seed_start=args.validation_seed_start,
        validation_seed_starts=args.validation_seed_starts,
        validation_episodes=args.validation_episodes,
        cvar_fraction=args.cvar_fraction,
        score_mean_weight=args.score_mean_weight,
        score_p10_weight=args.score_p10_weight,
        score_cvar_weight=args.score_cvar_weight,
        score_success_weight=args.score_success_weight,
        max_success_regression=args.max_success_regression,
        max_p10_regression=args.max_p10_regression,
        max_cvar_regression=args.max_cvar_regression,
        promotion_mode=args.promotion_mode,
        final_seed_start=args.final_seed_start,
        final_seed_starts=args.final_seed_starts,
        final_eval_episodes=args.final_eval_episodes,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        vec_normalize=bool(candidate["vec_normalize"]),
        vec_normalize_reward=args.vec_normalize_reward,
        vec_normalize_clip_obs=args.vec_normalize_clip_obs,
        device=args.device,
        policy=args.policy,
        net_arch=str(candidate["net_arch"]),
        activation=args.activation,
        learning_rate=float(candidate["learning_rate"]),
        n_steps=int(candidate["n_steps"]),
        batch_size=args.batch_size,
        n_epochs=int(candidate["n_epochs"]),
        gamma=args.gamma,
        gae_lambda=float(candidate["gae_lambda"]),
        clip_range=float(candidate["clip_range"]),
        ent_coef=float(candidate["ent_coef"]),
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        late_survival_bonus=float(candidate["late_survival_bonus"]),
        late_survival_start_fraction=args.late_survival_start_fraction,
        teacher_anchor_model_path=args.teacher_anchor_model_path,
        teacher_action_penalty=args.teacher_action_penalty,
        teacher_anchor_until_fraction=args.teacher_anchor_until_fraction,
        teacher_anchor_risk_threshold=args.teacher_anchor_risk_threshold,
        reward_mode=args.reward_mode,
        selection_mode=args.selection_mode,
        policy_terminal_blend=args.policy_terminal_blend,
        policy_terminal_scope=args.policy_terminal_scope,
        policy_observation_mode=args.policy_observation_mode,
        frame_stack=args.frame_stack,
        verbose=args.verbose,
        out=str(out),
    )


def candidate_summary(result: dict[str, Any], candidate: dict[str, Any], index: int) -> dict[str, Any]:
    best = result.get("best") or {}
    best_val = best.get("validation") or {}
    final = result.get("final_eval") or {}
    final_val = (final.get("recon_policy_terminal_eval") or {}) if final else {}
    metric = final_val or best_val
    return {
        "index": int(index),
        "status": result.get("status"),
        "candidate": candidate,
        "out": result.get("out", ""),
        "best_checkpoint": best.get("checkpoint", ""),
        "best_normalizer_path": best.get("normalizer_path", ""),
        "validation": best_val,
        "final_eval": final_val,
        "score": tail_score(metric) if metric else 0.0,
        "wall_clock_seconds": result.get("wall_clock_seconds", 0.0),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# PPO Terminal Hyperparameter Sweep",
        "",
        f"Status: `{result.get('status', 'running')}`",
        f"N poles: `{result.get('n_poles')}`",
        f"Validation seed starts: `{', '.join(str(s) for s in result.get('validation_seed_starts', []))}`",
        f"Final seed starts: `{', '.join(str(s) for s in result.get('final_seed_starts', [result.get('final_seed_start')]))}`",
        "",
        "| idx | grid | status | lr | clip | steps | epochs | gae | ent | net | vecnorm | late bonus | mean | p10 | cvar | success | score |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("candidates", []):
        cfg = row["candidate"]
        val = row.get("final_eval") or row.get("validation") or {}
        lines.append(
            f"| {row['index']} | {cfg.get('grid_index', row['index'])} | {row.get('status', '')} | {cfg['learning_rate']} | {cfg['clip_range']} | "
            f"{cfg['n_steps']} | {cfg['n_epochs']} | {cfg['gae_lambda']} | {cfg['ent_coef']} | "
            f"{cfg['net_arch']} | {cfg['vec_normalize']} | {cfg['late_survival_bonus']} | "
            f"{val.get('mean_survival', 0.0):.1f} | {val.get('p10_survival', 0.0):.1f} | "
            f"{val.get('cvar_survival', 0.0):.1f} | {val.get('success_rate', 0.0):.3f} | {row.get('score', 0.0):.1f} |"
        )
    best = result.get("best")
    if best:
        lines.extend(["", f"Best checkpoint: `{best.get('best_checkpoint', '')}`"])
    lines.extend([
        "",
        "## Claim Discipline",
        "",
        "This sweep varies PPO training knobs for the learned terminal. It records whether VecNormalize and late-survival reward shaping were active, and it only reports N=4 solved if the held-out final block clears the configured threshold.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = solve_threshold(args.n_poles)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    candidates = candidate_grid(args)
    for index, candidate in enumerate(candidates):
        run_out = out / f"candidate_{index:02d}"
        run_args = curriculum_args(args, candidate, index, run_out)
        try:
            run_result = run_tail_curriculum(run_args)
            run_result["out"] = str(run_out)
            row = candidate_summary(run_result, candidate, index)
        except Exception as exc:  # keep the sweep fail-fast per candidate, not per full batch
            row = {
                "index": int(index),
                "status": "failed",
                "candidate": candidate,
                "out": str(run_out),
                "error": f"{type(exc).__name__}: {exc}",
                "score": float("-inf"),
            }
        rows.append(row)
        if row.get("status") != "failed" and (best is None or float(row["score"]) > float(best["score"])):
            best = row
        current = {
            "status": "running",
            "threshold": threshold,
            "n_poles": args.n_poles,
            "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
            "final_seed_start": args.final_seed_start,
            "final_seed_starts": args.final_seed_starts or [args.final_seed_start],
            "candidates": rows,
            "best": best,
            "mechanisms": {
                "ppo_policy_gradient": True,
                "policy_terminal": True,
                "vec_normalize_swept": True,
                "late_survival_bonus_swept": True,
                "gain_mutation": False,
            },
        }
        (out / "summary.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
        write_markdown(current, out / "summary.md")
        if best and best.get("final_eval") and passes(best["final_eval"], threshold):
            break
    solved = bool(best and best.get("final_eval") and passes(best["final_eval"], threshold))
    result = {
        "status": "solved" if solved else "completed_not_solved",
        "threshold": threshold,
        "n_poles": args.n_poles,
        "validation_seed_starts": args.validation_seed_starts or [args.validation_seed_start],
        "final_seed_start": args.final_seed_start,
        "final_seed_starts": args.final_seed_starts or [args.final_seed_start],
        "candidates": rows,
        "best": best,
        "wall_clock_seconds": time.perf_counter() - started,
        "mechanisms": {
            "ppo_policy_gradient": True,
            "policy_terminal": True,
            "vec_normalize_swept": True,
            "late_survival_bonus_swept": True,
            "gain_mutation": False,
        },
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
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
    parser.add_argument("--train-seed", type=int, default=2_300_000)
    parser.add_argument("--train-seed-stride", type=int, default=1000)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-train-seed-probability", type=float, default=0.55)
    parser.add_argument("--tail-seed-refresh-count", type=int, default=40)
    parser.add_argument("--tail-seed-min-steps", type=int, default=300)
    parser.add_argument("--tail-seed-pool-limit", type=int, default=800)
    parser.add_argument("--rebuild-env-each-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-seed-start", type=int, default=1_010_000)
    parser.add_argument("--validation-seed-starts", type=int, nargs="+", default=[900_000, 930_000, 970_000, 1_010_000])
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--cvar-fraction", type=float, default=0.10)
    parser.add_argument("--score-mean-weight", type=float, default=0.35)
    parser.add_argument("--score-p10-weight", type=float, default=0.75)
    parser.add_argument("--score-cvar-weight", type=float, default=0.75)
    parser.add_argument("--score-success-weight", type=float, default=130.0)
    parser.add_argument("--max-success-regression", type=float, default=0.01)
    parser.add_argument("--max-p10-regression", type=float, default=6.0)
    parser.add_argument("--max-cvar-regression", type=float, default=8.0)
    parser.add_argument("--promotion-mode", choices=["score", "lexicographic_success"], default="lexicographic_success")
    parser.add_argument("--final-seed-start", type=int, default=1_040_000)
    parser.add_argument("--final-seed-starts", type=int, nargs="+", default=None)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.0)
    parser.add_argument("--success-bonus", type=float, default=25.0)
    parser.add_argument("--failure-penalty", type=float, default=2.0)
    parser.add_argument("--late-survival-start-fraction", type=float, default=0.80)
    parser.add_argument("--teacher-anchor-model-path", default="")
    parser.add_argument("--teacher-action-penalty", type=float, default=0.0)
    parser.add_argument("--teacher-anchor-until-fraction", type=float, default=1.0)
    parser.add_argument("--teacher-anchor-risk-threshold", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["survival", "upright_shaping"], default="upright_shaping")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--policy-observation-mode", choices=POLICY_OBSERVATION_MODES, default="normalized_raw")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--learning-rates", default="2.5e-6,5e-6,1e-5")
    parser.add_argument("--clip-ranges", default="0.015,0.025,0.05")
    parser.add_argument("--n-steps-values", default="512,1024")
    parser.add_argument("--n-epochs-values", default="2,4")
    parser.add_argument("--gae-lambdas", default="0.9,0.95,0.98")
    parser.add_argument("--ent-coefs", default="0.0,0.001")
    parser.add_argument("--net-arch-values", default="64,64;128,128;256,128")
    parser.add_argument("--vec-normalize-values", default="false,true")
    parser.add_argument("--vec-normalize-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vec-normalize-clip-obs", type=float, default=10.0)
    parser.add_argument("--late-survival-bonus-values", default="0.0,0.02,0.05")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-indices", default="", help="Comma-separated original grid indices to run in this order.")
    parser.add_argument("--shuffle-stride", type=int, default=7)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/ppo_terminal_sweep")
    args = parser.parse_args()
    result = run_sweep(args)
    print(json.dumps({"out": args.out, "status": result["status"], "best": result.get("best")}, indent=2))


if __name__ == "__main__":
    main()
