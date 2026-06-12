from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.policy_observation import adjacent_subchain_features
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminalConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=args.n_poles,
            horizon=args.horizon,
            dt=args.dt,
            dynamics_mode=args.dynamics_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
        )
    )


def make_controller(args: argparse.Namespace) -> ReConCartPoleController:
    mingru = MinGRUTerminalConfig(
        enabled=True,
        hidden_size=args.mingru_hidden_size,
        sequence_length=args.mingru_sequence_length,
        observation_mode=args.mingru_observation_mode,
        include_prev_force=args.mingru_include_prev_force,
        include_context=args.mingru_include_context,
        scope=args.policy_terminal_scope,
        checkpoint_path=args.mingru_checkpoint,
        passthrough_enabled=True,
        passthrough_confidence_floor=args.passthrough_confidence_floor,
    )
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_mingru_terminal",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            mingru_terminal=mingru,
        )
    )


def subchain_vector(raw_state: Any, args: argparse.Namespace) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    needed = 2 + 2 * int(args.n_poles)
    if raw.size < needed:
        raise ValueError("raw_state is too short for subchain motif features")
    n = int(args.n_poles)
    theta = raw[2 : 2 + n] / max(float(args.theta_threshold), 1e-9)
    theta_dot = raw[2 + n : 2 + 2 * n] / max(float(args.pole_velocity_scale), 1e-9)
    features = adjacent_subchain_features(theta, theta_dot, max(4, n))
    cart = [float(raw[0]) / max(float(args.x_threshold), 1e-9), float(raw[1]) / max(float(args.cart_velocity_scale), 1e-9)]
    return np.asarray(cart + features, dtype=np.float32)


def diagnostic_tail(step: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    mingru = step.get("mingru_passthrough") or step.get("mingru_terminal") or {}
    return np.asarray(
        [
            float(step.get("force", 0.0) or 0.0) / max(float(args.force_mag), 1e-9),
            float(mingru.get("confidence", 0.0) or 0.0),
            float(mingru.get("failure_probability", 0.0) or 0.0),
            float(mingru.get("value", 0.0) or 0.0),
            float(mingru.get("hidden_norm", 0.0) or 0.0) / 10.0,
            float(mingru.get("passthrough_logit_margin", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )


def motif_vector(step: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    base = subchain_vector(step.get("raw_state", []), args)
    if bool(getattr(args, "include_recon_diagnostics", False)):
        return np.concatenate([base, diagnostic_tail(step, args)]).astype(np.float32)
    return base


def collect_rows(args: argparse.Namespace, seed_start: int, episodes: int) -> dict[str, Any]:
    controller = make_controller(args)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    seeds: list[int] = []
    step_indices: list[int] = []
    steps_out: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for idx in range(int(episodes)):
        seed = int(seed_start) + idx
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=True)
        trace = result.get("trace", [])
        steps = int(result.get("steps", len(trace)))
        success = steps >= int(args.horizon)
        steps_out.append(float(steps))
        for step_idx, item in enumerate(trace):
            if args.sample_stride > 1 and step_idx % int(args.sample_stride) != 0:
                continue
            try:
                vector = motif_vector(item, args)
            except ValueError:
                continue
            near_failure = (not success) and ((steps - step_idx) <= int(args.failure_window))
            if near_failure or success or not bool(args.drop_early_failure_negatives):
                rows.append(vector)
                labels.append(1 if near_failure else 0)
                seeds.append(seed)
                step_indices.append(step_idx)
        per_seed.append({"seed": seed, "steps": steps, "success": bool(success)})
    x = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 14), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    summary = summarize_steps(steps_out, args.horizon)
    summary.update({
        "episodes": int(episodes),
        "rows": int(x.shape[0]),
        "positive_rows": int(np.sum(y)),
        "per_seed": per_seed,
    })
    return {"x": x, "y": y, "seeds": np.asarray(seeds, dtype=np.int64), "steps": np.asarray(step_indices, dtype=np.int64), "summary": summary}


def fit_prototypes(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> dict[str, Any]:
    if x.shape[0] == 0 or not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("prototype fitting requires positive and negative rows")
    pos = x[y == 1]
    neg = x[y == 0]
    scale = np.std(x, axis=0) + float(eps)
    model = {
        "positive_mean": np.mean(pos, axis=0).astype(float).tolist(),
        "negative_mean": np.mean(neg, axis=0).astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "positive_rows": int(pos.shape[0]),
        "negative_rows": int(neg.shape[0]),
    }
    return model


def motif_scores(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    pos = np.asarray(model["positive_mean"], dtype=np.float32)
    neg = np.asarray(model["negative_mean"], dtype=np.float32)
    scale = np.asarray(model["scale"], dtype=np.float32)
    z = x / np.maximum(scale, 1e-6)
    p = pos / np.maximum(scale, 1e-6)
    n = neg / np.maximum(scale, 1e-6)
    d_pos = np.mean((z - p) ** 2, axis=1)
    d_neg = np.mean((z - n) ** 2, axis=1)
    return d_neg - d_pos


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    return float((np.sum(ranks[y == 1]) - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def evaluate_model(model: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    x = data["x"]
    y = data["y"]
    scores = motif_scores(model, x) if x.size else np.asarray([], dtype=np.float32)
    positives = scores[y == 1]
    negatives = scores[y == 0]
    return {
        **data["summary"],
        "auc": roc_auc(y, scores) if scores.size else 0.0,
        "score_mean_positive": float(np.mean(positives)) if positives.size else 0.0,
        "score_mean_negative": float(np.mean(negatives)) if negatives.size else 0.0,
        "score_p90_positive": float(np.percentile(positives, 90)) if positives.size else 0.0,
        "score_p90_negative": float(np.percentile(negatives, 90)) if negatives.size else 0.0,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    train = result["train_eval"]
    held = result["heldout_eval"]
    lines = [
        "# Subchain Motif Prototype Diagnostic",
        "",
        "This is a diagnostic learner only: it does not affect CartPole control and makes no solve claim.",
        "",
        f"Checkpoint: `{result['mingru_checkpoint']}`",
        f"Feature dim: `{result['feature_dim']}`; failure window: `{result['failure_window']}`; sample stride: `{result['sample_stride']}`; ReCoN diagnostics: `{result.get('include_recon_diagnostics', False)}`",
        "",
        "| split | episodes | rows | positive rows | mean | p10 | success | AUC | pos score | neg score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| train | {train['episodes']} | {train['rows']} | {train['positive_rows']} | {train['mean_survival']:.1f} | {train['p10_survival']:.1f} | {train['success_rate']:.3f} | {train['auc']:.3f} | {train['score_mean_positive']:.3f} | {train['score_mean_negative']:.3f} |",
        f"| heldout | {held['episodes']} | {held['rows']} | {held['positive_rows']} | {held['mean_survival']:.1f} | {held['p10_survival']:.1f} | {held['success_rate']:.3f} | {held['auc']:.3f} | {held['score_mean_positive']:.3f} | {held['score_mean_negative']:.3f} |",
        "",
        "Interpretation: AUC well above 0.5 means local subchain phase motifs carry reusable failure/success information. AUC near 0.5 means this motif representation is not enough by itself and should not be used as a control gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    train = collect_rows(args, args.train_seed_start, args.train_episodes)
    model = fit_prototypes(train["x"], train["y"])
    heldout = collect_rows(args, args.heldout_seed_start, args.heldout_episodes)
    result = {
        "status": "completed",
        "mingru_checkpoint": args.mingru_checkpoint,
        "feature_dim": int(train["x"].shape[1]),
        "failure_window": int(args.failure_window),
        "sample_stride": int(args.sample_stride),
        "include_recon_diagnostics": bool(getattr(args, "include_recon_diagnostics", False)),
        "prototype_model": model,
        "train_eval": evaluate_model(model, train),
        "heldout_eval": evaluate_model(model, heldout),
        "mechanisms": {
            "subchain_motif_prototypes": True,
            "diagnostic_only": True,
            "control_policy_changed": False,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "prototype_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    write_markdown(result, out / "report.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mingru-checkpoint", default="reports/n4_mingru_dagger_iter2_20260612_seed2760k/supervised_h256_seq32_noctx/mingru_terminal.pt")
    parser.add_argument("--mingru-hidden-size", type=int, default=256)
    parser.add_argument("--mingru-sequence-length", type=int, default=32)
    parser.add_argument("--mingru-observation-mode", default="normalized_raw4_prev_force")
    parser.add_argument("--mingru-include-prev-force", action="store_true", default=True)
    parser.add_argument("--mingru-no-prev-force", dest="mingru_include_prev_force", action="store_false")
    parser.add_argument("--mingru-include-context", action="store_true", default=False)
    parser.add_argument("--mingru-no-context", dest="mingru_include_context", action="store_false")
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.90)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--x-threshold", type=float, default=2.4)
    parser.add_argument("--theta-threshold", type=float, default=12.0 * 2.0 * np.pi / 360.0)
    parser.add_argument("--cart-velocity-scale", type=float, default=5.0)
    parser.add_argument("--pole-velocity-scale", type=float, default=5.0)
    parser.add_argument("--failure-window", type=int, default=80)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--include-recon-diagnostics", action="store_true", default=False)
    parser.add_argument("--drop-early-failure-negatives", action="store_true", default=True)
    parser.add_argument("--keep-early-failure-negatives", dest="drop_early_failure_negatives", action="store_false")
    parser.add_argument("--train-seed-start", type=int, default=2420000)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--heldout-seed-start", type=int, default=2100000)
    parser.add_argument("--heldout-episodes", type=int, default=60)
    parser.add_argument("--out", default="reports/n4_subchain_motif_diagnostic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps({"out": args.out, "heldout_auc": result["heldout_eval"]["auc"], "heldout_success": result["heldout_eval"]["success_rate"]}, indent=2))


if __name__ == "__main__":
    main()
