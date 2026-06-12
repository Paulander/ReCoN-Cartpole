from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.policy_observation import policy_observation_from_state
from recon_cartpole.control.residual_features import residual_aux_features
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout


def make_env(args: argparse.Namespace, force_noise: float | None = None) -> CartPoleNEnv:
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
            force_noise=args.force_noise if force_noise is None else force_noise,
            link_coupling=args.link_coupling,
        )
    )


def make_controller(args: argparse.Namespace, residual_model_path: str = "", gate_threshold: float | None = None) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_policy_terminal",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.base_model_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.base_observation_mode,
            policy_terminal_normalizer_path=args.base_normalizer_path,
            residual_policy_terminal_path=residual_model_path,
            residual_policy_terminal_mode="bin_delta",
            residual_policy_terminal_action_bins=args.residual_action_bins,
            residual_policy_terminal_gate_threshold=args.residual_gate_threshold if gate_threshold is None else gate_threshold,
            residual_policy_terminal_feature_mode=args.residual_feature_mode,
        )
    )


def set_env_state(env: CartPoleNEnv, raw_state: list[float], step: int) -> None:
    env.state = np.asarray(raw_state, dtype=float).copy()
    env.steps = int(step)


def force_to_index(force: float, args: argparse.Namespace) -> int:
    values = np.linspace(-args.force_mag, args.force_mag, int(args.discrete_action_bins))
    return int(np.argmin(np.abs(values - float(force))))


def index_to_force(idx: int, args: argparse.Namespace) -> float:
    values = np.linspace(-args.force_mag, args.force_mag, int(args.discrete_action_bins))
    return float(values[int(np.clip(idx, 0, len(values) - 1))])


def stability_margin(raw_state: Any, args: argparse.Namespace) -> float:
    raw = np.asarray(raw_state, dtype=float).reshape(-1)
    n = int(args.n_poles)
    if raw.size < 2 + 2 * n:
        return -10.0
    x = abs(float(raw[0])) / 2.4
    theta = np.abs(raw[2 : 2 + n]) / 0.20943951023931953
    theta_dot = np.abs(raw[2 + n : 2 + 2 * n]) / 5.0
    angle_pressure = float(np.max(theta)) if theta.size else 0.0
    velocity_pressure = float(np.mean(theta_dot)) if theta_dot.size else 0.0
    return float(1.0 - angle_pressure - 0.10 * x - 0.03 * velocity_pressure)


def residual_observation(args: argparse.Namespace, raw_state: Any, base_force: float, step: int) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    policy_obs = policy_observation_from_state(
        raw,
        raw,
        args.n_poles,
        args.base_observation_mode,
        previous_force=base_force,
        force_mag=args.force_mag,
    )
    aux = residual_aux_features(
        raw,
        n_poles=args.n_poles,
        force_mag=args.force_mag,
        base_force=base_force,
        previous_force=base_force,
        horizon=args.horizon,
        episode_step=step,
        mode=args.residual_feature_mode,
    )
    return np.concatenate([policy_obs.astype(np.float32), aux]).astype(np.float32, copy=False)


def collect_episode_states(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env = make_env(args)
    controller = make_controller(args)
    obs, info = env.reset(seed=seed)
    controller.start_episode()
    states: list[dict[str, Any]] = []
    total = 0.0
    for step in range(args.horizon):
        raw_before = np.asarray(info["raw_state"], dtype=float).copy()
        action, diagnostics = controller.act(obs, raw_before)
        states.append(
            {
                "step": step,
                "raw_before": raw_before.tolist(),
                "action": int(action),
                "force": float(diagnostics.get("force", 0.0)),
                "selected_regime": diagnostics.get("selected_regime", ""),
            }
        )
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += float(reward)
        if terminated or truncated:
            return {"seed": seed, "steps": step + 1, "return": total, "success": bool(truncated and step + 1 >= args.horizon), "states": states}
    return {"seed": seed, "steps": args.horizon, "return": total, "success": True, "states": states}


def counterfactual_score(args: argparse.Namespace, raw_state: list[float], step: int, base_force: float, residual_class: int) -> dict[str, Any]:
    max_shift = int(args.residual_action_bins) // 2
    shift = int(residual_class) - max_shift
    base_idx = force_to_index(base_force, args)
    first_action = int(np.clip(base_idx + shift, 0, int(args.discrete_action_bins) - 1))
    env = make_env(args, force_noise=0.0 if args.counterfactual_no_noise else args.force_noise)
    controller = make_controller(args)
    set_env_state(env, raw_state, step)
    obs = env._get_obs()
    controller.start_episode()
    obs, _reward, terminated, truncated, info = env.step(first_action)
    survived = 1
    final_raw = np.asarray(info.get("raw_state", []), dtype=float)
    if not (terminated or truncated):
        for _ in range(max(1, int(args.probe_horizon)) - 1):
            raw = np.asarray(info["raw_state"], dtype=float).copy()
            action, _diagnostics = controller.act(obs, raw)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            survived += 1
            final_raw = np.asarray(info.get("raw_state", []), dtype=float)
            if terminated or truncated:
                break
    margin = stability_margin(final_raw, args)
    score = float(survived) + float(args.margin_weight) * margin - float(args.shift_penalty) * abs(shift)
    return {"class": int(residual_class), "shift": shift, "first_action": first_action, "survived": int(survived), "margin": float(margin), "score": score}


def label_state(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    options = [counterfactual_score(args, state["raw_before"], int(state["step"]), float(state["force"]), cls) for cls in range(int(args.residual_action_bins))]
    center = int(args.residual_action_bins) // 2
    scores = [float(item["score"]) for item in options]
    best_score = max(scores)
    center_score = scores[center]
    best_classes = [idx for idx, value in enumerate(scores) if abs(value - best_score) <= float(args.score_tolerance)]
    label = center
    if (best_score - center_score) >= float(args.min_score_gap) and center not in best_classes:
        label = int(best_classes[0])
    feature = residual_observation(args, state["raw_before"], float(state["force"]), int(state["step"]))
    return {
        "feature": feature.tolist(),
        "label": int(label),
        "seed": int(state.get("seed", -1)),
        "step": int(state["step"]),
        "base_force": float(state["force"]),
        "options": options,
        "center_score": float(center_score),
        "best_score": float(best_score),
        "score_gap": float(best_score - center_score),
    }



def collect_seed_values(args: argparse.Namespace) -> list[int]:
    seed_list = str(getattr(args, "collect_seed_list", "") or "").strip()
    if not seed_list:
        return [int(args.collect_seed_start) + idx for idx in range(int(args.collect_episodes))]
    path = Path(seed_list)
    raw = path.read_text(encoding="utf-8")
    seeds: list[int] = []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("hard_seeds", "seeds", "tail_seeds"):
            if isinstance(payload.get(key), list):
                seeds = [int(item) for item in payload[key]]
                break
    elif isinstance(payload, list):
        seeds = [int(item) for item in payload]
    if not seeds:
        for item in raw.replace(",", "\n").splitlines():
            value = item.strip()
            if value:
                seeds.append(int(value))
    return seeds[: int(args.collect_episodes)]

def collect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in collect_seed_values(args):
        episode = collect_episode_states(args, int(seed))
        episodes.append({key: episode[key] for key in ("seed", "steps", "return", "success")})
        states = episode["states"]
        selected: list[dict[str, Any]] = []
        if episode["success"]:
            if int(args.success_negative_stride) > 0:
                selected = states[:: int(args.success_negative_stride)][-int(args.max_success_states) :]
                for state in selected:
                    feature = residual_observation(args, state["raw_before"], float(state["force"]), int(state["step"]))
                    rows.append({"feature": feature.tolist(), "label": int(args.residual_action_bins) // 2, "seed": episode["seed"], "step": int(state["step"]), "success_negative": True})
        else:
            seen: set[int] = set()
            for offset in args.failure_offsets:
                pos = len(states) - 1 - int(offset)
                if 0 <= pos < len(states) and int(states[pos]["step"]) not in seen:
                    seen.add(int(states[pos]["step"]))
                    state = dict(states[pos])
                    state["seed"] = episode["seed"]
                    selected.append(state)
            for state in selected[-int(args.max_failure_states) :]:
                rows.append(label_state(args, state))
        partial = {"episodes": episodes, "rows": len(rows), "positive_rows": sum(1 for row in rows if int(row["label"]) != int(args.residual_action_bins) // 2)}
        (out / "partial_dataset.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    return {"episodes": episodes, "rows": rows}


def train_model(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    if not rows:
        raise ValueError("no counterfactual residual rows collected")
    x = torch.tensor([row["feature"] for row in rows], dtype=torch.float32)
    y = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    classes = int(args.residual_action_bins)
    model = nn.Sequential(nn.Linear(x.shape[1], int(args.hidden_size)), nn.ReLU(), nn.Linear(int(args.hidden_size), classes))
    counts = torch.bincount(y, minlength=classes).float()
    weights = torch.clamp(counts.sum() / torch.clamp(counts, min=1.0), max=float(args.max_class_weight))
    center = classes // 2
    weights[center] = min(float(weights[center]), float(args.noop_class_weight))
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    rng = torch.Generator().manual_seed(int(args.train_seed))
    for _ in range(int(args.epochs)):
        order = torch.randperm(x.shape[0], generator=rng)
        for start in range(0, x.shape[0], int(args.batch_size)):
            batch = order[start : start + int(args.batch_size)]
            loss = loss_fn(model(x[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        acc = float((pred == y).float().mean().item())
        non_noop = y != center
        non_noop_recall = float((pred[non_noop] == y[non_noop]).float().mean().item()) if bool(non_noop.any()) else 0.0
    meta = {
        "input_size": int(x.shape[1]),
        "hidden_size": int(args.hidden_size),
        "classes": classes,
        "label_counts": {str(i): int(counts[i].item()) for i in range(classes)},
        "train_accuracy": acc,
        "non_noop_recall": non_noop_recall,
        "format": "counterfactual_residual_terminal_v1",
    }
    return model, meta


def save_model(model: Any, meta: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def tail_metrics(steps: list[float], horizon: int) -> dict[str, Any]:
    summary = summarize_steps(steps, horizon)
    values = np.asarray(steps, dtype=float)
    if values.size:
        count = max(1, int(np.ceil(values.size * 0.10)))
        summary["cvar_survival"] = float(np.mean(np.sort(values)[:count]))
    else:
        summary["cvar_survival"] = 0.0
    return summary


def eval_seed_values(args: argparse.Namespace) -> list[int]:
    starts = args.eval_seed_starts or [args.eval_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.eval_episodes)))
    return seeds


def evaluate_controller(args: argparse.Namespace, residual_path: str = "", gate_threshold: float | None = None) -> dict[str, Any]:
    controller = make_controller(args, residual_path, gate_threshold)
    steps: list[float] = []
    returns: list[float] = []
    residual_deltas: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed in eval_seed_values(args):
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=True if residual_path else False)
        step_count = float(result["steps"])
        steps.append(step_count)
        returns.append(float(result["return"]))
        if residual_path:
            for item in result.get("trace", []):
                residual = ((item.get("policy_terminal") or {}).get("residual_policy_terminal") or {})
                residual_deltas.append(abs(float(residual.get("residual_delta", 0.0) or 0.0)))
        per_seed.append({"seed": int(seed), "steps": int(step_count), "success": step_count >= args.horizon})
    summary = tail_metrics(steps, args.horizon)
    summary.update({"episodes": len(steps), "returns_mean": float(np.mean(returns)) if returns else 0.0, "mean_abs_residual_delta": float(np.mean(residual_deltas)) if residual_deltas else 0.0, "per_seed": per_seed})
    return summary


def label_summary(rows: list[dict[str, Any]], classes: int) -> dict[str, Any]:
    counts = {str(idx): 0 for idx in range(classes)}
    gaps: list[float] = []
    for row in rows:
        label = int(row.get("label", 0))
        counts[str(label)] = counts.get(str(label), 0) + 1
        gaps.append(float(row.get("score_gap", 0.0)))
    center = classes // 2
    return {"row_count": len(rows), "label_counts": counts, "non_noop_count": sum(v for k, v in counts.items() if int(k) != center), "max_score_gap": max(gaps) if gaps else 0.0, "mean_score_gap": float(np.mean(gaps)) if gaps else 0.0}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect_dataset(args)
    rows = data["rows"]
    model, meta = train_model(rows, args)
    model_path = out / "counterfactual_residual_terminal.pt"
    save_model(model, meta, model_path)
    base_eval = evaluate_controller(args)
    residual_eval = evaluate_controller(args, str(model_path), args.residual_gate_threshold)
    result = {
        "status": "completed",
        "base_model_path": args.base_model_path,
        "residual_model_path": str(model_path),
        "dataset": {**label_summary(rows, int(args.residual_action_bins)), "episodes": data["episodes"], "meta": meta},
        "base_eval": base_eval,
        "residual_eval": residual_eval,
        "eval_seeds": eval_seed_values(args),
        "mechanisms": {"counterfactual_residual_terminal": True, "recon_integration_eval": True, "gain_mutation": False},
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "report.md")
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    base = result["base_eval"]
    residual = result["residual_eval"]
    ds = result["dataset"]
    lines = [
        "# Counterfactual Residual Terminal",
        "",
        f"Status: `{result['status']}`",
        f"Residual model: `{result['residual_model_path']}`",
        f"Rows: `{ds['row_count']}`, non-noop labels: `{ds['non_noop_count']}`",
        f"Label counts: `{ds['label_counts']}`",
        "",
        "| evaluator | mean | p10 | cvar | success | mean abs delta | episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| recon_frozen_base | {base['mean_survival']:.1f} | {base['p10_survival']:.1f} | {base.get('cvar_survival', 0.0):.1f} | {base['success_rate']:.3f} | 0.000 | {base['episodes']} |",
        f"| recon_counterfactual_residual | {residual['mean_survival']:.1f} | {residual['p10_survival']:.1f} | {residual.get('cvar_survival', 0.0):.1f} | {residual['success_rate']:.3f} | {residual.get('mean_abs_residual_delta', 0.0):.3f} | {residual['episodes']} |",
        "",
        "## Claim Discipline",
        "",
        "Residual labels come from short-horizon counterfactual probes on collection seeds. Evaluation uses separate held-out seeds and the normal ReCoN residual-terminal integration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a residual terminal from short-horizon counterfactual residual-bin probes.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--base-normalizer-path", default="")
    parser.add_argument("--base-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics"], default="proposal_diagnostics")
    parser.add_argument("--residual-action-bins", type=int, default=5)
    parser.add_argument("--residual-gate-threshold", type=float, default=0.60)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--collect-seed-start", type=int, default=2380000)
    parser.add_argument("--collect-seed-list", default="")
    parser.add_argument("--collect-episodes", type=int, default=80)
    parser.add_argument("--failure-offsets", type=int, nargs="*", default=[0, 2, 5, 10, 20, 40])
    parser.add_argument("--max-failure-states", type=int, default=6)
    parser.add_argument("--success-negative-stride", type=int, default=80)
    parser.add_argument("--max-success-states", type=int, default=3)
    parser.add_argument("--probe-horizon", type=int, default=100)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--shift-penalty", type=float, default=0.02)
    parser.add_argument("--min-score-gap", type=float, default=0.03)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--counterfactual-no-noise", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--noop-class-weight", type=float, default=1.0)
    parser.add_argument("--train-seed", type=int, default=2380)
    parser.add_argument("--eval-seed-start", type=int, default=2100000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--out", default="reports/counterfactual_residual_terminal")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"out": result["residual_model_path"], "base_success": result["base_eval"]["success_rate"], "residual_success": result["residual_eval"]["success_rate"], "non_noop": result["dataset"]["non_noop_count"]}, indent=2))


if __name__ == "__main__":
    main()
