from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.policy_observation import (
    POLICY_OBSERVATION_MODES,
    policy_observation_from_state,
)
from recon_cartpole.control.residual_features import residual_aux_features
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps


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


def make_controller(args: argparse.Namespace) -> ReConCartPoleController:
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
            policy_terminal_path=args.model_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_observation_mode,
        )
    )


def action_force(action: int, args: argparse.Namespace) -> float:
    bins = max(2, int(args.discrete_action_bins))
    idx = int(np.clip(action, 0, bins - 1))
    if bins == 2:
        return float(args.force_mag if idx == 1 else -args.force_mag)
    return float(np.linspace(-args.force_mag, args.force_mag, bins)[idx])


def set_env_state(env: CartPoleNEnv, raw_state: list[float], step: int) -> None:
    env.state = np.asarray(raw_state, dtype=float).copy()
    env.steps = int(step)


def gate_features(raw_state: Any, base_action: int, base_force: float, step: int, args: argparse.Namespace) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    policy_obs = policy_observation_from_state(
        raw,
        raw,
        args.n_poles,
        args.policy_observation_mode,
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
    action_one_hot = np.zeros(int(args.discrete_action_bins), dtype=np.float32)
    action_one_hot[int(np.clip(base_action, 0, args.discrete_action_bins - 1))] = 1.0
    return np.concatenate([policy_obs.astype(np.float32), aux, action_one_hot]).astype(np.float32, copy=False)


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
        action_int = int(action)
        states.append(
            {
                "step": step,
                "raw_before": raw_before.tolist(),
                "obs_before": np.asarray(obs, dtype=float).tolist(),
                "action": action_int,
                "force": float(diagnostics.get("force", action_force(action_int, args))),
                "selected_regime": diagnostics.get("selected_regime", ""),
            }
        )
        obs, reward, terminated, truncated, info = env.step(action_int)
        total += float(reward)
        if terminated or truncated:
            return {"seed": seed, "steps": step + 1, "return": total, "success": bool(truncated and step + 1 >= args.horizon), "states": states}
    return {"seed": seed, "steps": args.horizon, "return": total, "success": True, "states": states}


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


def counterfactual_score(args: argparse.Namespace, raw_state: list[float], step: int, first_action: int) -> dict[str, Any]:
    env = make_env(args, force_noise=0.0 if args.counterfactual_no_noise else args.force_noise)
    controller = make_controller(args)
    set_env_state(env, raw_state, step)
    obs = env._get_obs()  # exact-state counterfactual probe
    controller.start_episode()
    obs, _reward, terminated, truncated, info = env.step(int(first_action))
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
    return {"action": int(first_action), "survived": int(survived), "margin": float(margin), "score": float(survived) + float(args.margin_weight) * float(margin)}


def label_state(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    chosen = int(state["action"])
    options = [counterfactual_score(args, state["raw_before"], int(state["step"]), action) for action in range(args.discrete_action_bins)]
    survivals = [int(item["survived"]) for item in options]
    scores = [float(item["score"]) for item in options]
    best_score = max(scores)
    chosen_score = scores[chosen]
    best_survived = max(survivals)
    chosen_survived = survivals[chosen]
    best_actions = [idx for idx, value in enumerate(scores) if abs(value - best_score) <= float(args.score_tolerance)]
    label = 0
    if (best_score - chosen_score) >= float(args.min_score_gap) and chosen not in best_actions:
        label = int(best_actions[0]) + 1
    feature = gate_features(state["raw_before"], chosen, float(state["force"]), int(state["step"]), args)
    return {
        "feature": feature.tolist(),
        "label": label,
        "seed": state.get("seed"),
        "step": int(state["step"]),
        "chosen_action": chosen,
        "best_actions": best_actions,
        "survivals": survivals,
        "scores": scores,
        "chosen_score": chosen_score,
        "best_score": best_score,
        "chosen_survived": chosen_survived,
        "best_survived": best_survived,
    }


def collect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for idx in range(args.collect_episodes):
        episode = collect_episode_states(args, args.collect_seed_start + idx)
        episodes.append({k: episode[k] for k in ("seed", "steps", "return", "success")})
        states = episode["states"]
        selected: list[dict[str, Any]] = []
        if episode["success"]:
            if args.success_negative_stride > 0:
                selected = states[:: args.success_negative_stride][-args.max_success_states :]
        else:
            for offset in args.failure_offsets:
                pos = len(states) - 1 - int(offset)
                if 0 <= pos < len(states):
                    selected.append(states[pos])
            selected = selected[-args.max_failure_states :]
        seen: set[int] = set()
        for state in selected:
            step = int(state["step"])
            if step in seen:
                continue
            seen.add(step)
            state = dict(state)
            state["seed"] = episode["seed"]
            if episode["success"]:
                feature = gate_features(state["raw_before"], int(state["action"]), float(state["force"]), step, args)
                rows.append({"feature": feature.tolist(), "label": 0, "seed": state["seed"], "step": step, "chosen_action": int(state["action"]), "success_negative": True})
            else:
                rows.append(label_state(args, state))
        partial = {"episodes": episodes, "rows": len(rows), "positive_rows": sum(1 for row in rows if int(row["label"]) > 0)}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "partial_dataset.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    return {"episodes": episodes, "rows": rows}


class GateNetConfig(dict):
    pass


def train_gate(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    if not rows:
        raise ValueError("no gate training rows collected")
    x = torch.tensor([row["feature"] for row in rows], dtype=torch.float32)
    y = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    classes = int(args.discrete_action_bins) + 1
    model = nn.Sequential(
        nn.Linear(x.shape[1], int(args.hidden_size)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_size), classes),
    )
    counts = torch.bincount(y, minlength=classes).float()
    weights = torch.ones(classes, dtype=torch.float32)
    if counts[1:].sum() > 0:
        weights = torch.clamp(counts.sum() / torch.clamp(counts, min=1.0), max=float(args.max_class_weight))
        weights[0] = min(float(weights[0]), float(args.no_override_weight))
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(int(args.epochs)):
        order = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], int(args.batch_size)):
            batch = order[start : start + int(args.batch_size)]
            loss = loss_fn(model(x[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        acc = float((pred == y).float().mean().item())
        positive_recall = 0.0
        positive_mask = y > 0
        if bool(positive_mask.any()):
            positive_recall = float((pred[positive_mask] == y[positive_mask]).float().mean().item())
    meta = {
        "input_size": int(x.shape[1]),
        "classes": classes,
        "label_counts": {str(i): int(counts[i].item()) for i in range(classes)},
        "train_accuracy": acc,
        "positive_recall": positive_recall,
        "hidden_size": int(args.hidden_size),
    }
    return model, meta


def save_gate(model: Any, meta: dict[str, Any], path: Path) -> None:
    import torch

    payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, path)


def load_gate(path: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    model = nn.Sequential(
        nn.Linear(int(meta["input_size"]), int(meta["hidden_size"])),
        nn.ReLU(),
        nn.Linear(int(meta["hidden_size"]), int(meta["classes"])),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


def evaluate_base_or_gate(args: argparse.Namespace, seeds: list[int], gate_path: Path | None = None) -> dict[str, Any]:
    import torch

    model = None
    if gate_path is not None:
        model, _meta = load_gate(gate_path)
    steps: list[float] = []
    returns: list[float] = []
    overrides = 0
    checked = 0
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_env(args)
        controller = make_controller(args)
        obs, info = env.reset(seed=seed)
        controller.start_episode()
        total = 0.0
        episode_overrides = 0
        for step in range(args.horizon):
            raw_before = np.asarray(info["raw_state"], dtype=float).copy()
            action, diagnostics = controller.act(obs, raw_before)
            action_int = int(action)
            final_action = action_int
            if model is not None:
                force = float(diagnostics.get("force", action_force(action_int, args)))
                feat = gate_features(raw_before, action_int, force, step, args)
                with torch.no_grad():
                    probs = torch.softmax(model(torch.tensor(feat, dtype=torch.float32).unsqueeze(0)), dim=1).numpy()[0]
                cls = int(np.argmax(probs))
                confidence = float(probs[cls])
                checked += 1
                if cls > 0 and confidence >= float(args.gate_confidence):
                    final_action = cls - 1
                    if final_action != action_int:
                        overrides += 1
                        episode_overrides += 1
            obs, reward, terminated, truncated, info = env.step(final_action)
            total += float(reward)
            if terminated or truncated:
                step_count = step + 1
                break
        else:
            step_count = args.horizon
        steps.append(float(step_count))
        returns.append(total)
        per_seed.append({"seed": int(seed), "steps": int(step_count), "success": step_count >= args.horizon, "overrides": int(episode_overrides)})
    summary = summarize_steps(steps, args.horizon)
    values = np.asarray(steps, dtype=float)
    if values.size:
        count = max(1, int(np.ceil(values.size * 0.10)))
        summary["cvar_survival"] = float(np.mean(np.sort(values)[:count]))
    summary.update({"episodes": len(seeds), "returns_mean": float(np.mean(returns)) if returns else 0.0, "override_count": overrides, "checked_steps": checked, "override_rate": float(overrides / checked) if checked else 0.0, "per_seed": per_seed})
    return summary


def eval_seeds(args: argparse.Namespace) -> list[int]:
    starts = args.eval_seed_starts or [args.eval_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.eval_episodes)))
    return seeds


def dataset_label_summary(rows: list[dict[str, Any]], classes: int) -> dict[str, Any]:
    counts = {str(idx): 0 for idx in range(classes)}
    positive_count = 0
    max_survival_gap = 0.0
    max_score_gap = 0.0
    for row in rows:
        label = int(row.get("label", 0))
        counts[str(label)] = counts.get(str(label), 0) + 1
        if label > 0:
            positive_count += 1
        max_survival_gap = max(max_survival_gap, float(row.get("best_survived", 0.0)) - float(row.get("chosen_survived", 0.0)))
        max_score_gap = max(max_score_gap, float(row.get("best_score", 0.0)) - float(row.get("chosen_score", 0.0)))
    return {
        "row_count": len(rows),
        "positive_count": positive_count,
        "label_counts": counts,
        "max_survival_gap": max_survival_gap,
        "max_score_gap": max_score_gap,
    }


def gate_eval_from_base(base_eval: dict[str, Any]) -> dict[str, Any]:
    copied = dict(base_eval)
    copied["override_count"] = 0
    copied["checked_steps"] = 0
    copied["override_rate"] = 0.0
    return copied


def write_markdown(result: dict[str, Any], path: Path) -> None:
    base = result["base_eval"]
    gate = result["gate_eval"]
    lines = [
        "# Counterfactual Action Gate",
        "",
        f"Status: `{result['status']}`",
        f"Model: `{result['model_path']}`",
        f"Gate path: `{result['gate_path']}`",
        f"Training rows: `{result['dataset']['row_count']}`, positives: `{result['dataset']['positive_count']}`",
        f"Label counts: `{result['dataset'].get('label_counts', {})}`",
        f"Max survival gap: `{result['dataset'].get('max_survival_gap', 0.0):.3f}`",
        f"Max score gap: `{result['dataset'].get('max_score_gap', 0.0):.6g}`",
        f"Gate confidence: `{result['gate_confidence']}`",
        "",
        "| evaluator | mean | p10 | cvar | success | overrides | override rate | episodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| base_recon | {base['mean_survival']:.1f} | {base['p10_survival']:.1f} | {base.get('cvar_survival', 0.0):.1f} | {base['success_rate']:.3f} | 0 | 0.000 | {base['episodes']} |",
        f"| gated_recon | {gate['mean_survival']:.1f} | {gate['p10_survival']:.1f} | {gate.get('cvar_survival', 0.0):.1f} | {gate['success_rate']:.3f} | {gate['override_count']} | {gate['override_rate']:.4f} | {gate['episodes']} |",
        "",
        "## Claim Discipline",
        "",
        "The gate is trained from short-horizon counterfactual probes near failures and evaluated on separately requested seed blocks. It is not a solve claim unless held-out metrics clear the configured solve threshold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect_dataset(args)
    rows = data["rows"]
    label_summary = dataset_label_summary(rows, int(args.discrete_action_bins) + 1)
    seeds = eval_seeds(args)
    base_eval = evaluate_base_or_gate(args, seeds)
    meta: dict[str, Any] = {}
    gate_path: Path | None = None
    if label_summary["positive_count"] > 0:
        model, meta = train_gate(rows, args)
        gate_path = out / "counterfactual_action_gate.pt"
        save_gate(model, meta, gate_path)
        gate_eval = evaluate_base_or_gate(args, seeds, gate_path)
        status = "completed"
    else:
        gate_eval = gate_eval_from_base(base_eval)
        status = "completed_no_positive_labels"
    result = {
        "status": status,
        "model_path": args.model_path,
        "gate_path": str(gate_path) if gate_path is not None else "",
        "gate_confidence": args.gate_confidence,
        "dataset": {**label_summary, "episodes": data["episodes"], "meta": meta},
        "base_eval": base_eval,
        "gate_eval": gate_eval,
        "eval_seeds": seeds,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "report.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a learned action override gate from near-failure counterfactual probes.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--policy-observation-mode", choices=POLICY_OBSERVATION_MODES, default="normalized_raw")
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics", "subchain_diagnostics"], default="proposal_diagnostics")
    parser.add_argument("--collect-seed-start", type=int, default=980000)
    parser.add_argument("--collect-episodes", type=int, default=80)
    parser.add_argument("--failure-offsets", type=int, nargs="*", default=[0, 2, 5, 10, 20, 40])
    parser.add_argument("--max-failure-states", type=int, default=6)
    parser.add_argument("--success-negative-stride", type=int, default=60)
    parser.add_argument("--max-success-states", type=int, default=4)
    parser.add_argument("--probe-horizon", type=int, default=80)
    parser.add_argument("--min-survival-gap", type=int, default=5)  # kept for report compatibility; score gap drives labels
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--counterfactual-no-noise", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--no-override-weight", type=float, default=1.0)
    parser.add_argument("--gate-confidence", type=float, default=0.70)
    parser.add_argument("--eval-seed-start", type=int, default=1500000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--out", default="reports/counterfactual_action_gate")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"out": result["gate_path"], "base_success": result["base_eval"]["success_rate"], "gate_success": result["gate_eval"]["success_rate"], "overrides": result["gate_eval"]["override_count"]}, indent=2))


if __name__ == "__main__":
    main()
