from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.policy_observation import POLICY_OBSERVATION_MODES, policy_observation_from_state
from recon_cartpole.control.residual_features import residual_aux_features
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_recurrent_terminal_ladder import terminal_config  # noqa: E402


def make_env(args: argparse.Namespace, *, force_noise: float | None = None) -> CartPoleNEnv:
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
            mode="recon_mingru_terminal",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            mingru_terminal=terminal_config(args, args.checkpoint_path, args.hidden_size, args.sequence_length),
        )
    )


def action_force(action: int, args: argparse.Namespace) -> float:
    idx = int(np.clip(action, 0, int(args.discrete_action_bins) - 1))
    return float(np.linspace(-args.force_mag, args.force_mag, int(args.discrete_action_bins))[idx])


def failure_class(raw_state: Any, args: argparse.Namespace) -> str:
    raw = np.asarray(raw_state, dtype=float).reshape(-1)
    if raw.size < 2 + 2 * int(args.n_poles):
        return "unknown"
    if abs(float(raw[0])) > 2.2:
        return "rail_left" if float(raw[0]) < 0.0 else "rail_right"
    angles = raw[2 : 2 + int(args.n_poles)]
    if angles.size:
        return f"pole_{int(np.argmax(np.abs(angles)))}_angle"
    return "unknown"


def logit_margin(values: Any) -> float:
    logits = [float(item) for item in values or [] if np.isfinite(float(item))]
    if len(logits) < 2:
        return 0.0
    top = sorted(logits, reverse=True)[:2]
    return float(top[0] - top[1])


def gate_features(
    raw_state: Any,
    base_action: int,
    base_force: float,
    diagnostics: dict[str, Any],
    step: int,
    args: argparse.Namespace,
) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    policy_obs = policy_observation_from_state(
        raw,
        raw,
        args.n_poles,
        args.observation_mode,
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
    action_one_hot = np.zeros(int(args.discrete_action_bins), dtype=np.float32)
    action_one_hot[int(np.clip(base_action, 0, int(args.discrete_action_bins) - 1))] = 1.0
    mingru = diagnostics.get("mingru_terminal", {}) or diagnostics.get("mingru_passthrough", {}) or {}
    diag = np.asarray(
        [
            float(mingru.get("confidence", 0.0)),
            float(mingru.get("failure_probability", 0.0)),
            float(mingru.get("value", 0.0)),
            float(mingru.get("hidden_norm", 0.0)) / 10.0,
            logit_margin(mingru.get("logits", [])),
            float(step) / max(1.0, float(args.horizon)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([policy_obs.astype(np.float32), aux, action_one_hot, diag]).astype(np.float32)


def stability_margin(raw_state: Any, args: argparse.Namespace) -> float:
    raw = np.asarray(raw_state, dtype=float).reshape(-1)
    n = int(args.n_poles)
    if raw.size < 2 + 2 * n:
        return -10.0
    x_pressure = abs(float(raw[0])) / 2.4
    theta = np.abs(raw[2 : 2 + n]) / 0.20943951023931953
    theta_dot = np.abs(raw[2 + n : 2 + 2 * n]) / 5.0
    return float(1.0 - float(np.max(theta)) - 0.10 * x_pressure - 0.03 * float(np.mean(theta_dot)))


def rollout_episode(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env = make_env(args)
    controller = make_controller(args)
    obs, info = env.reset(seed=seed)
    controller.start_episode()
    total = 0.0
    states: list[dict[str, Any]] = []
    final_raw: Any = info.get("raw_state", [])
    for step in range(int(args.horizon)):
        raw_before = np.asarray(info["raw_state"], dtype=float).copy()
        action, diagnostics = controller.act(obs, raw_before)
        action_int = int(action)
        states.append(
            {
                "seed": int(seed),
                "step": int(step),
                "raw_before": raw_before.tolist(),
                "action": action_int,
                "force": float(diagnostics.get("force", action_force(action_int, args))),
                "diagnostics": diagnostics,
            }
        )
        obs, reward, terminated, truncated, info = env.step(action_int)
        total += float(reward)
        final_raw = info.get("raw_state", final_raw)
        if terminated or truncated:
            success = bool(truncated and step + 1 >= int(args.horizon))
            return {
                "seed": int(seed),
                "steps": step + 1,
                "return": total,
                "success": success,
                "failure": "success" if success else failure_class(final_raw, args),
                "states": states,
            }
    return {"seed": int(seed), "steps": int(args.horizon), "return": total, "success": True, "failure": "success", "states": states}


def counterfactual_score(args: argparse.Namespace, seed: int, target_step: int, first_action: int) -> dict[str, Any]:
    env = make_env(args, force_noise=0.0 if args.counterfactual_no_noise else None)
    controller = make_controller(args)
    obs, info = env.reset(seed=seed)
    controller.start_episode()
    total = 0.0
    final_raw: Any = info.get("raw_state", [])
    for step in range(int(target_step)):
        raw = np.asarray(info["raw_state"], dtype=float).copy()
        action, _diagnostics = controller.act(obs, raw)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += float(reward)
        final_raw = info.get("raw_state", final_raw)
        if terminated or truncated:
            return {"action": int(first_action), "survived": 0, "margin": stability_margin(final_raw, args), "score": -1.0}
    # Advance recurrent state at each forced tick, but override the environment action.
    forced_steps = max(1, int(getattr(args, "forced_action_hold_steps", 1)))
    survived = 0
    terminated = False
    truncated = False
    for _ in range(forced_steps):
        raw = np.asarray(info["raw_state"], dtype=float).copy()
        _base_action, _diagnostics = controller.act(obs, raw)
        obs, reward, terminated, truncated, info = env.step(int(first_action))
        total += float(reward)
        survived += 1
        final_raw = info.get("raw_state", final_raw)
        if controller.mingru_terminal is not None:
            controller.mingru_terminal.prev_force = action_force(int(first_action), args)
        if terminated or truncated:
            break
    if not (terminated or truncated):
        for _ in range(max(1, int(args.probe_horizon)) - survived):
            raw = np.asarray(info["raw_state"], dtype=float).copy()
            action, _diagnostics = controller.act(obs, raw)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            survived += 1
            final_raw = info.get("raw_state", final_raw)
            if terminated or truncated:
                break
    margin = stability_margin(final_raw, args)
    return {
        "action": int(first_action),
        "survived": int(survived),
        "margin": float(margin),
        "score": float(survived) + float(args.margin_weight) * float(margin),
    }


def _gate_threshold(args: argparse.Namespace, name: str, default: float | int) -> float | int:
    value = getattr(args, name, None)
    return default if value is None else value


def label_state(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    chosen = int(state["action"])
    options = [counterfactual_score(args, int(state["seed"]), int(state["step"]), action) for action in range(int(args.discrete_action_bins))]
    scores = [float(item["score"]) for item in options]
    survivals = [int(item["survived"]) for item in options]
    margins = [float(item["margin"]) for item in options]
    best_score = max(scores)
    chosen_score = scores[chosen]
    chosen_survival = survivals[chosen]
    chosen_margin = margins[chosen]
    best_actions = [idx for idx, value in enumerate(scores) if abs(value - best_score) <= float(args.score_tolerance)]
    label = 0
    if (best_score - chosen_score) >= float(args.min_score_gap) and chosen not in best_actions:
        label = int(best_actions[0]) + 1
    target_action = chosen if label == 0 else label - 1
    target_score_gap = scores[target_action] - chosen_score
    target_survival_gain = survivals[target_action] - chosen_survival
    target_margin_gain = margins[target_action] - chosen_margin
    apply_label = int(
        label > 0
        and target_score_gap >= float(_gate_threshold(args, "apply_min_score_gap", float(args.min_score_gap)))
        and target_survival_gain >= int(_gate_threshold(args, "apply_min_survival_gain", 0))
        and target_margin_gain >= float(_gate_threshold(args, "apply_min_margin_gain", 0.0))
    )
    return {
        "feature": gate_features(state["raw_before"], chosen, float(state["force"]), state.get("diagnostics", {}), int(state["step"]), args).tolist(),
        "label": int(label),
        "apply_label": int(apply_label),
        "seed": int(state["seed"]),
        "step": int(state["step"]),
        "chosen_action": chosen,
        "target_action": int(target_action),
        "best_actions": best_actions,
        "scores": scores,
        "survivals": survivals,
        "margins": margins,
        "chosen_score": float(chosen_score),
        "best_score": float(best_score),
        "target_score_gap": float(target_score_gap),
        "target_survival_gain": int(target_survival_gain),
        "target_margin_gain": float(target_margin_gain),
        "best_survival_gain": int(survivals[int(best_actions[0])] - chosen_survival) if best_actions else 0,
        "best_margin_gain": float(margins[int(best_actions[0])] - chosen_margin) if best_actions else 0.0,
    }


def collect_seed_values(args: argparse.Namespace) -> list[int]:
    path_value = str(getattr(args, "collect_seeds_path", "") or "")
    if not path_value:
        return [int(args.collect_seed_start) + idx for idx in range(int(args.collect_episodes))]
    path = Path(path_value)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("hard_seeds", payload if isinstance(payload, list) else [])
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seeds = [int(item["seed"] if isinstance(item, dict) else item) for item in values]
    return seeds[: int(args.collect_episodes)]


def collect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    target_failures = set(args.failure_classes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in collect_seed_values(args):
        episode = rollout_episode(args, int(seed))
        episodes.append({k: episode[k] for k in ("seed", "steps", "return", "success", "failure")})
        states = episode["states"]
        selected: list[dict[str, Any]] = []
        if bool(episode["success"]):
            if int(args.success_negative_stride) > 0:
                selected = states[:: int(args.success_negative_stride)][-int(args.max_success_states) :]
                for state in selected:
                    rows.append(
                        {
                            "feature": gate_features(state["raw_before"], int(state["action"]), float(state["force"]), state.get("diagnostics", {}), int(state["step"]), args).tolist(),
                            "label": 0,
                            "apply_label": 0,
                            "seed": int(state["seed"]),
                            "step": int(state["step"]),
                            "chosen_action": int(state["action"]),
                            "success_negative": True,
                        }
                    )
        elif episode["failure"] in target_failures:
            for offset in args.failure_offsets:
                pos = len(states) - 1 - int(offset)
                if 0 <= pos < len(states):
                    selected.append(states[pos])
            selected = selected[-int(args.max_failure_states) :]
            seen: set[int] = set()
            for state in selected:
                step = int(state["step"])
                if step in seen:
                    continue
                seen.add(step)
                rows.append(label_state(args, state))
        partial = {
            "episodes": episodes,
            "row_count": len(rows),
            "positive_count": sum(1 for row in rows if int(row.get("label", 0)) > 0),
            "apply_positive_count": sum(1 for row in rows if int(row.get("apply_label", 0)) > 0),
        }
        (out / "partial_dataset.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    return {"episodes": episodes, "rows": rows}


def train_gate(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    if not rows:
        raise ValueError("no gate training rows collected")
    factor = max(1, int(round(float(getattr(args, "positive_oversample_factor", 1.0)))))
    train_rows = list(rows)
    if factor > 1:
        positive_rows = [row for row in rows if int(row.get("label", 0)) > 0]
        for _ in range(factor - 1):
            train_rows.extend(positive_rows)
    x = torch.tensor([row["feature"] for row in train_rows], dtype=torch.float32)
    y = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    apply_y = torch.tensor([int(row.get("apply_label", int(int(row["label"]) > 0))) for row in train_rows], dtype=torch.float32).reshape(-1, 1)
    classes = int(args.discrete_action_bins) + 1
    model = nn.Sequential(
        nn.Linear(x.shape[1], int(args.hidden_size_gate)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_size_gate), classes),
    )
    counts = torch.bincount(y, minlength=classes).float()
    weights = torch.ones(classes, dtype=torch.float32)
    if counts[1:].sum() > 0:
        weights = torch.clamp(counts.sum() / torch.clamp(counts, min=1.0), max=float(args.max_class_weight))
        weights[0] = min(float(weights[0]), float(args.no_override_weight))
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

    apply_model = None
    apply_accuracy = 0.0
    apply_positive_rate = float(apply_y.mean().item()) if apply_y.numel() else 0.0
    if bool(getattr(args, "train_apply_gate", True)):
        apply_model = nn.Sequential(
            nn.Linear(x.shape[1], int(args.hidden_size_gate)),
            nn.ReLU(),
            nn.Linear(int(args.hidden_size_gate), 1),
        )
        positives = torch.clamp(apply_y.sum(), min=0.0)
        negatives = torch.clamp(torch.tensor(float(apply_y.numel())) - positives, min=0.0)
        pos_weight_value = float(getattr(args, "apply_positive_weight", 0.0))
        if pos_weight_value <= 0.0:
            pos_weight_value = float((negatives / torch.clamp(positives, min=1.0)).item())
        pos_weight = torch.tensor(
            [max(1.0, min(float(getattr(args, "max_apply_positive_weight", 12.0)), pos_weight_value))],
            dtype=torch.float32,
        )
        apply_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        apply_opt = torch.optim.Adam(apply_model.parameters(), lr=float(args.learning_rate))
        apply_rng = torch.Generator().manual_seed(int(args.train_seed) + 17)
        for _ in range(int(getattr(args, "apply_epochs", args.epochs))):
            order = torch.randperm(x.shape[0], generator=apply_rng)
            for start in range(0, x.shape[0], int(args.batch_size)):
                batch = order[start : start + int(args.batch_size)]
                loss = apply_loss_fn(apply_model(x[batch]), apply_y[batch])
                apply_opt.zero_grad()
                loss.backward()
                apply_opt.step()
        with torch.no_grad():
            apply_pred = (torch.sigmoid(apply_model(x)) >= float(args.gate_apply_threshold)).float()
            apply_accuracy = float((apply_pred == apply_y).float().mean().item())

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        positive = y > 0
        recall = float((pred[positive] == y[positive]).float().mean().item()) if bool(positive.any()) else 0.0
        acc = float((pred == y).float().mean().item())
    meta = {
        "format": "mingru_action_gate_v2" if apply_model is not None else "mingru_action_gate_v1",
        "input_size": int(x.shape[1]),
        "hidden_size": int(args.hidden_size_gate),
        "classes": classes,
        "label_counts": {str(idx): int(counts[idx].item()) for idx in range(classes)},
        "apply_label_counts": {"0": int((apply_y == 0).sum().item()), "1": int((apply_y == 1).sum().item())},
        "apply_gate_enabled": apply_model is not None,
        "apply_positive_rate": apply_positive_rate,
        "apply_accuracy": apply_accuracy,
        "apply_threshold": float(args.gate_apply_threshold),
        "original_row_count": int(len(rows)),
        "expanded_row_count": int(len(train_rows)),
        "positive_oversample_factor": int(factor),
        "train_accuracy": acc,
        "positive_recall": recall,
    }
    return (model, apply_model) if apply_model is not None else model, meta


def save_gate(model: Any, meta: dict[str, Any], path: Path) -> None:
    import torch

    if isinstance(model, tuple):
        action_model, apply_model = model
        payload = {"state_dict": action_model.state_dict(), "meta": meta}
        if apply_model is not None:
            payload["apply_state_dict"] = apply_model.state_dict()
    else:
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
    apply_model = None
    if "apply_state_dict" in payload:
        apply_model = nn.Sequential(
            nn.Linear(int(meta["input_size"]), int(meta["hidden_size"])),
            nn.ReLU(),
            nn.Linear(int(meta["hidden_size"]), 1),
        )
        apply_model.load_state_dict(payload["apply_state_dict"])
        apply_model.eval()
    return (model, apply_model) if apply_model is not None else model, meta


def gate_decision_from_probs(
    probs: np.ndarray,
    base_action: int,
    gate_confidence: float,
    gate_margin: float,
    gate_apply_threshold: float,
    apply_probability: float | None = None,
) -> tuple[int, bool]:
    cls = int(np.argmax(probs))
    if cls <= 0:
        return int(base_action), False
    confidence = float(probs[cls])
    margin = confidence - float(probs[0])
    apply_allowed = apply_probability is None or float(apply_probability) >= float(gate_apply_threshold)
    candidate = int(cls - 1)
    should_override = (
        candidate != int(base_action)
        and confidence >= float(gate_confidence)
        and margin >= float(gate_margin)
        and apply_allowed
    )
    return (candidate if should_override else int(base_action)), bool(should_override)


def record_gate_probability_trace(args: argparse.Namespace, seeds: list[int], gate_path: Path) -> dict[str, Any]:
    import torch

    loaded, _meta = load_gate(gate_path)
    if isinstance(loaded, tuple):
        model, apply_model = loaded
    else:
        model, apply_model = loaded, None
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_env(args)
        controller = make_controller(args)
        obs, info = env.reset(seed=int(seed))
        controller.start_episode()
        total = 0.0
        steps: list[dict[str, Any]] = []
        final_raw: Any = info.get("raw_state", [])
        for step in range(int(args.horizon)):
            raw_before = np.asarray(info["raw_state"], dtype=float).copy()
            action, diagnostics = controller.act(obs, raw_before)
            base_action = int(action)
            force = float(diagnostics.get("force", action_force(base_action, args)))
            feat = gate_features(raw_before, base_action, force, diagnostics, step, args)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(feat_tensor), dim=1).cpu().numpy()[0]
                apply_probability = None
                if apply_model is not None:
                    apply_probability = float(torch.sigmoid(apply_model(feat_tensor)).cpu().numpy().reshape(-1)[0])
            steps.append(
                {
                    "step": int(step),
                    "base_action": int(base_action),
                    "probs": [float(value) for value in probs],
                    "apply_probability": apply_probability,
                }
            )
            obs, reward, terminated, truncated, info = env.step(base_action)
            total += float(reward)
            final_raw = info.get("raw_state", final_raw)
            if terminated or truncated:
                step_count = step + 1
                break
        else:
            step_count = int(args.horizon)
        episodes.append(
            {
                "seed": int(seed),
                "steps": int(step_count),
                "return": float(total),
                "success": int(step_count) >= int(args.horizon),
                "failure": "success" if int(step_count) >= int(args.horizon) else failure_class(final_raw, args),
                "trace": steps,
            }
        )
    return {"episodes": episodes, "seeds": [int(seed) for seed in seeds], "gate_path": str(gate_path), "horizon": int(args.horizon)}


def sweep_probability_trace(trace: dict[str, Any], configs: list[dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_steps = sum(len(ep.get("trace", [])) for ep in trace.get("episodes", []))
    base_steps = [float(ep.get("steps", 0)) for ep in trace.get("episodes", [])]
    horizon = int(trace.get("horizon", max(base_steps) if base_steps else 0))
    base_summary = summarize_steps(base_steps, horizon) if base_steps else {}
    for config in configs:
        overrides = 0
        override_episodes = 0
        for episode in trace.get("episodes", []):
            episode_overrides = 0
            for item in episode.get("trace", []):
                _action, should_override = gate_decision_from_probs(
                    np.asarray(item["probs"], dtype=float),
                    int(item["base_action"]),
                    float(config.get("gate_confidence", 0.75)),
                    float(config.get("gate_margin", 0.0)),
                    float(config.get("gate_apply_threshold", 0.5)),
                    item.get("apply_probability"),
                )
                if should_override:
                    overrides += 1
                    episode_overrides += 1
            if episode_overrides:
                override_episodes += 1
        rows.append(
            {
                **config,
                "override_count": int(overrides),
                "override_rate": float(overrides / total_steps) if total_steps else 0.0,
                "override_episodes": int(override_episodes),
                "episodes": len(trace.get("episodes", [])),
                "base_success_rate": float(base_summary.get("success_rate", 0.0)),
                "base_mean_survival": float(base_summary.get("mean_survival", 0.0)),
            }
        )
    return rows

def evaluate(args: argparse.Namespace, seeds: list[int], gate_path: Path | None = None) -> dict[str, Any]:
    import torch

    model = None
    apply_model = None
    if gate_path is not None:
        loaded, _meta = load_gate(gate_path)
        if isinstance(loaded, tuple):
            model, apply_model = loaded
        else:
            model = loaded
    steps: list[float] = []
    returns: list[float] = []
    overrides = 0
    checked = 0
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_env(args)
        controller = make_controller(args)
        obs, info = env.reset(seed=int(seed))
        controller.start_episode()
        total = 0.0
        episode_overrides = 0
        final_raw: Any = info.get("raw_state", [])
        for step in range(int(args.horizon)):
            raw_before = np.asarray(info["raw_state"], dtype=float).copy()
            action, diagnostics = controller.act(obs, raw_before)
            base_action = int(action)
            final_action = base_action
            if model is not None:
                force = float(diagnostics.get("force", action_force(base_action, args)))
                feat = gate_features(raw_before, base_action, force, diagnostics, step, args)
                feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(feat_tensor), dim=1).numpy()[0]
                    apply_probability = None
                    if apply_model is not None:
                        apply_probability = float(torch.sigmoid(apply_model(feat_tensor)).cpu().numpy().reshape(-1)[0])
                checked += 1
                candidate, should_override = gate_decision_from_probs(
                    probs,
                    base_action,
                    float(args.gate_confidence),
                    float(args.gate_margin),
                    float(args.gate_apply_threshold),
                    apply_probability,
                )
                if should_override:
                    final_action = candidate
                    overrides += 1
                    episode_overrides += 1
                    if controller.mingru_terminal is not None:
                        controller.mingru_terminal.prev_force = action_force(final_action, args)
            obs, reward, terminated, truncated, info = env.step(int(final_action))
            total += float(reward)
            final_raw = info.get("raw_state", final_raw)
            if terminated or truncated:
                step_count = step + 1
                break
        else:
            step_count = int(args.horizon)
        steps.append(float(step_count))
        returns.append(float(total))
        per_seed.append(
            {
                "seed": int(seed),
                "steps": int(step_count),
                "success": int(step_count) >= int(args.horizon),
                "failure": "success" if int(step_count) >= int(args.horizon) else failure_class(final_raw, args),
                "overrides": int(episode_overrides),
            }
        )
    summary = summarize_steps(steps, int(args.horizon))
    values = np.asarray(steps, dtype=float)
    cvar_count = max(1, int(np.ceil(values.size * 0.10))) if values.size else 0
    summary.update(
        {
            "cvar_survival": float(np.mean(np.sort(values)[:cvar_count])) if cvar_count else 0.0,
            "episodes": len(seeds),
            "returns_mean": float(np.mean(returns)) if returns else 0.0,
            "override_count": int(overrides),
            "checked_steps": int(checked),
            "override_rate": float(overrides / checked) if checked else 0.0,
            "per_seed": per_seed,
        }
    )
    return summary


def eval_seeds(args: argparse.Namespace) -> list[int]:
    starts = args.eval_seed_starts or [args.eval_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.eval_episodes)))
    return seeds


def dataset_summary(rows: list[dict[str, Any]], classes: int) -> dict[str, Any]:
    counts = {str(idx): 0 for idx in range(classes)}
    for row in rows:
        label = int(row.get("label", 0))
        counts[str(label)] = counts.get(str(label), 0) + 1
    positives = sum(value for key, value in counts.items() if key != "0")
    apply_positives = sum(1 for row in rows if int(row.get("apply_label", 0)) > 0)
    max_gap = 0.0
    for row in rows:
        max_gap = max(max_gap, float(row.get("best_score", 0.0)) - float(row.get("chosen_score", 0.0)))
    return {
        "row_count": len(rows),
        "positive_count": int(positives),
        "apply_positive_count": int(apply_positives),
        "label_counts": counts,
        "max_score_gap": float(max_gap),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    base = result["base_eval"]
    gate = result["gate_eval"]
    lines = [
        "# minGRU Counterfactual Action Gate",
        "",
        f"Status: `{result['status']}`",
        f"Checkpoint: `{result['checkpoint_path']}`",
        f"Gate path: `{result['gate_path']}`",
        f"Training rows: `{result['dataset']['row_count']}`, positives: `{result['dataset']['positive_count']}`, apply positives: `{result['dataset'].get('apply_positive_count', 0)}`",
        f"Label counts: `{result['dataset']['label_counts']}`",
        f"Failure classes: `{result['failure_classes']}`",
        f"Gate confidence: `{result['gate_confidence']}`",
        f"Gate margin: `{result['gate_margin']}`",
        f"Gate apply threshold: `{result['gate_apply_threshold']}`",
        f"Forced action hold steps: `{result['forced_action_hold_steps']}`",
        "",
        "| evaluator | mean | p10 | cvar | success | overrides | override rate | episodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| base_recon_mingru | {base['mean_survival']:.1f} | {base['p10_survival']:.1f} | {base.get('cvar_survival', 0.0):.1f} | {base['success_rate']:.3f} | 0 | 0.0000 | {base['episodes']} |",
        f"| gated_recon_mingru | {gate['mean_survival']:.1f} | {gate['p10_survival']:.1f} | {gate.get('cvar_survival', 0.0):.1f} | {gate['success_rate']:.3f} | {gate['override_count']} | {gate['override_rate']:.4f} | {gate['episodes']} |",
        "",
        "## Claim Discipline",
        "",
        "The gate is trained from recurrent-prefix counterfactual probes near selected failure classes and evaluated on separately requested held-out seeds. It is not a solve claim unless held-out metrics clear the configured solve threshold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect_dataset(args)
    rows = data["rows"]
    summary = dataset_summary(rows, int(args.discrete_action_bins) + 1)
    seeds = eval_seeds(args)
    base_eval = evaluate(args, seeds)
    gate_path: Path | None = None
    meta: dict[str, Any] = {}
    if summary["positive_count"] > 0:
        model, meta = train_gate(rows, args)
        gate_path = out / "mingru_action_gate.pt"
        save_gate(model, meta, gate_path)
        gate_eval = evaluate(args, seeds, gate_path)
        status = "completed"
    else:
        gate_eval = {**base_eval, "override_count": 0, "checked_steps": 0, "override_rate": 0.0}
        status = "completed_no_positive_labels"
    result = {
        "status": status,
        "out": str(out),
        "checkpoint_path": args.checkpoint_path,
        "gate_path": str(gate_path) if gate_path else "",
        "gate_confidence": float(args.gate_confidence),
        "gate_margin": float(args.gate_margin),
        "gate_apply_threshold": float(args.gate_apply_threshold),
        "forced_action_hold_steps": int(args.forced_action_hold_steps),
        "failure_classes": list(args.failure_classes),
        "dataset": {**summary, "episodes": data["episodes"], "meta": meta},
        "base_eval": base_eval,
        "gate_eval": gate_eval,
        "eval_seeds": seeds,
        "wall_clock_seconds": time.perf_counter() - started,
        "mechanisms": {
            "minGRU_terminal": True,
            "counterfactual_action_gate": bool(gate_path),
            "edge_plasticity": False,
            "bandit_persistence": False,
            "slow_consolidation": False,
            "gain_mutation": False,
        },
    }
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "report.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a recurrent-prefix counterfactual action gate for a minGRU terminal.")
    parser.add_argument("--checkpoint-path", required=True)
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
    parser.add_argument("--observation-mode", choices=POLICY_OBSERVATION_MODES, default="normalized_raw4_subchains_prev_force")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-motif-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motif-model-path", default="")
    parser.add_argument("--motif-score-scale", type=float, default=10.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-logit-margin-floor", type=float, default=0.0)
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics", "subchain_diagnostics"], default="subchain_diagnostics")
    parser.add_argument("--failure-classes", nargs="*", default=["pole_1_angle", "pole_2_angle"])
    parser.add_argument("--collect-seed-start", type=int, default=9120000)
    parser.add_argument("--collect-seeds-path", default="")
    parser.add_argument("--collect-episodes", type=int, default=40)
    parser.add_argument("--failure-offsets", type=int, nargs="*", default=[0, 5, 10, 20, 40, 80])
    parser.add_argument("--max-failure-states", type=int, default=6)
    parser.add_argument("--success-negative-stride", type=int, default=80)
    parser.add_argument("--max-success-states", type=int, default=4)
    parser.add_argument("--probe-horizon", type=int, default=80)
    parser.add_argument("--forced-action-hold-steps", type=int, default=1)
    parser.add_argument("--min-score-gap", type=float, default=0.10)
    parser.add_argument("--apply-min-score-gap", type=float, default=None)
    parser.add_argument("--apply-min-survival-gain", type=int, default=0)
    parser.add_argument("--apply-min-margin-gain", type=float, default=0.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--counterfactual-no-noise", action="store_true")
    parser.add_argument("--hidden-size-gate", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--no-override-weight", type=float, default=1.0)
    parser.add_argument("--positive-oversample-factor", type=float, default=1.0)
    parser.add_argument("--train-apply-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-epochs", type=int, default=120)
    parser.add_argument("--apply-positive-weight", type=float, default=0.0)
    parser.add_argument("--max-apply-positive-weight", type=float, default=12.0)
    parser.add_argument("--gate-confidence", type=float, default=0.75)
    parser.add_argument("--gate-margin", type=float, default=0.0)
    parser.add_argument("--gate-apply-threshold", type=float, default=0.5)
    parser.add_argument("--eval-seed-start", type=int, default=1900000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out", default="reports/mingru_action_gate")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "out": result["out"],
                "gate_path": result["gate_path"],
                "base_success": result["base_eval"]["success_rate"],
                "gate_success": result["gate_eval"]["success_rate"],
                "overrides": result["gate_eval"]["override_count"],
                "positives": result["dataset"]["positive_count"],
                "apply_positives": result["dataset"].get("apply_positive_count", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
