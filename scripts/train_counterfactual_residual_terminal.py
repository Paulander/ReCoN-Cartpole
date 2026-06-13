from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.policy_observation import adjacent_subchain_features, policy_observation_from_state
from recon_cartpole.control.residual_features import residual_aux_features
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout
from recon_cartpole.control.policy_observation import POLICY_OBSERVATION_MODES


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
            residual_policy_terminal_apply_threshold=float(getattr(args, "residual_apply_threshold", 0.5)),
            residual_policy_terminal_feature_mode=args.residual_feature_mode,
            residual_policy_terminal_hold_steps=getattr(args, "option_hold_steps", 1),
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


def recovery_pressure(raw_state: Any, n_poles: int) -> float:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n = int(n_poles)
    if raw.size < 2 + 2 * n:
        return 0.0
    cart = abs(float(raw[0])) / 2.4
    angle = float(np.max(np.abs(raw[2 : 2 + n]))) / 0.20943951023931953
    velocity = float(np.max(np.abs(raw[2 + n : 2 + 2 * n]))) / 5.0
    return float(np.clip(0.35 * cart + 0.45 * angle + 0.20 * velocity, 0.0, 2.0))


def load_motif_model(args: argparse.Namespace) -> dict[str, Any] | None:
    path = str(getattr(args, "motif_model_path", "") or "").strip()
    if not path:
        return None
    cached = getattr(args, "_motif_model_cache", None)
    if cached is not None:
        return cached
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    args._motif_model_cache = model
    return model


def motif_feature_vector(args: argparse.Namespace, raw_state: Any, base_force: float, model: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n = int(args.n_poles)
    needed = 2 + 2 * n
    if raw.size < needed:
        raise ValueError("raw_state is too short for motif features")
    theta_threshold = float(getattr(args, "theta_threshold", 12.0 * 2.0 * np.pi / 360.0))
    theta = raw[2 : 2 + n] / max(theta_threshold, 1e-9)
    theta_dot = raw[2 + n : 2 + 2 * n] / max(float(getattr(args, "pole_velocity_scale", 5.0)), 1e-9)
    cart = [
        float(raw[0]) / max(float(getattr(args, "x_threshold", 2.4)), 1e-9),
        float(raw[1]) / max(float(getattr(args, "cart_velocity_scale", 5.0)), 1e-9),
    ]
    values = cart + adjacent_subchain_features(theta, theta_dot, max(4, n))
    target_dim = len(model.get("scale", values))
    if target_dim > len(values):
        diagnostics = [
            float(base_force) / max(float(args.force_mag), 1e-9),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        values.extend(diagnostics[: target_dim - len(values)])
    if len(values) < target_dim:
        values.extend([0.0] * (target_dim - len(values)))
    return np.asarray(values[:target_dim], dtype=np.float32)


def motif_score_for_state(args: argparse.Namespace, model: dict[str, Any], state: dict[str, Any]) -> float:
    vector = motif_feature_vector(args, state["raw_before"], float(state.get("force", 0.0)), model)
    pos = np.asarray(model["positive_mean"], dtype=np.float32)
    neg = np.asarray(model["negative_mean"], dtype=np.float32)
    scale = np.maximum(np.asarray(model["scale"], dtype=np.float32), 1e-6)
    z = vector / scale
    p = pos / scale
    n = neg / scale
    d_pos = float(np.mean((z - p) ** 2))
    d_neg = float(np.mean((z - n) ** 2))
    return d_neg - d_pos


def rank_failure_candidates(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = load_motif_model(args)
    if model is None or not candidates:
        return candidates
    scored: list[dict[str, Any]] = []
    for item in candidates:
        candidate = dict(item)
        try:
            candidate["motif_score"] = motif_score_for_state(args, model, candidate)
        except ValueError:
            candidate["motif_score"] = float("-inf")
        scored.append(candidate)
    min_score = float(getattr(args, "motif_score_min", float("-inf")))
    scored = [item for item in scored if float(item.get("motif_score", float("-inf"))) >= min_score]
    motif_weight = float(getattr(args, "motif_rank_weight", 1.0))
    pressure_weight = float(getattr(args, "pressure_rank_weight", 0.0))
    for item in scored:
        item["candidate_rank"] = (
            motif_weight * float(item.get("motif_score", float("-inf")))
            + pressure_weight * float(item.get("recovery_pressure", 0.0))
        )
    top_k = int(getattr(args, "motif_top_k", 0))
    scored.sort(
        key=lambda item: (
            float(item.get("candidate_rank", float("-inf"))),
            float(item.get("motif_score", float("-inf"))),
            float(item.get("recovery_pressure", 0.0)),
            -abs(int(item.get("failure_offset", 0)) - int(getattr(args, "failure_window_target_offset", 40))),
        ),
        reverse=True,
    )
    if top_k > 0:
        scored = scored[:top_k]
    return scored


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


def residual_center_class(args: argparse.Namespace) -> int:
    return int(args.residual_action_bins) // 2


def residual_shift_for_class(args: argparse.Namespace, residual_class: int) -> int:
    return int(residual_class) - residual_center_class(args)


def candidate_residual_sequences(args: argparse.Namespace, residual_class: int) -> list[tuple[list[int], list[int]]]:
    first_class = int(residual_class)
    shift = residual_shift_for_class(args, first_class)
    first_steps = max(1, int(getattr(args, "option_hold_steps", 1))) if shift != 0 else 1
    tail_steps = max(0, int(getattr(args, "option_tail_steps", 0)))
    if tail_steps <= 0 or shift == 0:
        return [([first_class], [first_steps])]
    return [
        ([first_class, int(tail_class)], [first_steps, tail_steps])
        for tail_class in range(int(args.residual_action_bins))
    ]


def _score_from_rollout(
    args: argparse.Namespace,
    residual_class: int,
    sequence: list[int],
    sequence_steps: list[int],
    survived: int,
    final_raw: Any,
    pressures: list[float],
) -> dict[str, Any]:
    initial_pressure = float(pressures[0]) if pressures else 0.0
    margin = stability_margin(final_raw, args)
    pressure_mean = float(np.mean(pressures)) if pressures else 0.0
    pressure_max = float(np.max(pressures)) if pressures else 0.0
    pressure_final = float(pressures[-1]) if pressures else 0.0
    pressure_drop = float(initial_pressure - pressure_final)
    first_shift = residual_shift_for_class(args, int(residual_class))
    tail_class = int(sequence[1]) if len(sequence) > 1 else residual_center_class(args)
    tail_shift = residual_shift_for_class(args, tail_class)
    tail_penalty = float(getattr(args, "tail_shift_penalty", 0.0)) * abs(tail_shift)
    score = (
        float(survived)
        + float(args.margin_weight) * margin
        + float(getattr(args, "pressure_drop_weight", 0.0)) * pressure_drop
        - float(getattr(args, "pressure_mean_weight", 0.0)) * pressure_mean
        - float(getattr(args, "pressure_max_weight", 0.0)) * pressure_max
        - float(getattr(args, "pressure_final_weight", 0.0)) * pressure_final
        - float(args.shift_penalty) * abs(first_shift)
        - tail_penalty
    )
    return {
        "class": int(residual_class),
        "shift": int(first_shift),
        "tail_class": tail_class,
        "tail_shift": int(tail_shift),
        "sequence": [int(item) for item in sequence],
        "sequence_steps": [int(item) for item in sequence_steps],
        "forced_steps": int(sum(sequence_steps)),
        "survived": int(survived),
        "margin": float(margin),
        "pressure_initial": float(initial_pressure),
        "pressure_mean": pressure_mean,
        "pressure_max": pressure_max,
        "pressure_final": pressure_final,
        "pressure_drop": pressure_drop,
        "score": score,
    }


def simulate_residual_sequence(
    args: argparse.Namespace,
    raw_state: list[float],
    step: int,
    residual_class: int,
    sequence: list[int],
    sequence_steps: list[int],
) -> dict[str, Any]:
    env = make_env(args, force_noise=0.0 if args.counterfactual_no_noise else args.force_noise)
    controller = make_controller(args)
    set_env_state(env, raw_state, step)
    obs = env._get_obs()
    controller.start_episode()
    survived = 0
    final_raw = np.asarray(raw_state, dtype=float)
    pressures: list[float] = [recovery_pressure(final_raw, int(args.n_poles))]
    terminated = False
    truncated = False
    info: dict[str, Any] = {"raw_state": raw_state}
    first_action: int | None = None
    for class_idx, hold_steps in zip(sequence, sequence_steps):
        shift = residual_shift_for_class(args, int(class_idx))
        for _ in range(max(1, int(hold_steps))):
            raw = np.asarray(info["raw_state"], dtype=float).copy()
            base_action, _diagnostics = controller.act(obs, raw)
            action = int(np.clip(int(base_action) + shift, 0, int(args.discrete_action_bins) - 1))
            if first_action is None:
                first_action = action
            obs, _reward, terminated, truncated, info = env.step(action)
            survived += 1
            final_raw = np.asarray(info.get("raw_state", []), dtype=float)
            pressures.append(recovery_pressure(final_raw, int(args.n_poles)))
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    if not (terminated or truncated):
        for _ in range(max(1, int(args.probe_horizon)) - survived):
            raw = np.asarray(info["raw_state"], dtype=float).copy()
            action, _diagnostics = controller.act(obs, raw)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            survived += 1
            final_raw = np.asarray(info.get("raw_state", []), dtype=float)
            pressures.append(recovery_pressure(final_raw, int(args.n_poles)))
            if terminated or truncated:
                break
    result = _score_from_rollout(args, residual_class, sequence, sequence_steps, survived, final_raw, pressures)
    result["first_action"] = int(first_action if first_action is not None else 0)
    return result


def counterfactual_score(args: argparse.Namespace, raw_state: list[float], step: int, base_force: float, residual_class: int) -> dict[str, Any]:
    del base_force  # The simulator asks the frozen controller for the base action at each forced tick.
    options = [
        simulate_residual_sequence(args, raw_state, step, residual_class, sequence, sequence_steps)
        for sequence, sequence_steps in candidate_residual_sequences(args, int(residual_class))
    ]
    return max(options, key=lambda item: float(item["score"]))


def _gate_threshold(args: argparse.Namespace, name: str, fallback: float | int) -> float:
    value = getattr(args, name, None)
    return float(fallback if value is None else value)


def label_state(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    options = [counterfactual_score(args, state["raw_before"], int(state["step"]), float(state["force"]), cls) for cls in range(int(args.residual_action_bins))]
    center = int(args.residual_action_bins) // 2
    center_option = options[center]
    center_score = float(center_option["score"])
    center_survived = int(center_option["survived"])
    center_margin = float(center_option["margin"])
    center_pressure_final = float(center_option.get("pressure_final", 0.0))
    best_option = max(options, key=lambda item: float(item["score"]))
    best_score = float(best_option["score"])
    best_classes = [
        idx for idx, item in enumerate(options)
        if abs(float(item["score"]) - best_score) <= float(args.score_tolerance)
    ]
    label = center
    eligible: list[dict[str, Any]] = []
    for idx, item in enumerate(options):
        survival_gain = int(item["survived"]) - center_survived
        margin_gain = float(item["margin"]) - center_margin
        pressure_gain = center_pressure_final - float(item.get("pressure_final", center_pressure_final))
        score_gap = float(item["score"]) - center_score
        if idx == center:
            continue
        if score_gap < float(args.min_score_gap):
            continue
        if survival_gain < int(getattr(args, "min_survival_gain", 0)):
            continue
        if margin_gain < float(getattr(args, "min_margin_gain", 0.0)):
            continue
        if pressure_gain < float(getattr(args, "min_pressure_gain", -999.0)):
            continue
        eligible.append(item)
    if eligible and center not in best_classes:
        label = int(max(eligible, key=lambda item: float(item["score"]))["class"])
    feature = residual_observation(args, state["raw_before"], float(state["force"]), int(state["step"]))
    chosen = options[int(label)]
    chosen_survival_gain = int(chosen["survived"]) - center_survived
    chosen_margin_gain = float(chosen["margin"]) - center_margin
    chosen_pressure_gain = center_pressure_final - float(chosen.get("pressure_final", center_pressure_final))
    chosen_score_gap = float(chosen["score"]) - center_score
    apply_label = int(
        label != center
        and chosen_score_gap >= _gate_threshold(args, "apply_min_score_gap", float(args.min_score_gap))
        and chosen_survival_gain >= _gate_threshold(args, "apply_min_survival_gain", int(getattr(args, "min_survival_gain", 0)))
        and chosen_margin_gain >= _gate_threshold(args, "apply_min_margin_gain", float(getattr(args, "min_margin_gain", 0.0)))
        and chosen_pressure_gain >= _gate_threshold(args, "apply_min_pressure_gain", float(getattr(args, "min_pressure_gain", -999.0)))
    )
    return {
        "feature": feature.tolist(),
        "label": int(label),
        "apply_label": apply_label,
        "seed": int(state.get("seed", -1)),
        "step": int(state["step"]),
        "base_force": float(state["force"]),
        "options": options,
        "center_score": float(center_score),
        "best_score": float(best_score),
        "score_gap": float(best_score - center_score),
        "chosen_score": float(chosen["score"]),
        "chosen_score_gap": float(chosen_score_gap),
        "chosen_survival_gain": chosen_survival_gain,
        "chosen_margin_gain": chosen_margin_gain,
        "chosen_pressure_gain": chosen_pressure_gain,
        "best_survival_gain": int(best_option["survived"]) - center_survived,
        "best_margin_gain": float(best_option["margin"]) - center_margin,
        "best_pressure_gain": center_pressure_final - float(best_option.get("pressure_final", center_pressure_final)),
    }


def select_failure_states(args: argparse.Namespace, episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Pick candidate states from failed episodes for residual counterfactual probing."""
    states = episode["states"]
    if not states:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    if bool(getattr(args, "use_failure_window", False)):
        start = max(0, int(getattr(args, "failure_window_start", 0)))
        end = max(start, int(getattr(args, "failure_window_end", 120)))
        stride = max(1, int(getattr(args, "failure_window_stride", 5)))
        candidates: list[dict[str, Any]] = []
        for offset in range(start, end + 1, stride):
            pos = len(states) - 1 - int(offset)
            if 0 <= pos < len(states):
                state = states[pos]
                step = int(state["step"])
                if step in seen:
                    continue
                seen.add(step)
                candidate = dict(state)
                candidate["seed"] = episode["seed"]
                candidate["failure_offset"] = int(offset)
                candidate["recovery_pressure"] = recovery_pressure(candidate["raw_before"], int(args.n_poles))
                candidates.append(candidate)
        target = int(getattr(args, "failure_window_target_offset", 40))
        candidates = rank_failure_candidates(args, candidates)
        candidates.sort(
            key=lambda item: (
                float(item.get("candidate_rank", item.get("motif_score", 0.0))),
                float(item.get("motif_score", 0.0)),
                float(item.get("recovery_pressure", 0.0)),
                -abs(int(item.get("failure_offset", 0)) - target),
            ),
            reverse=True,
        )
        selected.extend(candidates[: int(getattr(args, "max_window_states", args.max_failure_states))])
    for offset in args.failure_offsets:
        pos = len(states) - 1 - int(offset)
        if 0 <= pos < len(states) and int(states[pos]["step"]) not in seen:
            seen.add(int(states[pos]["step"]))
            state = dict(states[pos])
            state["seed"] = episode["seed"]
            state["failure_offset"] = int(offset)
            state["recovery_pressure"] = recovery_pressure(state["raw_before"], int(args.n_poles))
            selected.append(state)
    cap = max(1, int(args.max_failure_states))
    if len(selected) > cap:
        target = int(getattr(args, "failure_window_target_offset", 40))
        selected = rank_failure_candidates(args, selected)
        selected.sort(
            key=lambda item: (
                float(item.get("candidate_rank", item.get("motif_score", 0.0))),
                float(item.get("motif_score", 0.0)),
                float(item.get("recovery_pressure", 0.0)),
                -abs(int(item.get("failure_offset", 0)) - target),
                int(item.get("step", 0)),
            ),
            reverse=True,
        )
        selected = selected[:cap]
    selected.sort(key=lambda item: int(item["step"]))
    return selected


def collect_seed_values(args: argparse.Namespace) -> list[int]:
    seed_list = str(getattr(args, "collect_seed_list", "") or "").strip()
    if not seed_list:
        starts = [int(item) for item in getattr(args, "collect_seed_starts", []) or []]
        if starts:
            return [starts[idx % len(starts)] + idx // len(starts) for idx in range(int(args.collect_episodes))]
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


def select_success_negative_states(args: argparse.Namespace, episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Pick preservation negatives from solved episodes, including high-risk states."""
    states = episode["states"]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    if int(args.success_negative_stride) > 0:
        for state in states[:: int(args.success_negative_stride)][-int(args.max_success_states) :]:
            item = dict(state)
            item["seed"] = episode["seed"]
            item["success_negative"] = True
            item["success_negative_kind"] = "stride"
            item["recovery_pressure"] = recovery_pressure(item["raw_before"], int(args.n_poles))
            selected.append(item)
            seen.add(int(item["step"]))
    risk_count = int(getattr(args, "success_risk_negative_count", 0) or 0)
    if risk_count <= 0:
        selected.sort(key=lambda item: int(item["step"]))
        return selected
    start = max(0, int(getattr(args, "success_risk_window_start", 0)))
    end = int(getattr(args, "success_risk_window_end", 0) or int(args.horizon))
    end = min(max(start, end), len(states) - 1)
    stride = max(1, int(getattr(args, "success_risk_stride", 5)))
    candidates: list[dict[str, Any]] = []
    for pos in range(start, end + 1, stride):
        state = states[pos]
        step = int(state["step"])
        if step in seen:
            continue
        item = dict(state)
        item["seed"] = episode["seed"]
        item["success_negative"] = True
        item["success_negative_kind"] = "risk"
        item["recovery_pressure"] = recovery_pressure(item["raw_before"], int(args.n_poles))
        candidates.append(item)
    candidates.sort(key=lambda item: (float(item.get("recovery_pressure", 0.0)), int(item["step"])), reverse=True)
    selected.extend(candidates[:risk_count])
    selected.sort(key=lambda item: int(item["step"]))
    return selected


def collect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in collect_seed_values(args):
        episode = collect_episode_states(args, int(seed))
        episodes.append({key: episode[key] for key in ("seed", "steps", "return", "success")})
        if episode["success"]:
            selected = select_success_negative_states(args, episode)
            for state in selected:
                feature = residual_observation(args, state["raw_before"], float(state["force"]), int(state["step"]))
                rows.append(
                    {
                        "feature": feature.tolist(),
                        "label": int(args.residual_action_bins) // 2,
                        "apply_label": 0,
                        "seed": episode["seed"],
                        "step": int(state["step"]),
                        "success_negative": True,
                        "success_negative_kind": state.get("success_negative_kind", "stride"),
                        "recovery_pressure": float(state.get("recovery_pressure", 0.0)),
                    }
                )
        else:
            selected = select_failure_states(args, episode)
            for state in selected:
                rows.append(label_state(args, state))
        partial = {"episodes": episodes, "rows": len(rows), "positive_rows": sum(1 for row in rows if int(row["label"]) != int(args.residual_action_bins) // 2)}
        (out / "partial_dataset.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    return {"episodes": episodes, "rows": rows}


def train_model(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    if not rows:
        raise ValueError("no counterfactual residual rows collected")
    classes = int(args.residual_action_bins)
    center = classes // 2
    factor = max(1, int(round(float(getattr(args, "non_noop_oversample_factor", 1.0)))))
    train_rows = list(rows)
    if factor > 1:
        non_noop_rows = [row for row in rows if int(row["label"]) != center]
        for _ in range(factor - 1):
            train_rows.extend(non_noop_rows)
    x = torch.tensor([row["feature"] for row in train_rows], dtype=torch.float32)
    y = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    apply_y = torch.tensor(
        [int(row.get("apply_label", int(int(row["label"]) != center))) for row in train_rows],
        dtype=torch.float32,
    ).reshape(-1, 1)
    model = nn.Sequential(nn.Linear(x.shape[1], int(args.hidden_size)), nn.ReLU(), nn.Linear(int(args.hidden_size), classes))
    counts = torch.bincount(y, minlength=classes).float()
    weights = torch.clamp(counts.sum() / torch.clamp(counts, min=1.0), max=float(args.max_class_weight))
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

    apply_model = None
    apply_accuracy = 0.0
    apply_positive_rate = float(apply_y.mean().item()) if apply_y.numel() else 0.0
    if bool(getattr(args, "train_apply_gate", True)):
        apply_model = nn.Sequential(nn.Linear(x.shape[1], int(args.hidden_size)), nn.ReLU(), nn.Linear(int(args.hidden_size), 1))
        positives = torch.clamp(apply_y.sum(), min=0.0)
        negatives = torch.clamp(torch.tensor(float(apply_y.numel())) - positives, min=0.0)
        pos_weight_value = float(getattr(args, "apply_positive_weight", 0.0))
        if pos_weight_value <= 0.0:
            pos_weight_value = float((negatives / torch.clamp(positives, min=1.0)).item())
        pos_weight = torch.tensor([max(1.0, min(float(getattr(args, "max_apply_positive_weight", 12.0)), pos_weight_value))], dtype=torch.float32)
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
            apply_pred = (torch.sigmoid(apply_model(x)) >= float(getattr(args, "residual_apply_threshold", 0.5))).float()
            apply_accuracy = float((apply_pred == apply_y).float().mean().item())

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
        "apply_label_counts": {"0": int((apply_y == 0).sum().item()), "1": int((apply_y == 1).sum().item())},
        "original_row_count": int(len(rows)),
        "expanded_row_count": int(len(train_rows)),
        "non_noop_oversample_factor": int(factor),
        "train_accuracy": acc,
        "non_noop_recall": non_noop_recall,
        "apply_gate_enabled": apply_model is not None,
        "apply_positive_rate": apply_positive_rate,
        "apply_accuracy": apply_accuracy,
        "apply_threshold": float(getattr(args, "residual_apply_threshold", 0.5)),
        "format": "counterfactual_gated_residual_terminal_v2" if apply_model is not None else "counterfactual_residual_terminal_v1",
    }
    return (model, apply_model) if apply_model is not None else model, meta


def save_model(model: Any, meta: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, tuple):
        action_model, apply_model = model
        payload = {"state_dict": action_model.state_dict(), "meta": meta}
        if apply_model is not None:
            payload["apply_state_dict"] = apply_model.state_dict()
    else:
        payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, path)


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
    chosen_survival_gains: list[float] = []
    chosen_margin_gains: list[float] = []
    best_survival_gains: list[float] = []
    best_margin_gains: list[float] = []
    chosen_pressure_gains: list[float] = []
    best_pressure_gains: list[float] = []
    success_negative_count = 0
    risk_success_negative_count = 0
    for row in rows:
        label = int(row.get("label", 0))
        counts[str(label)] = counts.get(str(label), 0) + 1
        gaps.append(float(row.get("score_gap", 0.0)))
        chosen_survival_gains.append(float(row.get("chosen_survival_gain", 0.0)))
        chosen_margin_gains.append(float(row.get("chosen_margin_gain", 0.0)))
        best_survival_gains.append(float(row.get("best_survival_gain", 0.0)))
        best_margin_gains.append(float(row.get("best_margin_gain", 0.0)))
        chosen_pressure_gains.append(float(row.get("chosen_pressure_gain", 0.0)))
        best_pressure_gains.append(float(row.get("best_pressure_gain", 0.0)))
        if bool(row.get("success_negative", False)):
            success_negative_count += 1
            if row.get("success_negative_kind") == "risk":
                risk_success_negative_count += 1
    center = classes // 2
    return {
        "row_count": len(rows),
        "label_counts": counts,
        "non_noop_count": sum(v for k, v in counts.items() if int(k) != center),
        "max_score_gap": max(gaps) if gaps else 0.0,
        "mean_score_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_chosen_survival_gain": float(np.mean(chosen_survival_gains)) if chosen_survival_gains else 0.0,
        "mean_chosen_margin_gain": float(np.mean(chosen_margin_gains)) if chosen_margin_gains else 0.0,
        "max_best_survival_gain": max(best_survival_gains) if best_survival_gains else 0.0,
        "max_best_margin_gain": max(best_margin_gains) if best_margin_gains else 0.0,
        "mean_chosen_pressure_gain": float(np.mean(chosen_pressure_gains)) if chosen_pressure_gains else 0.0,
        "max_best_pressure_gain": max(best_pressure_gains) if best_pressure_gains else 0.0,
        "success_negative_count": success_negative_count,
        "risk_success_negative_count": risk_success_negative_count,
    }


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
        "option_hold_steps": int(getattr(args, "option_hold_steps", 1)),
        "failure_state_selection": {
            "collect_seed_starts": [int(item) for item in getattr(args, "collect_seed_starts", []) or []],
            "use_failure_window": bool(getattr(args, "use_failure_window", False)),
            "failure_window_start": int(getattr(args, "failure_window_start", 0)),
            "failure_window_end": int(getattr(args, "failure_window_end", 0)),
            "failure_window_stride": int(getattr(args, "failure_window_stride", 0)),
            "failure_window_target_offset": int(getattr(args, "failure_window_target_offset", 0)),
            "max_window_states": int(getattr(args, "max_window_states", 0)),
            "failure_offsets": [int(item) for item in args.failure_offsets],
            "motif_model_path": str(getattr(args, "motif_model_path", "") or ""),
            "motif_score_min": float(getattr(args, "motif_score_min", float("-inf"))),
            "motif_top_k": int(getattr(args, "motif_top_k", 0)),
            "motif_rank_weight": float(getattr(args, "motif_rank_weight", 1.0)),
            "pressure_rank_weight": float(getattr(args, "pressure_rank_weight", 0.0)),
            "success_risk_negative_count": int(getattr(args, "success_risk_negative_count", 0)),
            "success_risk_window_start": int(getattr(args, "success_risk_window_start", 0)),
            "success_risk_window_end": int(getattr(args, "success_risk_window_end", 0) or int(args.horizon)),
            "success_risk_stride": int(getattr(args, "success_risk_stride", 5)),
        },
        "label_gates": {
            "min_score_gap": float(args.min_score_gap),
            "min_survival_gain": int(getattr(args, "min_survival_gain", 0)),
            "min_margin_gain": float(getattr(args, "min_margin_gain", 0.0)),
            "min_pressure_gain": float(getattr(args, "min_pressure_gain", -999.0)),
            "apply_min_score_gap": getattr(args, "apply_min_score_gap", None),
            "apply_min_survival_gain": getattr(args, "apply_min_survival_gain", None),
            "apply_min_margin_gain": getattr(args, "apply_min_margin_gain", None),
            "apply_min_pressure_gain": getattr(args, "apply_min_pressure_gain", None),
            "score_tolerance": float(args.score_tolerance),
            "pressure_drop_weight": float(getattr(args, "pressure_drop_weight", 0.0)),
            "pressure_mean_weight": float(getattr(args, "pressure_mean_weight", 0.0)),
            "pressure_max_weight": float(getattr(args, "pressure_max_weight", 0.0)),
            "pressure_final_weight": float(getattr(args, "pressure_final_weight", 0.0)),
        },
        "mechanisms": {"counterfactual_residual_terminal": True, "residual_apply_gate": bool(meta.get("apply_gate_enabled", False)), "residual_option_hold": int(getattr(args, "option_hold_steps", 1)) > 1, "recon_integration_eval": True, "gain_mutation": False},
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
        f"Success negatives: `{ds.get('success_negative_count', 0)}`, high-risk success negatives: `{ds.get('risk_success_negative_count', 0)}`",
        f"Apply gate: `{ds.get('meta', {}).get('apply_gate_enabled', False)}`, apply label counts: `{ds.get('meta', {}).get('apply_label_counts', {})}`",
        f"Option hold steps: `{result.get('option_hold_steps', 1)}`",
        f"Failure state selection: `{result.get('failure_state_selection', {})}`",
        f"Label gates: `{result.get('label_gates', {})}`",
        f"Mean chosen survival gain: `{ds.get('mean_chosen_survival_gain', 0.0):.3f}`; max best survival gain: `{ds.get('max_best_survival_gain', 0.0):.3f}`",
        f"Mean chosen pressure gain: `{ds.get('mean_chosen_pressure_gain', 0.0):.3f}`; max best pressure gain: `{ds.get('max_best_pressure_gain', 0.0):.3f}`",
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
    parser.add_argument("--base-observation-mode", choices=POLICY_OBSERVATION_MODES, default="normalized_raw")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics", "subchain_diagnostics"], default="proposal_diagnostics")
    parser.add_argument("--residual-action-bins", type=int, default=5)
    parser.add_argument("--residual-gate-threshold", type=float, default=0.60)
    parser.add_argument("--residual-apply-threshold", type=float, default=0.50)
    parser.add_argument("--option-hold-steps", type=int, default=1)
    parser.add_argument("--option-tail-steps", type=int, default=0)
    parser.add_argument("--tail-shift-penalty", type=float, default=0.0)
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
    parser.add_argument("--collect-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--collect-seed-list", default="")
    parser.add_argument("--collect-episodes", type=int, default=80)
    parser.add_argument("--failure-offsets", type=int, nargs="*", default=[0, 2, 5, 10, 20, 40])
    parser.add_argument("--max-failure-states", type=int, default=6)
    parser.add_argument("--use-failure-window", action="store_true")
    parser.add_argument("--failure-window-start", type=int, default=0)
    parser.add_argument("--failure-window-end", type=int, default=120)
    parser.add_argument("--failure-window-stride", type=int, default=5)
    parser.add_argument("--failure-window-target-offset", type=int, default=40)
    parser.add_argument("--max-window-states", type=int, default=18)
    parser.add_argument("--motif-model-path", default="")
    parser.add_argument("--motif-score-min", type=float, default=float("-inf"))
    parser.add_argument("--motif-top-k", type=int, default=0)
    parser.add_argument("--motif-rank-weight", type=float, default=1.0)
    parser.add_argument("--pressure-rank-weight", type=float, default=0.0)
    parser.add_argument("--x-threshold", type=float, default=2.4)
    parser.add_argument("--theta-threshold", type=float, default=12.0 * 2.0 * np.pi / 360.0)
    parser.add_argument("--cart-velocity-scale", type=float, default=5.0)
    parser.add_argument("--pole-velocity-scale", type=float, default=5.0)
    parser.add_argument("--success-negative-stride", type=int, default=80)
    parser.add_argument("--max-success-states", type=int, default=3)
    parser.add_argument("--success-risk-negative-count", type=int, default=0)
    parser.add_argument("--success-risk-window-start", type=int, default=0)
    parser.add_argument("--success-risk-window-end", type=int, default=0)
    parser.add_argument("--success-risk-stride", type=int, default=5)
    parser.add_argument("--probe-horizon", type=int, default=100)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--shift-penalty", type=float, default=0.02)
    parser.add_argument("--min-score-gap", type=float, default=0.03)
    parser.add_argument("--min-survival-gain", type=int, default=0)
    parser.add_argument("--min-margin-gain", type=float, default=0.0)
    parser.add_argument("--min-pressure-gain", type=float, default=-999.0)
    parser.add_argument("--apply-min-score-gap", type=float, default=None)
    parser.add_argument("--apply-min-survival-gain", type=float, default=None)
    parser.add_argument("--apply-min-margin-gain", type=float, default=None)
    parser.add_argument("--apply-min-pressure-gain", type=float, default=None)
    parser.add_argument("--pressure-drop-weight", type=float, default=0.0)
    parser.add_argument("--pressure-mean-weight", type=float, default=0.0)
    parser.add_argument("--pressure-max-weight", type=float, default=0.0)
    parser.add_argument("--pressure-final-weight", type=float, default=0.0)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--counterfactual-no-noise", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--noop-class-weight", type=float, default=1.0)
    parser.add_argument("--non-noop-oversample-factor", type=float, default=1.0)
    parser.add_argument("--train-apply-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-epochs", type=int, default=120)
    parser.add_argument("--apply-positive-weight", type=float, default=0.0)
    parser.add_argument("--max-apply-positive-weight", type=float, default=12.0)
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
