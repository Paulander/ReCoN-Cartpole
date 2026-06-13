from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.control.policy_observation import policy_observation_from_state
from recon_cartpole.control.sensors import features_from_state
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.subchain_terminal import (
    SharedSubchainTerminal,
    SubchainTerminalConfig,
    save_subchain_terminal_checkpoint,
)
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


def set_env_state(env: CartPoleNEnv, raw_state: Any, step: int) -> None:
    env.state = np.asarray(raw_state, dtype=float).copy()
    env.steps = int(step)


def force_values(args: argparse.Namespace) -> np.ndarray:
    return np.linspace(-float(args.force_mag), float(args.force_mag), int(args.discrete_action_bins))


def force_to_index(force: float, args: argparse.Namespace) -> int:
    values = force_values(args)
    return int(np.argmin(np.abs(values - float(force))))


def candidate_force_sequences(args: argparse.Namespace, first_force: float) -> list[dict[str, Any]]:
    first_steps = max(1, int(args.option_hold_steps))
    tail_steps = max(0, int(getattr(args, "option_tail_steps", 0)))
    if tail_steps <= 0:
        return [{"forces": [float(first_force)], "steps": [first_steps]}]
    return [
        {
            "forces": [float(first_force), float(tail_force)],
            "steps": [first_steps, tail_steps],
        }
        for tail_force in force_values(args)
    ]


def baseline_force_sequence(args: argparse.Namespace, base_force: float) -> dict[str, Any]:
    first_steps = max(1, int(args.option_hold_steps))
    tail_steps = max(0, int(getattr(args, "option_tail_steps", 0)))
    forces = [float(base_force)]
    steps = [first_steps]
    if tail_steps > 0:
        forces.append(float(base_force))
        steps.append(tail_steps)
    return {"forces": forces, "steps": steps}


def pressure_from_raw(raw_state: Any, n_poles: int) -> float:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n = int(n_poles)
    if raw.size < 2 + 2 * n:
        return 0.0
    cart = abs(float(raw[0])) / 2.4
    angle = float(np.max(np.abs(raw[2 : 2 + n]))) / 0.20943951023931953
    velocity = float(np.max(np.abs(raw[2 + n : 2 + 2 * n]))) / 5.0
    return float(np.clip(0.35 * cart + 0.45 * angle + 0.20 * velocity, 0.0, 2.0))


def stability_margin(raw_state: Any, n_poles: int) -> float:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n = int(n_poles)
    if raw.size < 2 + 2 * n:
        return -10.0
    cart = abs(float(raw[0])) / 2.4
    angle = float(np.max(np.abs(raw[2 : 2 + n]))) / 0.20943951023931953
    velocity = float(np.mean(np.abs(raw[2 + n : 2 + 2 * n]))) / 5.0
    return float(1.0 - angle - 0.10 * cart - 0.03 * velocity)


def make_teacher(args: argparse.Namespace) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode=args.teacher_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.policy_terminal_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_terminal_observation_mode,
        )
    )


def rollout_episode(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    teacher = make_teacher(args)
    env = make_env(args)
    obs, info = env.reset(seed=int(seed))
    teacher.start_episode()
    states: list[dict[str, Any]] = []
    total = 0.0
    prev_force = 0.0
    for step in range(int(args.horizon)):
        raw_before = np.asarray(info["raw_state"], dtype=float).copy()
        action, diagnostics = teacher.act(obs, raw_before)
        base_force = float(diagnostics.get("force", force_values(args)[int(action)]))
        states.append(
            {
                "step": int(step),
                "raw_before": raw_before.tolist(),
                "action": int(action),
                "force": base_force,
                "prev_force": float(prev_force),
            }
        )
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += float(reward)
        prev_force = base_force
        if terminated or truncated:
            return {
                "seed": int(seed),
                "steps": int(step + 1),
                "return": total,
                "success": bool(truncated and step + 1 >= int(args.horizon)),
                "states": states,
            }
    return {"seed": int(seed), "steps": int(args.horizon), "return": total, "success": True, "states": states}


def counterfactual_sequence_score(
    args: argparse.Namespace,
    raw_state: Any,
    step: int,
    base_force: float,
    sequence: dict[str, Any],
) -> dict[str, Any]:
    env = make_env(args)
    controller = make_teacher(args)
    set_env_state(env, raw_state, step)
    obs = env._get_obs()
    controller.start_episode()
    survived = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {"raw_state": raw_state}
    pressures = [pressure_from_raw(raw_state, int(args.n_poles))]
    seq_forces = [float(force) for force in sequence.get("forces", [])]
    seq_steps = [max(1, int(steps)) for steps in sequence.get("steps", [])]
    forced_trace: list[dict[str, Any]] = []
    prev_force = float(base_force)
    for force, hold_steps in zip(seq_forces, seq_steps):
        force_idx = force_to_index(force, args)
        for _ in range(hold_steps):
            raw_before = np.asarray(info.get("raw_state", raw_state), dtype=float).copy()
            forced_trace.append(
                {
                    "raw_state": raw_before.tolist(),
                    "step": int(step + survived),
                    "force": float(force),
                    "prev_force": float(prev_force),
                }
            )
            obs, _reward, terminated, truncated, info = env.step(force_idx)
            prev_force = float(force)
            survived += 1
            pressures.append(pressure_from_raw(info.get("raw_state", raw_state), int(args.n_poles)))
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
            pressures.append(pressure_from_raw(info.get("raw_state", raw_state), int(args.n_poles)))
            if terminated or truncated:
                break
    final_raw = np.asarray(info.get("raw_state", raw_state), dtype=float)
    pressure_final = float(pressures[-1]) if pressures else 0.0
    pressure_drop = float(pressures[0] - pressure_final) if pressures else 0.0
    first_force = seq_forces[0] if seq_forces else float(base_force)
    tail_force = seq_forces[1] if len(seq_forces) > 1 else first_force
    first_shift = abs(force_to_index(first_force, args) - force_to_index(base_force, args))
    tail_shift = (
        abs(force_to_index(tail_force, args) - force_to_index(base_force, args))
        if len(seq_forces) > 1
        else 0
    )
    score = (
        float(survived)
        + float(args.margin_weight) * stability_margin(final_raw, int(args.n_poles))
        + float(args.pressure_drop_weight) * pressure_drop
        - float(args.pressure_final_weight) * pressure_final
        - float(args.shift_penalty) * float(first_shift)
        - float(getattr(args, "tail_shift_penalty", 0.0)) * float(tail_shift)
    )
    return {
        "force": float(first_force),
        "tail_force": float(tail_force),
        "forces": [float(force) for force in seq_forces],
        "force_steps": [int(steps) for steps in seq_steps],
        "forced_steps": int(sum(seq_steps)),
        "forced_trace": forced_trace,
        "survived": int(survived),
        "pressure_final": pressure_final,
        "pressure_drop": pressure_drop,
        "score": float(score),
    }


def counterfactual_score(
    args: argparse.Namespace, raw_state: Any, step: int, base_force: float, candidate_force: float
) -> dict[str, Any]:
    options = [
        counterfactual_sequence_score(args, raw_state, step, base_force, sequence)
        for sequence in candidate_force_sequences(args, candidate_force)
    ]
    return max(options, key=lambda item: float(item["score"]))


def selected_counterfactual_states(args: argparse.Namespace, episode: dict[str, Any]) -> list[dict[str, Any]]:
    states = list(episode.get("states", []))
    if not states:
        return []
    if bool(episode.get("success", False)):
        stride = int(getattr(args, "success_preserve_stride", 0))
        if stride <= 0:
            return []
        return [dict(state, preserve_success=True) for state in states[::stride]][-int(args.max_success_preserve_states) :]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for offset in [int(item) for item in args.failure_offsets]:
        pos = len(states) - 1 - offset
        if 0 <= pos < len(states):
            state = dict(states[pos])
            if int(state["step"]) in seen:
                continue
            seen.add(int(state["step"]))
            state["failure_offset"] = int(offset)
            state["preserve_success"] = False
            selected.append(state)
    if len(selected) > int(args.max_failure_states):
        selected = selected[-int(args.max_failure_states) :]
    return selected


def append_pair_rows(
    args: argparse.Namespace,
    terminal: SharedSubchainTerminal,
    rows: dict[str, list[Any]],
    raw_state: Any,
    target_force: float,
    confidence: float,
    weight: float,
    seed: int,
    step: int,
    source: str,
) -> None:
    features = features_from_state(raw_state, raw_state, args.n_poles)
    for pair in range(max(0, len(features.poles) - 1)):
        vec, pressure = terminal.pair_feature_vector(features, pair)
        rows["x"].append(vec)
        rows["force_targets"].append(float(target_force) / max(float(args.force_mag), 1e-9))
        rows["confidence_targets"].append(float(np.clip(confidence, 0.0, 1.0)))
        rows["sample_weights"].append(float(max(0.0, weight)))
        rows["pair_indices"].append(int(pair))
        rows["seeds"].append(int(seed))
        rows["steps"].append(int(step))
        rows["sources"].append(str(source))
        rows["pressures"].append(float(pressure))


def empty_policy_rows() -> dict[str, list[Any]]:
    return {
        key: []
        for key in [
            "observations",
            "prev_forces",
            "teacher_forces",
            "teacher_actions",
            "returns_to_go",
            "failure_within_k",
            "seeds",
            "sources",
            "rollout_sources",
            "rollout_forces",
            "rollout_actions",
            "motif_scores",
            "episodes",
            "step_indices",
            "sample_weights",
        ]
    }


def append_policy_option_row(
    args: argparse.Namespace,
    rows: dict[str, list[Any]],
    raw_state: Any,
    target_force: float,
    prev_force: float,
    confidence: float,
    weight: float,
    seed: int,
    episode: int,
    step: int,
    source: str,
) -> None:
    observation = policy_observation_from_state(
        raw_state,
        raw_state,
        int(args.n_poles),
        str(getattr(args, "option_policy_observation_mode", "normalized_raw4_prev_force")),
        previous_force=float(prev_force),
        force_mag=float(args.force_mag),
    )
    action = int(action_from_force(float(target_force), "discrete", float(args.force_mag), int(args.discrete_action_bins)))
    rows["observations"].append(observation.astype(np.float32, copy=False))
    rows["prev_forces"].append(float(prev_force))
    rows["teacher_forces"].append(float(target_force))
    rows["teacher_actions"].append(action)
    rows["returns_to_go"].append(float(max(0, int(args.horizon) - int(step))))
    rows["failure_within_k"].append(0.0 if str(source).startswith("counterfactual_recovery") else 1.0)
    rows["seeds"].append(int(seed))
    rows["sources"].append(str(source))
    rows["rollout_sources"].append("counterfactual_option")
    rows["rollout_forces"].append(float(target_force))
    rows["rollout_actions"].append(action)
    rows["motif_scores"].append(0.0)
    rows["episodes"].append(int(episode))
    rows["step_indices"].append(int(step))
    rows["sample_weights"].append(float(max(0.0, weight) * max(0.0, confidence)))


def append_policy_option_trace_rows(
    args: argparse.Namespace,
    rows: dict[str, list[Any]],
    option: dict[str, Any],
    confidence: float,
    weight: float,
    seed: int,
    episode: int,
) -> None:
    stride = max(1, int(getattr(args, "option_trace_stride", 1)))
    trace = list(option.get("forced_trace", []))
    if not bool(getattr(args, "append_option_trace", True)) or not trace:
        return
    max_rows = max(1, int(getattr(args, "max_option_trace_states", 12)))
    for item in trace[::stride][:max_rows]:
        append_policy_option_row(
            args,
            rows,
            item["raw_state"],
            float(item["force"]),
            float(item.get("prev_force", item["force"])),
            confidence=confidence,
            weight=weight,
            seed=seed,
            episode=episode,
            step=int(item["step"]),
            source="counterfactual_recovery_option",
        )


def finalize_policy_dataset(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    if not rows["observations"]:
        return {}
    return {
        "observations": np.stack(rows["observations"]).astype(np.float32),
        "prev_forces": np.asarray(rows["prev_forces"], dtype=np.float32),
        "teacher_forces": np.asarray(rows["teacher_forces"], dtype=np.float32),
        "teacher_actions": np.asarray(rows["teacher_actions"], dtype=np.int64),
        "returns_to_go": np.asarray(rows["returns_to_go"], dtype=np.float32),
        "failure_within_k": np.asarray(rows["failure_within_k"], dtype=np.float32),
        "seeds": np.asarray(rows["seeds"], dtype=np.int64),
        "sources": np.asarray(rows["sources"]),
        "rollout_sources": np.asarray(rows["rollout_sources"]),
        "rollout_forces": np.asarray(rows["rollout_forces"], dtype=np.float32),
        "rollout_actions": np.asarray(rows["rollout_actions"], dtype=np.int64),
        "motif_scores": np.asarray(rows["motif_scores"], dtype=np.float32),
        "episodes": np.asarray(rows["episodes"], dtype=np.int64),
        "step_indices": np.asarray(rows["step_indices"], dtype=np.int64),
        "sample_weights": np.asarray(rows["sample_weights"], dtype=np.float32),
    }


def append_option_trace_rows(
    args: argparse.Namespace,
    terminal: SharedSubchainTerminal,
    rows: dict[str, list[Any]],
    option: dict[str, Any],
    confidence: float,
    weight: float,
    seed: int,
) -> None:
    stride = max(1, int(getattr(args, "option_trace_stride", 1)))
    trace = list(option.get("forced_trace", []))
    if not bool(getattr(args, "append_option_trace", True)) or not trace:
        return
    max_rows = max(1, int(getattr(args, "max_option_trace_states", 12)))
    sampled = trace[::stride][:max_rows]
    for item in sampled:
        append_pair_rows(
            args,
            terminal,
            rows,
            item["raw_state"],
            float(item["force"]),
            confidence=confidence,
            weight=weight,
            seed=seed,
            step=int(item["step"]),
            source="counterfactual_recovery_option",
        )


def collect_teacher_dataset(args: argparse.Namespace, terminal: SharedSubchainTerminal) -> dict[str, np.ndarray]:
    rows: dict[str, list[Any]] = {key: [] for key in ["x", "force_targets", "confidence_targets", "sample_weights", "pair_indices", "seeds", "steps", "sources", "pressures"]}
    teacher = make_teacher(args)
    env = make_env(args)
    for ep in range(int(args.episodes)):
        seed = int(args.seed_start) + ep
        obs, info = env.reset(seed=seed)
        teacher.start_episode()
        for step in range(int(args.horizon)):
            raw = info.get("raw_state")
            action, diagnostics = teacher.act(obs, raw)
            force = float(diagnostics.get("force", 0.0))
            features = features_from_state(obs, raw, args.n_poles)
            for pair in range(max(0, len(features.poles) - 1)):
                vec, pressure = terminal.pair_feature_vector(features, pair)
                rows["x"].append(vec)
                rows["force_targets"].append(force / max(float(args.force_mag), 1e-9))
                rows["confidence_targets"].append(float(min(1.0, max(0.0, pressure))))
                rows["sample_weights"].append(1.0)
                rows["pair_indices"].append(pair)
                rows["seeds"].append(seed)
                rows["steps"].append(step)
                rows["sources"].append("teacher")
                rows["pressures"].append(float(pressure))
            obs, _reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
    return finalize_dataset(rows)


def collect_counterfactual_dataset(args: argparse.Namespace, terminal: SharedSubchainTerminal) -> dict[str, np.ndarray]:
    rows: dict[str, list[Any]] = {key: [] for key in ["x", "force_targets", "confidence_targets", "sample_weights", "pair_indices", "seeds", "steps", "sources", "pressures"]}
    policy_rows = empty_policy_rows()
    option_episode = 0
    episodes: list[dict[str, Any]] = []
    candidate_forces = force_values(args)
    for ep in range(int(args.episodes)):
        seed = int(args.seed_start) + ep
        episode = rollout_episode(args, seed)
        episodes.append({key: episode[key] for key in ("seed", "steps", "return", "success")})
        for state in selected_counterfactual_states(args, episode):
            raw = state["raw_before"]
            base_force = float(state["force"])
            prev_force = float(state.get("prev_force", base_force))
            if bool(state.get("preserve_success", False)):
                append_pair_rows(
                    args,
                    terminal,
                    rows,
                    raw,
                    base_force,
                    confidence=float(args.preserve_confidence),
                    weight=float(args.preserve_weight),
                    seed=seed,
                    step=int(state["step"]),
                    source="preserve_success",
                )
                append_policy_option_row(
                    args,
                    policy_rows,
                    raw,
                    base_force,
                    prev_force,
                    confidence=float(args.preserve_confidence),
                    weight=float(args.preserve_weight),
                    seed=seed,
                    episode=option_episode,
                    step=int(state["step"]),
                    source="preserve_success",
                )
                option_episode += 1
                continue
            options = [counterfactual_score(args, raw, int(state["step"]), base_force, force) for force in candidate_forces]
            center = counterfactual_sequence_score(
                args, raw, int(state["step"]), base_force, baseline_force_sequence(args, base_force)
            )
            best = max(options, key=lambda item: float(item["score"]))
            score_gap = float(best["score"] - center["score"])
            if score_gap >= float(args.min_score_gap):
                target_force = float(best["force"])
                confidence = min(1.0, max(0.0, score_gap / max(float(args.confidence_score_scale), 1e-9)))
                weight = float(args.recovery_weight) * max(0.1, confidence)
                source = "counterfactual_recovery"
            else:
                target_force = base_force
                confidence = float(args.preserve_confidence)
                weight = float(args.weak_preserve_weight)
                source = "counterfactual_no_better"
            append_pair_rows(
                args,
                terminal,
                rows,
                raw,
                target_force,
                confidence=confidence,
                weight=weight,
                seed=seed,
                step=int(state["step"]),
                source=source,
            )
            append_policy_option_row(
                args,
                policy_rows,
                raw,
                target_force,
                prev_force,
                confidence=confidence,
                weight=weight,
                seed=seed,
                episode=option_episode,
                step=int(state["step"]),
                source=source,
            )
            if source == "counterfactual_recovery":
                append_option_trace_rows(args, terminal, rows, best, confidence, weight, seed)
                append_policy_option_trace_rows(args, policy_rows, best, confidence, weight, seed, option_episode)
            option_episode += 1
    data = finalize_dataset(rows)
    policy_data = finalize_policy_dataset(policy_rows)
    if policy_data:
        data["option_policy_dataset"] = policy_data
    data["episode_summaries"] = np.asarray([json.dumps(item) for item in episodes])
    return data


def finalize_dataset(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    if not rows["x"]:
        raise ValueError("no subchain pair samples collected")
    return {
        "x": np.stack(rows["x"]).astype(np.float32),
        "force_targets": np.asarray(rows["force_targets"], dtype=np.float32),
        "confidence_targets": np.asarray(rows["confidence_targets"], dtype=np.float32),
        "sample_weights": np.asarray(rows["sample_weights"], dtype=np.float32),
        "pair_indices": np.asarray(rows["pair_indices"], dtype=np.int64),
        "seeds": np.asarray(rows["seeds"], dtype=np.int64),
        "steps": np.asarray(rows["steps"], dtype=np.int64),
        "sources": np.asarray(rows["sources"]),
        "pressures": np.asarray(rows["pressures"], dtype=np.float32),
    }


def collect_dataset(args: argparse.Namespace, terminal: SharedSubchainTerminal) -> dict[str, np.ndarray]:
    if str(args.label_mode) == "teacher_force":
        return collect_teacher_dataset(args, terminal)
    if str(args.label_mode) == "counterfactual_recovery":
        return collect_counterfactual_dataset(args, terminal)
    raise ValueError(f"unsupported label mode: {args.label_mode}")

def train_model(args: argparse.Namespace, data: dict[str, np.ndarray], terminal: SharedSubchainTerminal):
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install RL extras with torch to train the subchain pair terminal") from exc
    torch.manual_seed(int(args.train_seed))
    rng = np.random.default_rng(int(args.train_seed))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = terminal.build_model(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
    force_target = torch.as_tensor(data["force_targets"], dtype=torch.float32, device=device)
    confidence_target = torch.as_tensor(data["confidence_targets"], dtype=torch.float32, device=device)
    sample_weight = torch.as_tensor(data.get("sample_weights", np.ones(data["force_targets"].shape[0], dtype=np.float32)), dtype=torch.float32, device=device)
    sample_weight = sample_weight / torch.clamp(sample_weight.mean(), min=1e-6)
    indices = np.arange(x.shape[0])
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        rng.shuffle(indices)
        rows: list[dict[str, float]] = []
        for start in range(0, len(indices), int(args.batch_size)):
            idx = indices[start : start + int(args.batch_size)]
            if idx.size == 0:
                continue
            xb = x[idx]
            out = model(xb)
            pred_force = torch.tanh(out[:, 0])
            pred_conf = torch.sigmoid(out[:, 1])
            weight = sample_weight[idx]
            force_loss = torch.mean(F.mse_loss(pred_force, force_target[idx], reduction="none") * weight)
            confidence_loss = torch.mean(F.binary_cross_entropy(pred_conf, confidence_target[idx], reduction="none") * weight)
            loss = force_loss + float(args.confidence_loss_weight) * confidence_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "force_loss": float(force_loss.detach().cpu()),
                    "confidence_loss": float(confidence_loss.detach().cpu()),
                }
            )
        history.append({key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {})
    return model, history


def eval_controller(args: argparse.Namespace, checkpoint_path: str, mode: str) -> dict[str, Any]:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode=mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.policy_terminal_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_terminal_observation_mode,
            learned_subchain_terminal=SubchainTerminalConfig(
                enabled=bool(checkpoint_path),
                checkpoint_path=checkpoint_path,
                blend=args.subchain_blend,
                min_confidence=args.min_confidence,
                min_pair_pressure=args.min_pair_pressure,
                max_force_fraction=args.max_force_fraction,
                confidence_boost=args.confidence_boost,
                urgency_boost=args.urgency_boost,
            ),
        )
    )
    steps: list[float] = []
    for seed in eval_seeds(args):
        result = rollout(make_env(args), controller, seed=int(seed), horizon=args.horizon, trace=False)
        steps.append(float(result["steps"]))
    summary = summarize_steps(steps, args.horizon)
    summary["episodes"] = len(steps)
    return summary


def eval_seeds(args: argparse.Namespace) -> list[int]:
    starts = args.eval_seed_starts or [args.eval_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.eval_episodes)))
    return seeds


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SubchainTerminalConfig(
        hidden_size=args.hidden_size,
        include_pair_position=not args.no_pair_position,
        theta_scale=args.theta_scale,
        velocity_scale=args.velocity_scale,
        x_scale=args.x_scale,
        force_mag=args.force_mag,
    )
    terminal = SharedSubchainTerminal(args.n_poles, args.force_mag, cfg)
    data = collect_dataset(args, terminal)
    option_policy_data = data.pop("option_policy_dataset", {})
    np.savez_compressed(out / "dataset.npz", **data)
    option_policy_path = ""
    if option_policy_data:
        option_policy_path = str(out / "option_policy_dataset.npz")
        np.savez_compressed(option_policy_path, **option_policy_data)
    model, history = train_model(args, data, terminal)
    checkpoint = out / "subchain_pair_terminal.pt"
    save_subchain_terminal_checkpoint(checkpoint, model, cfg, {"samples": int(data["x"].shape[0])})
    base_mode = args.teacher_mode if args.teacher_mode == "recon_policy_terminal" else "static_recon"
    base_eval = eval_controller(args, "", base_mode)
    learned_eval = eval_controller(args, str(checkpoint), base_mode)
    report = {
        "status": "completed",
        "checkpoint_path": str(checkpoint),
        "samples": int(data["x"].shape[0]),
        "option_policy_dataset": option_policy_path,
        "option_policy_samples": int(option_policy_data.get("observations", np.asarray([])).shape[0])
        if option_policy_data
        else 0,
        "option_policy_source_counts": {
            str(item): int(np.sum(option_policy_data.get("sources", np.asarray([])) == item))
            for item in np.unique(option_policy_data.get("sources", np.asarray([])))
        }
        if option_policy_data
        else {},
        "source_counts": {str(item): int(np.sum(data.get("sources", np.asarray([])) == item)) for item in np.unique(data.get("sources", np.asarray([])))},
        "pairs": int(max(0, args.n_poles - 1)),
        "history": history,
        "base_eval": base_eval,
        "learned_subchain_eval": learned_eval,
        "config": vars(args),
        "mechanisms": {
            "shared_subchain_terminal": True,
            "supervised_teacher_distillation": str(args.label_mode) == "teacher_force",
            "counterfactual_recovery_labels": str(args.label_mode) == "counterfactual_recovery",
            "long_option_counterfactuals": str(args.label_mode) == "counterfactual_recovery"
            and int(getattr(args, "option_tail_steps", 0)) > 0,
            "policy_terminal": args.teacher_mode == "recon_policy_terminal",
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared learned adjacent-subchain pair terminal.")
    parser.add_argument("--out", default="reports/subchain_pair_terminal")
    parser.add_argument("--label-mode", choices=["teacher_force", "counterfactual_recovery"], default="teacher_force")
    parser.add_argument("--teacher-mode", choices=["static_recon", "recon_policy_terminal"], default="recon_policy_terminal")
    parser.add_argument("--policy-terminal-path", default="")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--policy-terminal-observation-mode", default="normalized_raw")
    parser.add_argument("--option-policy-observation-mode", default="normalized_raw4_prev_force")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=8_000_000)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--failure-offsets", type=int, nargs="*", default=[0, 2, 5, 10, 20, 40])
    parser.add_argument("--max-failure-states", type=int, default=8)
    parser.add_argument("--success-preserve-stride", type=int, default=0)
    parser.add_argument("--max-success-preserve-states", type=int, default=8)
    parser.add_argument("--option-hold-steps", type=int, default=2)
    parser.add_argument("--option-tail-steps", type=int, default=0)
    parser.add_argument("--append-option-trace", action="store_true", default=True)
    parser.add_argument("--no-append-option-trace", dest="append_option_trace", action="store_false")
    parser.add_argument("--option-trace-stride", type=int, default=4)
    parser.add_argument("--max-option-trace-states", type=int, default=12)
    parser.add_argument("--probe-horizon", type=int, default=100)
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--pressure-drop-weight", type=float, default=2.0)
    parser.add_argument("--pressure-final-weight", type=float, default=0.25)
    parser.add_argument("--shift-penalty", type=float, default=0.05)
    parser.add_argument("--tail-shift-penalty", type=float, default=0.02)
    parser.add_argument("--confidence-score-scale", type=float, default=4.0)
    parser.add_argument("--recovery-weight", type=float, default=3.0)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--weak-preserve-weight", type=float, default=0.25)
    parser.add_argument("--preserve-confidence", type=float, default=0.10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--no-pair-position", action="store_true")
    parser.add_argument("--theta-scale", type=float, default=0.21)
    parser.add_argument("--velocity-scale", type=float, default=5.0)
    parser.add_argument("--x-scale", type=float, default=2.4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=8101)
    parser.add_argument("--subchain-blend", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-pair-pressure", type=float, default=0.02)
    parser.add_argument("--max-force-fraction", type=float, default=1.0)
    parser.add_argument("--confidence-boost", type=float, default=0.08)
    parser.add_argument("--urgency-boost", type=float, default=0.15)
    parser.add_argument("--eval-seed-start", type=int, default=1_900_000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "out": report["config"]["out"],
                "checkpoint_path": report["checkpoint_path"],
                "base_success": report["base_eval"].get("success_rate", 0.0),
                "learned_success": report["learned_subchain_eval"].get("success_rate", 0.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
