from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.control.controllers import heuristic_force
from recon_cartpole.control.policy_observation import policy_observation_from_state
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig


def action_force(action: Any, force_mag: float, bins: int) -> float:
    idx = int(np.asarray(action).reshape(-1)[0])
    bins = max(2, int(bins))
    if bins == 2:
        return float(force_mag if idx == 1 else -force_mag)
    return float(np.linspace(-force_mag, force_mag, bins)[max(0, min(bins - 1, idx))])


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


def explicit_seeds(args: argparse.Namespace) -> list[int] | None:
    seed_list = str(getattr(args, "seed_list", "") or "").strip()
    if not seed_list:
        return None
    path = Path(seed_list)
    raw = path.read_text(encoding="utf-8")
    seeds: list[int] = []
    for item in raw.replace(",", "\n").splitlines():
        value = item.strip()
        if value:
            seeds.append(int(value))
    return seeds


def make_teacher(args: argparse.Namespace):
    if args.teacher == "heuristic":
        return None
    if args.teacher in {"static_recon", "recon_policy_terminal"}:
        return ReConCartPoleController(
            RunnerConfig(
                n_poles=args.n_poles,
                mode=args.teacher,
                action_mode="discrete",
                discrete_action_bins=args.discrete_action_bins,
                force_mag=args.force_mag,
                selection_mode=args.selection_mode,
                learn=False,
                policy_terminal_path=args.policy_terminal_path,
                policy_terminal_blend=args.policy_terminal_blend,
                policy_terminal_scope=args.policy_terminal_scope,
                policy_terminal_observation_mode=getattr(args, "teacher_observation_mode", args.observation_mode),
            )
        )
    raise ValueError(f"unsupported teacher: {args.teacher}")


def label_action_and_force(args: argparse.Namespace, teacher_action: int, teacher_force: float, rollout_action: int, rollout_force: float) -> tuple[int, float, str]:
    source = str(getattr(args, "label_source", "teacher") or "teacher")
    if source == "teacher":
        return int(teacher_action), float(teacher_force), "teacher"
    if source == "rollout":
        return int(rollout_action), float(rollout_force), "rollout"
    raise ValueError(f"unsupported label_source: {source}")


def make_behavior(args: argparse.Namespace):
    rollout_policy = getattr(args, "rollout_policy", "teacher")
    if rollout_policy == "teacher":
        return None
    if rollout_policy == "mingru_terminal":
        if not args.behavior_checkpoint_path:
            raise ValueError("--behavior-checkpoint-path is required for --rollout-policy mingru_terminal")
        return MinGRUTerminal(
            args.n_poles,
            args.force_mag,
            args.discrete_action_bins,
            MinGRUTerminalConfig(
                enabled=True,
                hidden_size=args.behavior_hidden_size,
                sequence_length=args.behavior_sequence_length,
                observation_mode=args.behavior_observation_mode,
                include_prev_force=args.behavior_include_prev_force,
                include_context=args.behavior_include_context,
                blend=1.0,
                scope=args.policy_terminal_scope,
                confidence_floor=args.behavior_confidence_floor,
                checkpoint_path=args.behavior_checkpoint_path,
            ),
        )
    raise ValueError(f"unsupported rollout policy: {rollout_policy}")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    prev_forces: list[float] = []
    teacher_forces: list[float] = []
    teacher_actions: list[int] = []
    returns_to_go: list[float] = []
    failure_within_k: list[float] = []
    seeds: list[int] = []
    sources: list[str] = []
    rollout_sources: list[str] = []
    rollout_forces: list[float] = []
    rollout_actions: list[int] = []
    episodes: list[int] = []
    step_indices: list[int] = []

    teacher = make_teacher(args)
    behavior = make_behavior(args)
    env = make_env(args)
    seed_values = explicit_seeds(args)
    if seed_values is None:
        seed_values = [args.seed_start + ep for ep in range(args.episodes)]
    for ep, seed in enumerate(seed_values):
        obs, info = env.reset(seed=seed)
        if teacher is not None:
            teacher.start_episode()
        if behavior is not None:
            behavior.reset()
        episode_rows: list[dict[str, Any]] = []
        prev_force = 0.0
        for step in range(args.horizon):
            raw = info.get("raw_state")
            policy_obs = policy_observation_from_state(
                obs,
                raw,
                args.n_poles,
                args.observation_mode,
                previous_force=prev_force,
                force_mag=args.force_mag,
            )
            if teacher is None:
                from recon_cartpole.control.sensors import features_from_state

                force = heuristic_force(features_from_state(obs, raw, args.n_poles), args.force_mag)
                action = int(np.argmin(np.abs(np.linspace(-args.force_mag, args.force_mag, args.discrete_action_bins) - force)))
                diagnostics = {"force": force}
            else:
                action, diagnostics = teacher.act(obs, raw)
                force = float(diagnostics.get("force", action_force(action, args.force_mag, args.discrete_action_bins)))
            if behavior is None:
                rollout_action = int(action)
                rollout_force = float(force)
                rollout_source = args.teacher
            else:
                prediction = behavior.predict(obs, raw, {})
                rollout_force = 0.0 if prediction.force is None else float(prediction.force)
                rollout_action = int(action_from_force(rollout_force, "discrete", args.force_mag, args.discrete_action_bins))
                rollout_source = getattr(args, "rollout_policy", "teacher")
            label_action, label_force, label_source = label_action_and_force(
                args, int(action), float(force), int(rollout_action), float(rollout_force)
            )
            next_obs, reward, terminated, truncated, info = env.step(rollout_action)
            episode_rows.append(
                {
                    "observation": policy_obs.astype(np.float32),
                    "prev_force": float(prev_force),
                    "force": float(label_force),
                    "action": int(label_action),
                    "teacher_force": float(force),
                    "teacher_action": int(action),
                    "label_source": label_source,
                    "rollout_force": float(rollout_force),
                    "rollout_action": int(rollout_action),
                    "rollout_source": rollout_source,
                    "reward": float(reward),
                    "seed": seed,
                    "episode": ep,
                    "step": step,
                }
            )
            prev_force = rollout_force
            obs = next_obs
            if terminated or truncated:
                break
        rewards = np.asarray([row["reward"] for row in episode_rows], dtype=np.float32)
        rtg = np.cumsum(rewards[::-1])[::-1] if rewards.size else np.asarray([], dtype=np.float32)
        last = len(episode_rows) - 1
        for idx, row in enumerate(episode_rows):
            observations.append(row["observation"])
            prev_forces.append(row["prev_force"])
            teacher_forces.append(row["force"])
            teacher_actions.append(row["action"])
            returns_to_go.append(float(rtg[idx]))
            failure_within_k.append(float((last - idx) <= args.failure_window and last + 1 < args.horizon))
            seeds.append(row["seed"])
            sources.append(row["label_source"])
            rollout_sources.append(row["rollout_source"])
            rollout_forces.append(row["rollout_force"])
            rollout_actions.append(row["rollout_action"])
            episodes.append(row["episode"])
            step_indices.append(row["step"])
    return {
        "observations": np.stack(observations).astype(np.float32),
        "prev_forces": np.asarray(prev_forces, dtype=np.float32),
        "teacher_forces": np.asarray(teacher_forces, dtype=np.float32),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.int64),
        "returns_to_go": np.asarray(returns_to_go, dtype=np.float32),
        "failure_within_k": np.asarray(failure_within_k, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "sources": np.asarray(sources),
        "rollout_sources": np.asarray(rollout_sources),
        "rollout_forces": np.asarray(rollout_forces, dtype=np.float32),
        "rollout_actions": np.asarray(rollout_actions, dtype=np.int64),
        "episodes": np.asarray(episodes, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["heuristic", "static_recon", "recon_policy_terminal"], default="static_recon")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=710000)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw")
    parser.add_argument("--teacher-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw")
    parser.add_argument("--policy-terminal-path", default="")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--rollout-policy", choices=["teacher", "mingru_terminal"], default="teacher")
    parser.add_argument("--label-source", choices=["teacher", "rollout"], default="teacher")
    parser.add_argument("--behavior-checkpoint-path", default="")
    parser.add_argument("--behavior-hidden-size", type=int, default=64)
    parser.add_argument("--behavior-sequence-length", type=int, default=16)
    parser.add_argument("--behavior-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw4_prev_force")
    parser.add_argument("--behavior-include-prev-force", action="store_true", default=True)
    parser.add_argument("--behavior-no-prev-force", dest="behavior_include_prev_force", action="store_false")
    parser.add_argument("--behavior-include-context", action="store_true", default=True)
    parser.add_argument("--behavior-no-context", dest="behavior_include_context", action="store_false")
    parser.add_argument("--behavior-confidence-floor", type=float, default=0.05)
    parser.add_argument("--failure-window", type=int, default=50)
    parser.add_argument("--out", default="reports/mingru_dataset/dataset.npz")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect(args)
    np.savez_compressed(out, **data)
    metadata = {
        "teacher": args.teacher,
        "episodes": int(len(explicit_seeds(args) or [None] * args.episodes)),
        "seed_list": args.seed_list,
        "samples": int(data["observations"].shape[0]),
        "env": {
            "n_poles": args.n_poles,
            "horizon": args.horizon,
            "dt": args.dt,
            "dynamics_mode": args.dynamics_mode,
            "discrete_action_bins": args.discrete_action_bins,
            "force_mag": args.force_mag,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
            "observation_mode": args.observation_mode,
            "teacher_observation_mode": args.teacher_observation_mode,
            "rollout_policy": args.rollout_policy,
            "label_source": args.label_source,
            "behavior_checkpoint_path": args.behavior_checkpoint_path,
            "behavior_observation_mode": args.behavior_observation_mode,
        },
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(out), "samples": metadata["samples"]}, indent=2))


if __name__ == "__main__":
    main()
