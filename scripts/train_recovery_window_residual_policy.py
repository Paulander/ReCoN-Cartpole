from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from recon_cartpole.control.policy_observation import policy_observation_from_state, policy_observation_size
from recon_cartpole.control.residual_features import residual_aux_feature_size, residual_aux_features
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_counterfactual_residual_terminal import (  # noqa: E402
    collect_episode_states,
    collect_seed_values,
    recovery_pressure,
    select_failure_states,
    set_env_state,
)


def make_cartpole_env(args: argparse.Namespace) -> CartPoleNEnv:
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


def make_base_controller(args: argparse.Namespace, residual_model_path: str = "") -> ReConCartPoleController:
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
            residual_policy_terminal_gate_threshold=args.residual_gate_threshold,
            residual_policy_terminal_feature_mode=args.residual_feature_mode,
            residual_policy_terminal_hold_steps=args.residual_hold_steps,
        )
    )


def window_rows_from_episode(args: argparse.Namespace, episode: dict[str, Any]) -> list[dict[str, Any]]:
    if bool(episode.get("success", False)):
        return []
    rows: list[dict[str, Any]] = []
    for state in select_failure_states(args, episode):
        rows.append(
            {
                "seed": int(episode["seed"]),
                "step": int(state["step"]),
                "raw_state": list(state["raw_before"]),
                "base_force": float(state.get("force", 0.0)),
                "failure_offset": int(state.get("failure_offset", -1)),
                "recovery_pressure": float(state.get("recovery_pressure", 0.0)),
            }
        )
    return rows


def collect_windows(args: argparse.Namespace) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for seed in collect_seed_values(args):
        episode = collect_episode_states(args, int(seed))
        episodes.append({key: episode[key] for key in ("seed", "steps", "return", "success")})
        windows.extend(window_rows_from_episode(args, episode))
    return {"episodes": episodes, "windows": windows}


class RecoveryWindowResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, args: argparse.Namespace, windows: list[dict[str, Any]]):
        super().__init__()
        if not windows:
            raise ValueError("RecoveryWindowResidualEnv requires at least one window")
        self.args = args
        self.windows = list(windows)
        self.window_index = 0
        self.env = make_cartpole_env(args)
        self.controller = make_base_controller(args)
        self.local_step = 0
        self.previous_force = 0.0
        self.current_window: dict[str, Any] = {}
        self.base_action = int(args.discrete_action_bins) // 2
        self.base_force = 0.0
        self.last_raw = np.zeros(2 + 2 * int(args.n_poles), dtype=np.float32)
        self.last_obs = np.zeros(2 + 3 * int(args.n_poles), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(max(2, int(args.residual_action_bins)))
        obs_size = policy_observation_size(args.n_poles, args.base_observation_mode)
        obs_size += residual_aux_feature_size(args.residual_feature_mode)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    def _choose_window(self) -> dict[str, Any]:
        if bool(getattr(self.args, "cycle_windows", False)):
            window = self.windows[self.window_index % len(self.windows)]
            self.window_index += 1
            return window
        idx = int(self.np_random.integers(0, len(self.windows)))
        return self.windows[idx]

    def _prepare_base(self) -> None:
        self.last_obs = self.env._get_obs()
        self.last_raw = np.asarray(self.env.raw_state, dtype=np.float32)
        self.base_action, diagnostics = self.controller.act(self.last_obs, self.last_raw)
        self.base_action = int(self.base_action)
        self.base_force = float(diagnostics.get("force", 0.0))

    def _residual_obs(self) -> np.ndarray:
        policy_obs = policy_observation_from_state(
            self.last_obs,
            self.last_raw,
            self.args.n_poles,
            self.args.base_observation_mode,
            previous_force=self.previous_force,
            force_mag=self.args.force_mag,
        ).astype(np.float32, copy=False)
        aux = residual_aux_features(
            self.last_raw,
            n_poles=self.args.n_poles,
            force_mag=self.args.force_mag,
            base_force=self.base_force,
            previous_force=self.previous_force,
            horizon=self.args.horizon,
            episode_step=int(self.env.steps),
            mode=self.args.residual_feature_mode,
        )
        return np.concatenate([policy_obs, aux]).astype(np.float32, copy=False)

    def _risk_gate(self) -> float:
        return float(
            residual_aux_features(
                self.last_raw,
                n_poles=self.args.n_poles,
                force_mag=self.args.force_mag,
                base_force=self.base_force,
                previous_force=self.previous_force,
                horizon=self.args.horizon,
                episode_step=int(self.env.steps),
                mode="basic",
            )[1]
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.local_step = 0
        self.current_window = dict(self._choose_window())
        self.previous_force = float(self.current_window.get("base_force", 0.0))
        reset_seed = int(self.current_window.get("seed", 0))
        self.env.reset(seed=reset_seed, options=options)
        set_env_state(self.env, self.current_window["raw_state"], int(self.current_window["step"]))
        self.controller.start_episode()
        self._prepare_base()
        info = {
            "window_seed": int(self.current_window.get("seed", -1)),
            "window_step": int(self.current_window.get("step", -1)),
            "window_pressure": float(self.current_window.get("recovery_pressure", 0.0)),
        }
        return self._residual_obs(), info

    def step(self, action: Any):
        pressure_before = recovery_pressure(self.last_raw, int(self.args.n_poles))
        action_idx = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, self.action_space.n - 1))
        max_shift = self.action_space.n // 2
        requested_shift = action_idx - max_shift
        gate = self._risk_gate()
        applied_shift = requested_shift if gate >= float(self.args.residual_gate_threshold) else 0
        base_idx = int(np.clip(self.base_action, 0, int(self.args.discrete_action_bins) - 1))
        final_idx = int(np.clip(base_idx + applied_shift, 0, int(self.args.discrete_action_bins) - 1))
        force_values = np.linspace(-self.args.force_mag, self.args.force_mag, int(self.args.discrete_action_bins))
        final_force = float(force_values[final_idx])
        obs, env_reward, terminated, env_truncated, info = self.env.step(final_idx)
        self.local_step += 1
        pressure_after = recovery_pressure(info.get("raw_state", []), int(self.args.n_poles))
        pressure_drop = float(pressure_before - pressure_after)
        shift_size = abs(float(applied_shift))
        shift_penalty = float(self.args.shift_penalty) * shift_size
        low_risk_change_penalty = float(getattr(self.args, "low_risk_change_penalty", 0.0)) * shift_size * max(0.0, 1.0 - gate)
        reward = (
            float(env_reward)
            + float(self.args.pressure_drop_weight) * pressure_drop
            - float(self.args.pressure_after_weight) * pressure_after
            - shift_penalty
            - low_risk_change_penalty
        )
        if terminated:
            reward -= float(self.args.failure_penalty)
        local_truncated = self.local_step >= int(self.args.window_horizon)
        truncated = bool(env_truncated or local_truncated)
        if truncated and not terminated:
            reward += float(self.args.window_success_bonus)
        self.previous_force = final_force
        if not (terminated or truncated):
            self._prepare_base()
            next_obs = self._residual_obs()
        else:
            self.last_obs = obs
            self.last_raw = np.asarray(info.get("raw_state", []), dtype=np.float32)
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            **info,
            "base_action": base_idx,
            "final_action": final_idx,
            "requested_shift": int(requested_shift),
            "applied_shift": int(applied_shift),
            "risk_gate": float(gate),
            "pressure_before": float(pressure_before),
            "pressure_after": float(pressure_after),
            "pressure_drop": pressure_drop,
            "shift_penalty": float(shift_penalty),
            "low_risk_change_penalty": float(low_risk_change_penalty),
            "final_force": final_force,
        }
        return next_obs, float(reward), bool(terminated), bool(truncated), info


def make_window_env(args: argparse.Namespace, windows: list[dict[str, Any]]) -> RecoveryWindowResidualEnv:
    return RecoveryWindowResidualEnv(args, windows)


def eval_seed_values(args: argparse.Namespace) -> list[int]:
    starts = args.eval_seed_starts or [args.eval_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.eval_episodes)))
    return seeds


def tail_metrics(steps: list[float], horizon: int) -> dict[str, Any]:
    summary = summarize_steps(steps, horizon)
    values = np.asarray(steps, dtype=float)
    if values.size:
        count = max(1, int(np.ceil(values.size * 0.10)))
        summary["cvar_survival"] = float(np.mean(np.sort(values)[:count]))
    else:
        summary["cvar_survival"] = 0.0
    return summary


def evaluate_controller(args: argparse.Namespace, residual_path: str = "") -> dict[str, Any]:
    controller = make_base_controller(args, residual_path)
    steps: list[float] = []
    deltas: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed in eval_seed_values(args):
        result = rollout(make_cartpole_env(args), controller, seed=int(seed), horizon=args.horizon, trace=bool(residual_path))
        step_count = float(result["steps"])
        steps.append(step_count)
        if residual_path:
            for item in result.get("trace", []):
                residual = ((item.get("policy_terminal") or {}).get("residual_policy_terminal") or {})
                deltas.append(abs(float(residual.get("residual_delta", 0.0) or 0.0)))
        per_seed.append({"seed": int(seed), "steps": int(step_count), "success": bool(step_count >= args.horizon)})
    summary = tail_metrics(steps, args.horizon)
    summary.update({"episodes": len(steps), "mean_abs_residual_delta": float(np.mean(deltas)) if deltas else 0.0, "per_seed": per_seed})
    return summary


def write_markdown(report: dict[str, Any], path: Path) -> None:
    base = report["base_eval"]
    residual = report["residual_eval"]
    lines = [
        "# Recovery-Window Residual PPO",
        "",
        f"Status: `{report['status']}`",
        f"Residual model: `{report['residual_model_path']}`",
        f"Windows: `{report['window_count']}` from `{report['collection_episode_count']}` collection episodes",
        f"Window horizon: `{report['window_horizon']}`; residual feature mode: `{report['residual_feature_mode']}`",
        f"Reward config: `{report.get('reward_config', {})}`",
        "",
        "| evaluator | mean | p10 | cvar | success | mean abs delta | episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| recon_frozen_base | {base['mean_survival']:.1f} | {base['p10_survival']:.1f} | {base.get('cvar_survival', 0.0):.1f} | {base['success_rate']:.3f} | 0.000 | {base['episodes']} |",
        f"| recon_recovery_window_residual | {residual['mean_survival']:.1f} | {residual['p10_survival']:.1f} | {residual.get('cvar_survival', 0.0):.1f} | {residual['success_rate']:.3f} | {residual.get('mean_abs_residual_delta', 0.0):.3f} | {residual['episodes']} |",
        "",
        "## Claim Discipline",
        "",
        "Training starts from non-held-out recovery windows; evaluation uses separate held-out seeds through normal ReCoN residual-terminal integration. No train-seed solve claim is made.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install RL extras with `uv sync --extra rl`") from exc
    collected = collect_windows(args)
    windows = collected["windows"]
    if not windows:
        raise ValueError("no recovery windows collected")
    (out / "windows.json").write_text(json.dumps({"windows": windows, "episodes": collected["episodes"]}, indent=2), encoding="utf-8")
    if args.residual_model_path:
        model_path = Path(args.residual_model_path)
        model = PPO.load(str(model_path), device=args.device)
        status = "evaluated"
        timesteps = 0
    else:
        train_env = make_vec_env(lambda: make_window_env(args, windows), n_envs=int(args.n_envs), seed=int(args.train_seed))
        policy_kwargs = {"net_arch": [int(item) for item in args.net_arch.split(",") if item.strip()]}
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=int(args.train_seed),
            verbose=int(args.verbose),
            device=args.device,
            learning_rate=float(args.learning_rate),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            policy_kwargs=policy_kwargs,
        )
        model.learn(total_timesteps=int(args.timesteps))
        model_path = out / "recovery_window_residual_policy.zip"
        model.save(str(model_path))
        train_env.close()
        status = "completed"
        timesteps = int(args.timesteps)
    report = {
        "status": status,
        "base_model_path": args.base_model_path,
        "residual_model_path": str(model_path),
        "timesteps": timesteps,
        "window_count": len(windows),
        "collection_episode_count": len(collected["episodes"]),
        "window_horizon": int(args.window_horizon),
        "residual_feature_mode": args.residual_feature_mode,
        "reward_config": {
            "pressure_drop_weight": float(args.pressure_drop_weight),
            "pressure_after_weight": float(args.pressure_after_weight),
            "shift_penalty": float(args.shift_penalty),
            "low_risk_change_penalty": float(getattr(args, "low_risk_change_penalty", 0.0)),
            "failure_penalty": float(args.failure_penalty),
            "window_success_bonus": float(args.window_success_bonus),
        },
        "base_eval": evaluate_controller(args),
        "residual_eval": evaluate_controller(args, str(model_path)),
        "eval_seeds": eval_seed_values(args),
        "mechanisms": {
            "recovery_window_resets": True,
            "learned_residual_policy": True,
            "frozen_base_policy_terminal": True,
            "recon_integration_eval": True,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out / "report.md")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO residual policy from near-failure recovery-window resets.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--residual-model-path", default="")
    parser.add_argument("--base-normalizer-path", default="")
    parser.add_argument("--base-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw4")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics", "subchain_diagnostics"], default="subchain_diagnostics")
    parser.add_argument("--residual-action-bins", type=int, default=5)
    parser.add_argument("--residual-gate-threshold", type=float, default=0.20)
    parser.add_argument("--residual-hold-steps", type=int, default=1)
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--window-horizon", type=int, default=120)
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
    parser.add_argument("--max-failure-states", type=int, default=10)
    parser.add_argument("--use-failure-window", action="store_true")
    parser.add_argument("--failure-window-start", type=int, default=0)
    parser.add_argument("--failure-window-end", type=int, default=120)
    parser.add_argument("--failure-window-stride", type=int, default=5)
    parser.add_argument("--failure-window-target-offset", type=int, default=40)
    parser.add_argument("--max-window-states", type=int, default=14)
    parser.add_argument("--cycle-windows", action="store_true")
    parser.add_argument("--probe-horizon", type=int, default=100)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--shift-penalty", type=float, default=0.02)
    parser.add_argument("--low-risk-change-penalty", type=float, default=0.0)
    parser.add_argument("--min-score-gap", type=float, default=0.03)
    parser.add_argument("--min-survival-gain", type=int, default=0)
    parser.add_argument("--min-margin-gain", type=float, default=0.0)
    parser.add_argument("--min-pressure-gain", type=float, default=-999.0)
    parser.add_argument("--pressure-drop-weight", type=float, default=2.0)
    parser.add_argument("--pressure-after-weight", type=float, default=0.10)
    parser.add_argument("--failure-penalty", type=float, default=2.0)
    parser.add_argument("--window-success-bonus", type=float, default=1.0)
    parser.add_argument("--counterfactual-no-noise", action="store_true")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--train-seed", type=int, default=2900)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--clip-range", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--eval-seed-start", type=int, default=2100000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/recovery_window_residual_policy")
    return parser


def main() -> None:
    report = train(build_parser().parse_args())
    print(
        json.dumps(
            {
                "out": report["residual_model_path"],
                "base_success": report["base_eval"]["success_rate"],
                "residual_success": report["residual_eval"]["success_rate"],
                "window_count": report["window_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
