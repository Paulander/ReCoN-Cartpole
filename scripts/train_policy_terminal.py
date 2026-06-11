from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from recon_cartpole.control.policy_observation import (
    policy_observation_from_state,
    policy_observation_size,
)
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout


def _env_config(env: gym.Env):
    current: Any = env
    while current is not None:
        config = getattr(current, "config", None)
        if config is not None:
            return config
        current = getattr(current, "env", None)
    return None


class UprightShapingWrapper(gym.Wrapper):
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = np.asarray(info.get("raw_state", []), dtype=float)
        if raw.size >= 2 + 2 * self.env.config.n_poles:
            n = self.env.config.n_poles
            x = float(raw[0]) / max(self.env.config.x_threshold, 1e-9)
            theta = raw[2 : 2 + n] / max(self.env.config.theta_threshold_radians, 1e-9)
            theta_dot = raw[2 + n : 2 + 2 * n] / 5.0
            shaped = (
                1.0
                - 0.35 * float(np.mean(theta * theta))
                - 0.05 * x * x
                - 0.02 * float(np.mean(theta_dot * theta_dot))
            )
            reward = max(-1.0, shaped)
        if terminated:
            reward = -1.0
        return obs, float(reward), terminated, truncated, info


class SuccessBonusWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, bonus: float):
        super().__init__(env)
        self.bonus = max(0.0, float(bonus))
        self.config = _env_config(env)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated and not terminated and self.bonus > 0.0:
            reward = float(reward) + self.bonus
            info = {**info, "success_bonus": self.bonus}
        return obs, float(reward), terminated, truncated, info


class FailurePenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float):
        super().__init__(env)
        self.penalty = max(0.0, float(penalty))
        self.config = _env_config(env)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and self.penalty > 0.0:
            reward = float(reward) - self.penalty
            info = {**info, "failure_penalty": self.penalty}
        return obs, float(reward), terminated, truncated, info



class LateSurvivalBonusWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, bonus: float, start_fraction: float):
        super().__init__(env)
        self.bonus = max(0.0, float(bonus))
        self.start_fraction = max(0.0, min(1.0, float(start_fraction)))
        self.config = _env_config(env)
        self.elapsed_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.config is not None and self.bonus > 0.0:
            horizon = max(1, int(self.config.horizon))
            start_step = int(round(horizon * self.start_fraction))
            if self.elapsed_steps >= start_step:
                reward = float(reward) + self.bonus
                info = {**info, "late_survival_bonus": self.bonus}
        return obs, float(reward), terminated, truncated, info


class HardSeedResetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seeds: list[int], probability: float = 1.0):
        super().__init__(env)
        if not seeds:
            raise ValueError("HardSeedResetWrapper requires at least one seed")
        self.seeds = list(seeds)
        self.probability = max(0.0, min(1.0, float(probability)))
        self.index = 0
        self.rng = np.random.default_rng(99173)
        self.config = _env_config(env)

    def _apply_worker_seed(self, seed: int) -> None:
        seed_int = int(seed)
        self.rng = np.random.default_rng(99173 + seed_int)
        self.index = seed_int % len(self.seeds)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._apply_worker_seed(seed)
        if self.rng.random() < self.probability:
            chosen = self.seeds[self.index % len(self.seeds)]
            self.index += 1
            return self.env.reset(seed=chosen, options=options)
        return self.env.reset(seed=seed, options=options)


class PolicyObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, mode: str):
        super().__init__(env)
        self.mode = mode
        self.config = _env_config(env)
        self.previous_force = 0.0
        if mode == "env":
            self.observation_space = env.observation_space
        else:
            if self.config is None:
                raise ValueError("PolicyObservationWrapper requires env.config")
            size = policy_observation_size(self.config.n_poles, mode)
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(size,), dtype=np.float32
            )

    def _transform(self, observation: Any, info: dict[str, Any]):
        if self.mode == "env":
            return observation
        if self.config is None:
            raise ValueError("PolicyObservationWrapper requires env.config")
        return policy_observation_from_state(
            observation,
            info.get("raw_state"),
            self.config.n_poles,
            self.mode,
            x_threshold=self.config.x_threshold,
            theta_threshold=self.config.theta_threshold_radians,
            previous_force=self.previous_force,
            force_mag=self.config.force_mag,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.previous_force = 0.0
        observation, info = self.env.reset(seed=seed, options=options)
        return self._transform(observation, info), info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.config is not None:
            self.previous_force = _force_from_env_action(action, self.config)
        return self._transform(observation, info), reward, terminated, truncated, info


class FrameStackObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, frame_stack: int):
        super().__init__(env)
        self.frame_stack = max(1, int(frame_stack))
        self.frames: list[np.ndarray] = []
        self.config = _env_config(env)
        base_space = env.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("FrameStackObservationWrapper requires a Box observation space")
        low = np.tile(np.asarray(base_space.low, dtype=np.float32), self.frame_stack)
        high = np.tile(np.asarray(base_space.high, dtype=np.float32), self.frame_stack)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: Any):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.frames.append(obs)
        self.frames = self.frames[-self.frame_stack :]
        pad_count = self.frame_stack - len(self.frames)
        frames = [self.frames[0]] * pad_count + self.frames
        return np.concatenate(frames).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.frames = []
        return super().reset(seed=seed, options=options)


def parse_seed_list(value: Any) -> list[int]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    path = Path(text)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
            if isinstance(data, dict):
                for key in ("hard_seeds", "failed_seeds", "seeds"):
                    if key in data:
                        return [int(item) for item in data[key]]
            if isinstance(data, list):
                return [int(item) for item in data]
        text = raw
    return [int(part) for part in text.replace("\n", ",").split(",") if part.strip()]


def hard_train_seeds(args: argparse.Namespace) -> list[int]:
    return parse_seed_list(_arg(args, "hard_train_seeds", ""))


def make_env(
    args: argparse.Namespace,
    reward_mode: str = "survival",
    use_hard_seeds: bool = False,
    use_frame_stack: bool = True,
    use_success_bonus: bool = True,
    use_failure_penalty: bool = True,
):
    env = CartPoleNEnv(
        CartPoleNConfig(
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
        )
    )
    if use_hard_seeds:
        seeds = hard_train_seeds(args)
        if seeds:
            env = HardSeedResetWrapper(env, seeds, _arg(args, "hard_train_seed_probability", 1.0))
    if reward_mode == "upright_shaping":
        env = UprightShapingWrapper(env)
    late_survival_bonus = float(_arg(args, "late_survival_bonus", 0.0))
    if use_success_bonus and late_survival_bonus > 0.0:
        env = LateSurvivalBonusWrapper(
            env, late_survival_bonus, float(_arg(args, "late_survival_start_fraction", 0.80))
        )
    success_bonus = float(_arg(args, "success_bonus", 0.0))
    if use_success_bonus and success_bonus > 0.0:
        env = SuccessBonusWrapper(env, success_bonus)
    failure_penalty = float(_arg(args, "failure_penalty", 0.0))
    if use_failure_penalty and failure_penalty > 0.0:
        env = FailurePenaltyWrapper(env, failure_penalty)
    obs_mode = str(_arg(args, "policy_observation_mode", "env"))
    if obs_mode != "env":
        env = PolicyObservationWrapper(env, obs_mode)
    if use_frame_stack and int(_arg(args, "frame_stack", 1)) > 1:
        env = FrameStackObservationWrapper(env, int(_arg(args, "frame_stack", 1)))
    return env


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _force_from_env_action(action: Any, config: Any) -> float:
    if getattr(config, "action_mode", "discrete") == "continuous":
        return float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -config.force_mag, config.force_mag))
    bins = max(2, int(config.discrete_action_bins))
    idx = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, bins - 1))
    if bins == 2:
        return float(config.force_mag if idx == 1 else -config.force_mag)
    return float(np.linspace(-config.force_mag, config.force_mag, bins)[idx])


def policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    import torch as th

    arch_text = str(_arg(args, "net_arch", "64,64")).strip()
    arch = [int(item) for item in arch_text.split(",") if item.strip()]
    activation_name = str(_arg(args, "activation", "tanh")).lower()
    activation_fn = th.nn.ReLU if activation_name == "relu" else th.nn.Tanh
    return {"net_arch": arch, "activation_fn": activation_fn}


def ppo_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "learning_rate": float(_arg(args, "learning_rate", 3e-4)),
        "n_steps": int(_arg(args, "n_steps", 2048)),
        "batch_size": int(_arg(args, "batch_size", 64)),
        "n_epochs": int(_arg(args, "n_epochs", 10)),
        "gamma": float(_arg(args, "gamma", 0.99)),
        "gae_lambda": float(_arg(args, "gae_lambda", 0.95)),
        "clip_range": float(_arg(args, "clip_range", 0.2)),
        "ent_coef": float(_arg(args, "ent_coef", 0.0)),
        "vf_coef": float(_arg(args, "vf_coef", 0.5)),
        "max_grad_norm": float(_arg(args, "max_grad_norm", 0.5)),
        "policy_kwargs": policy_kwargs(args),
    }


def evaluate_model(model: Any, args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    steps: list[float] = []
    returns: list[float] = []
    for seed in seeds:
        env = make_env(
            args,
            reward_mode="survival",
            use_frame_stack=True,
            use_success_bonus=False,
            use_failure_penalty=False,
        )
        obs, _info = env.reset(seed=seed)
        total = 0.0
        for step in range(args.horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                steps.append(float(step + 1))
                returns.append(total)
                break
        else:
            steps.append(float(args.horizon))
            returns.append(total)
    summary = summarize_steps(steps, args.horizon)
    summary.update(
        {"returns_mean": float(np.mean(returns)) if returns else 0.0, "episodes": len(seeds)}
    )
    return summary


def evaluate_recon_terminal(
    model_path: Path,
    args: argparse.Namespace,
    seeds: list[int],
    trace_seed: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_policy_terminal",
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=str(model_path),
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_frame_stack=int(_arg(args, "frame_stack", 1)),
            policy_terminal_scope=str(_arg(args, "policy_terminal_scope", "stabilize_chain")),
            policy_terminal_observation_mode=str(_arg(args, "policy_observation_mode", "env")),
        )
    )
    steps: list[float] = []
    returns: list[float] = []
    for seed in seeds:
        result = rollout(
            make_env(
                args,
                reward_mode="survival",
                use_frame_stack=False,
                use_success_bonus=False,
                use_failure_penalty=False,
            ),
            controller,
            seed=seed,
            horizon=args.horizon,
            trace=False,
        )
        steps.append(float(result["steps"]))
        returns.append(float(result["return"]))
    summary = summarize_steps(steps, args.horizon)
    summary.update(
        {"returns_mean": float(np.mean(returns)) if returns else 0.0, "episodes": len(seeds)}
    )
    if trace_seed is not None and out_dir is not None:
        trace_result = rollout(
            make_env(
                args,
                reward_mode="survival",
                use_frame_stack=False,
                use_success_bonus=False,
                use_failure_penalty=False,
            ),
            controller,
            seed=trace_seed,
            horizon=args.horizon,
            trace=True,
        )
        (out_dir / "recon_policy_terminal_trace.json").write_text(
            json.dumps({"steps": trace_result["trace"]}, indent=2), encoding="utf-8"
        )
    return summary


def train_policy_terminal(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Install RL extras with `uv sync --extra rl` to train policy terminals"
        ) from exc

    if args.model_path:
        model_path = Path(args.model_path)
        model = PPO.load(str(model_path), device=args.device)
        train_timesteps = 0
        status = "evaluated"
    else:
        vec_env_cls = SubprocVecEnv if _arg(args, "vec_env", "dummy") == "subproc" else None
        train_env = make_vec_env(
            lambda: make_env(args, reward_mode=args.reward_mode, use_hard_seeds=True),
            n_envs=args.n_envs,
            seed=args.train_seed,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs={"start_method": "fork"} if vec_env_cls is SubprocVecEnv else None,
        )
        if args.resume_model_path:
            model = PPO.load(str(args.resume_model_path), env=train_env, device=args.device)
            model.set_random_seed(args.train_seed)
            status = "resumed"
        else:
            model = PPO(
                args.policy,
                train_env,
                seed=args.train_seed,
                verbose=args.verbose,
                device=args.device,
                **ppo_kwargs(args),
            )
            status = "completed"
        model.learn(
            total_timesteps=args.timesteps, reset_num_timesteps=not bool(args.resume_model_path)
        )
        model_path = out / "ppo_policy_terminal.zip"
        model.save(str(model_path))
        train_timesteps = args.timesteps

    seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]
    ppo_eval = evaluate_model(model, args, seeds)
    recon_eval = evaluate_recon_terminal(
        model_path, args, seeds, trace_seed=args.eval_seed_start + 999_999, out_dir=out
    )
    report = {
        "status": status,
        "model_path": str(model_path),
        "train_timesteps": train_timesteps,
        "train_seed": args.train_seed,
        "eval_seeds": seeds,
        "env": {
            "n_poles": args.n_poles,
            "horizon": args.horizon,
            "dt": args.dt,
            "dynamics_mode": args.dynamics_mode,
            "action_mode": args.action_mode,
            "discrete_action_bins": args.discrete_action_bins,
            "force_mag": args.force_mag,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
            "frame_stack": int(_arg(args, "frame_stack", 1)),
            "policy_observation_mode": str(_arg(args, "policy_observation_mode", "env")),
            "success_bonus": float(_arg(args, "success_bonus", 0.0)),
            "failure_penalty": float(_arg(args, "failure_penalty", 0.0)),
            "late_survival_bonus": float(_arg(args, "late_survival_bonus", 0.0)),
            "late_survival_start_fraction": float(_arg(args, "late_survival_start_fraction", 0.80)),
            "vec_env": str(_arg(args, "vec_env", "dummy")),
        },
        "reward_mode": args.reward_mode,
        "hard_train_seeds": hard_train_seeds(args),
        "hard_train_seed_probability": _arg(args, "hard_train_seed_probability", 1.0),
        "selection_mode": args.selection_mode,
        "policy_terminal_blend": args.policy_terminal_blend,
        "policy_terminal_scope": str(_arg(args, "policy_terminal_scope", "stabilize_chain")),
        "ppo_config": {
            "policy": args.policy,
            "net_arch": _arg(args, "net_arch", "64,64"),
            "activation": _arg(args, "activation", "tanh"),
            "learning_rate": _arg(args, "learning_rate", 3e-4),
            "n_steps": _arg(args, "n_steps", 2048),
            "batch_size": _arg(args, "batch_size", 64),
            "n_epochs": _arg(args, "n_epochs", 10),
            "gamma": _arg(args, "gamma", 0.99),
            "gae_lambda": _arg(args, "gae_lambda", 0.95),
            "clip_range": _arg(args, "clip_range", 0.2),
            "ent_coef": _arg(args, "ent_coef", 0.0),
            "vf_coef": _arg(args, "vf_coef", 0.5),
            "max_grad_norm": _arg(args, "max_grad_norm", 0.5),
            "frame_stack": int(_arg(args, "frame_stack", 1)),
            "policy_observation_mode": str(_arg(args, "policy_observation_mode", "env")),
            "success_bonus": float(_arg(args, "success_bonus", 0.0)),
            "failure_penalty": float(_arg(args, "failure_penalty", 0.0)),
            "late_survival_bonus": float(_arg(args, "late_survival_bonus", 0.0)),
            "late_survival_start_fraction": float(_arg(args, "late_survival_start_fraction", 0.80)),
            "vec_env": str(_arg(args, "vec_env", "dummy")),
        },
        "pure_ppo_eval": ppo_eval,
        "recon_policy_terminal_eval": recon_eval,
        "mechanisms": {
            "ppo_policy_gradient": True,
            "policy_terminal": True,
            "edge_plasticity": False,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report_md(report, out / "report.md")
    return report


def write_report_md(report: dict[str, Any], path: Path) -> None:
    ppo = report["pure_ppo_eval"]
    recon = report["recon_policy_terminal_eval"]
    lines = [
        "# Policy Terminal Training Report",
        "",
        f"Status: `{report['status']}`",
        f"Reward mode: `{report['reward_mode']}`",
        f"Model: `{report['model_path']}`",
        f"Train timesteps: `{report['train_timesteps']}`",
        f"PPO config: `{report.get('ppo_config', {})}`",
        f"Policy terminal scope: `{report.get('policy_terminal_scope', 'stabilize_chain')}`",
        f"Frame stack: `{report.get('ppo_config', {}).get('frame_stack', 1)}`",
        f"Policy observation mode: `{report.get('ppo_config', {}).get('policy_observation_mode', 'env')}`",
        f"Success bonus: `{report.get('ppo_config', {}).get('success_bonus', 0.0)}`",
        f"Failure penalty: `{report.get('ppo_config', {}).get('failure_penalty', 0.0)}`",
        f"Late survival bonus: `{report.get('ppo_config', {}).get('late_survival_bonus', 0.0)}` from fraction `{report.get('ppo_config', {}).get('late_survival_start_fraction', 0.8)}`",
        f"Vec env: `{report.get('ppo_config', {}).get('vec_env', 'dummy')}`",
        f"Wall-clock seconds: `{report['wall_clock_seconds']:.2f}`",
        "",
        "| evaluator | mean | p10 | success | max | episodes |",
        "|---|---:|---:|---:|---:|---:|",
        f"| pure_ppo | {ppo['mean_survival']:.1f} | {ppo['p10_survival']:.1f} | {ppo['success_rate']:.2f} | {ppo['max_survival']:.1f} | {ppo['episodes']} |",
        f"| recon_policy_terminal | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['success_rate']:.2f} | {recon['max_survival']:.1f} | {recon['episodes']} |",
        "",
        "## Claim Discipline",
        "",
        "This report separates a learned PPO policy from ReCoN's graph scaffold. It is not pure symbolic ReCoN, and it is not a solved claim unless the held-out solve thresholds are met.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel"
    )
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument(
        "--model-path",
        default="",
        help="Evaluate an existing PPO policy zip instead of training a new one.",
    )
    parser.add_argument(
        "--resume-model-path",
        default="",
        help="Continue training an existing PPO policy zip, then save a new terminal policy.",
    )
    parser.add_argument("--train-seed", type=int, default=510_000)
    parser.add_argument(
        "--hard-train-seeds",
        default="",
        help="Comma-separated seeds or JSON/text file of seeds to cycle through during training resets.",
    )
    parser.add_argument("--hard-train-seed-probability", type=float, default=1.0)
    parser.add_argument("--eval-seed-start", type=int, default=620_000)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument(
        "--success-bonus",
        type=float,
        default=0.0,
        help="Training-only bonus when an episode reaches the horizon.",
    )
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=0.0,
        help="Training-only penalty subtracted when an episode terminates before the horizon.",
    )
    parser.add_argument(
        "--late-survival-bonus",
        type=float,
        default=0.0,
        help="Training-only per-step bonus after late-survival-start-fraction of the horizon.",
    )
    parser.add_argument(
        "--late-survival-start-fraction",
        type=float,
        default=0.80,
        help="Fraction of the horizon after which late-survival-bonus is applied.",
    )
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--reward-mode", choices=["survival", "upright_shaping"], default="survival"
    )
    parser.add_argument(
        "--selection-mode", choices=["soft_select", "hard_select"], default="hard_select"
    )
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument(
        "--policy-terminal-scope",
        choices=["stabilize_chain", "selected", "all"],
        default="stabilize_chain",
        help="Which ReCoN proposals can be force-blended with the PPO terminal.",
    )
    parser.add_argument(
        "--policy-observation-mode",
        choices=["env", "normalized_raw"],
        default="env",
        help="Observation representation used by the learned PPO terminal.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Concatenate this many recent observations for the learned PPO terminal.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/policy_terminal_train")
    args = parser.parse_args()
    result = train_policy_terminal(args)
    print(
        json.dumps(
            {
                "out": args.out,
                "model_path": result["model_path"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
