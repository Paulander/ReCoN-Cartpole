from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.training.ablations import summarize_steps


@dataclass
class PPOBaselineConfig:
    n_poles: int
    horizon: int = 500
    train_timesteps: int = 50_000
    train_seed: int = 300_000
    eval_seeds: list[int] = field(default_factory=list)
    env_params: dict[str, Any] = field(default_factory=dict)
    policy: str = "MlpPolicy"
    device: str = "cpu"


def ppo_dependency_status() -> dict[str, Any]:
    missing: list[str] = []
    versions: dict[str, str] = {}
    for module_name in ("torch", "stable_baselines3"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - depends on optional env packages
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
    return {"available": not missing, "missing": missing, "versions": versions}


def make_ppo_env(config: PPOBaselineConfig, seed: int | None = None) -> CartPoleNEnv:
    env = CartPoleNEnv(
        CartPoleNConfig(
            n_poles=config.n_poles,
            horizon=config.horizon,
            **dict(config.env_params),
        )
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_ppo_model(model: Any, config: PPOBaselineConfig) -> dict[str, Any]:
    steps: list[float] = []
    returns: list[float] = []
    for seed in config.eval_seeds:
        env = make_ppo_env(config)
        obs, _ = env.reset(seed=seed)
        total_return = 0.0
        for step in range(config.horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                steps.append(float(step + 1))
                returns.append(total_return)
                break
        else:
            steps.append(float(config.horizon))
            returns.append(total_return)
    summary = summarize_steps(steps, config.horizon)
    summary.update({"steps": steps, "returns": returns})
    return summary


def run_ppo_baseline(config: PPOBaselineConfig) -> dict[str, Any]:
    deps = ppo_dependency_status()
    row: dict[str, Any] = {
        "mode": "ppo",
        "n_poles": config.n_poles,
        "horizon": config.horizon,
        "seeds": config.eval_seeds,
        "env_params": dict(config.env_params),
        "train_timesteps": config.train_timesteps,
        "policy": config.policy,
        "device": config.device,
        "mechanisms": {"ppo_policy_gradient": True},
        "dependency_status": deps,
    }
    if not deps["available"]:
        row.update(
            {
                "status": "unavailable",
                "mean_survival": 0.0,
                "p10_survival": 0.0,
                "success_rate": 0.0,
                "max_survival": 0.0,
                "note": "Install optional RL dependencies with `uv sync --extra rl` to run PPO.",
            }
        )
        return row

    started = time.perf_counter()
    from stable_baselines3 import PPO  # pragma: no cover - optional dependency path

    env = make_ppo_env(config, seed=config.train_seed)
    model = PPO(config.policy, env, seed=config.train_seed, verbose=0, device=config.device)
    model.learn(total_timesteps=config.train_timesteps)
    summary = evaluate_ppo_model(model, config)
    row.update(summary)
    row.update({"status": "completed", "wall_clock_seconds": time.perf_counter() - started})
    return row
