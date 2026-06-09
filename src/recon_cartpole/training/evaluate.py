from __future__ import annotations

import time
from typing import Any

import numpy as np

from recon_cartpole.control.rewards import reward_tick
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.envs.gym_cartpole_adapter import GymCartPoleAdapter
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.trace_schema import StepTrace


def make_env(env_name: str, n_poles: int = 1, render_mode: str | None = None):
    if env_name == "gym_cartpole_v1":
        return GymCartPoleAdapter(render_mode=render_mode)
    return CartPoleNEnv(CartPoleNConfig(n_poles=n_poles), render_mode=render_mode)


def rollout(
    env: Any,
    controller: ReConCartPoleController,
    seed: int = 0,
    horizon: int = 500,
    trace: bool = False,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    controller.start_episode()
    total_return = 0.0
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_reward_tick = 0.0
    reward_history: list[float] = []
    for step in range(horizon):
        controller.observe_reward(last_reward_tick)
        raw_before = info.get("raw_state")
        action, diagnostics = controller.act(observation, raw_before)
        before_obs = np.asarray(observation, dtype=float)
        next_observation, env_reward, terminated, truncated, info = env.step(action)
        total_return += float(env_reward)
        last_reward_tick = reward_tick(
            before_obs,
            next_observation,
            raw_before,
            info.get("raw_state"),
            controller.config.n_poles,
            terminated,
        )
        reward_history.append(float(last_reward_tick))
        if trace:
            traces.append(
                StepTrace(
                    step=step,
                    observation=np.asarray(next_observation, dtype=float).tolist(),
                    raw_state=np.asarray(info.get("raw_state", []), dtype=float).tolist(),
                    action=action,
                    force=float(diagnostics.get("force", 0.0)),
                    env_reward=float(env_reward),
                    reward_tick=float(last_reward_tick),
                    return_so_far=float(total_return),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    goal_vector=diagnostics.get("goal_vector", {}),
                    selected_regime=diagnostics.get("selected_regime", ""),
                    proposal=diagnostics.get("proposal", {}),
                    proposals=diagnostics.get("proposals", []),
                    suppressed_proposals=diagnostics.get("suppressed_proposals", []),
                    selection_mode=diagnostics.get("selection_mode", "soft_select"),
                    fired_edges=diagnostics.get("fired_edges", []),
                    plasticity=diagnostics.get("plasticity", {}),
                    fast_deltas=diagnostics.get("fast_deltas", {}),
                    node_params=diagnostics.get("node_params", {}),
                    node_param_deltas=diagnostics.get("node_param_deltas", {}),
                    mlp_terminal=diagnostics.get("mlp_terminal", {}),
                    policy_terminal=diagnostics.get("policy_terminal", {}),
                    bandit=diagnostics.get("bandit", {}),
                    consolidation=diagnostics.get("consolidation", {}),
                    graph_nodes=diagnostics.get("graph_nodes", {}),
                    graph_ticks=diagnostics.get("graph_ticks", []),
                ).to_dict()
            )
        observation = next_observation
        if terminated or truncated:
            break
    controller.observe_reward(last_reward_tick)
    episode_learning = controller.end_episode(reward_history, total_return, horizon)
    elapsed = time.perf_counter() - started
    return {
        "return": total_return,
        "steps": step + 1,
        "success": step + 1 >= horizon and total_return >= horizon - 1,
        "seconds": elapsed,
        "episode_learning": episode_learning,
        "trace": traces,
    }


def evaluate(
    env_name: str,
    mode: str,
    n_poles: int = 1,
    episodes: int = 10,
    horizon: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    results = []
    env = make_env(env_name, n_poles=n_poles)
    controller = ReConCartPoleController(RunnerConfig(n_poles=n_poles, mode=mode))
    for idx in range(episodes):
        results.append(rollout(env, controller, seed + idx, horizon))
    returns = np.asarray([item["return"] for item in results], dtype=float)
    steps = np.asarray([item["steps"] for item in results], dtype=float)
    return {
        "env": env_name,
        "mode": mode,
        "n_poles": n_poles,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "p10_return": float(np.percentile(returns, 10)),
        "p90_return": float(np.percentile(returns, 90)),
        "mean_survival_steps": float(np.mean(steps)),
        "success_rate_at_horizon": float(np.mean(steps >= horizon)),
    }

