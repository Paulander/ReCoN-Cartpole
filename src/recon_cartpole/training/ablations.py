from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig

ABLATION_MODES = [
    "baseline_heuristic",
    "static_recon",
    "recon_fast",
    "recon_bandit",
    "recon_fast_bandit",
    "recon_slow",
]

GAIN_SEARCH_MODES = ["gain_search_only", "gain_search_recon_fast_bandit"]


def summarize_steps(steps: list[float], horizon: int) -> dict[str, float]:
    values = np.asarray(steps, dtype=float)
    if values.size == 0:
        return {"mean_survival": 0.0, "p10_survival": 0.0, "success_rate": 0.0, "max_survival": 0.0}
    return {
        "mean_survival": float(mean(values)),
        "p10_survival": float(np.percentile(values, 10)),
        "success_rate": float(np.mean(values >= horizon)),
        "max_survival": float(np.max(values)),
    }


def run_mode_on_seeds(
    mode: str,
    seeds: list[int],
    n_poles: int,
    horizon: int,
    env_params: dict[str, Any] | None = None,
    train_episodes: int = 0,
) -> dict[str, Any]:
    env_params = dict(env_params or {})
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=n_poles,
            mode=mode,
            learn=train_episodes > 0,
            reset_bandit_each_episode=False,
            stage=f"ablation_n{n_poles}",
        )
    )
    if train_episodes > 0:
        train_env = CartPoleNEnv(CartPoleNConfig(n_poles=n_poles, horizon=horizon, **env_params))
        from recon_cartpole.training.evaluate import rollout

        for idx in range(train_episodes):
            rollout(train_env, controller, seed=seeds[0] - 10_000 + idx, horizon=horizon, trace=False)
    controller.config.learn = False

    from recon_cartpole.training.evaluate import rollout

    steps = []
    returns = []
    for seed in seeds:
        env = CartPoleNEnv(CartPoleNConfig(n_poles=n_poles, horizon=horizon, **env_params))
        result = rollout(env, controller, seed=seed, horizon=horizon, trace=False)
        steps.append(float(result["steps"]))
        returns.append(float(result["return"]))
    mechanisms = controller.learning_mechanisms()
    if mode == "gain_search_only":
        mechanisms["gain_mutation"] = True
    if mode == "gain_search_recon_fast_bandit":
        mechanisms["gain_mutation"] = True
    return {
        "mode": mode,
        "n_poles": n_poles,
        "horizon": horizon,
        "seeds": seeds,
        "env_params": env_params,
        "train_episodes": train_episodes,
        "mechanisms": mechanisms,
        "steps": steps,
        "returns": returns,
        **summarize_steps(steps, horizon),
    }


def run_ablations(
    n_poles: int = 1,
    horizon: int = 500,
    seeds: list[int] | None = None,
    modes: list[str] | None = None,
    env_params: dict[str, Any] | None = None,
    train_episodes: int = 0,
    include_gain_search: bool = False,
) -> list[dict[str, Any]]:
    seeds = seeds or list(range(230_000, 230_020))
    selected_modes = list(modes or ABLATION_MODES)
    if include_gain_search:
        selected_modes.extend(GAIN_SEARCH_MODES)
    return [run_mode_on_seeds(mode, seeds, n_poles, horizon, env_params, train_episodes) for mode in selected_modes]


def write_ablation_report(results: list[dict[str, Any]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# ReCoN CartPole Ablations",
        "",
        "All rows use identical environment parameters and held-out seeds. Mechanisms are reported separately so gain-search performance is not mislabeled as ReCoN learning.",
        "",
        "| mode | mechanisms | mean survival | p10 survival | success rate | max survival |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        mechanisms = ", ".join(key for key, active in result.get("mechanisms", {}).items() if active) or "none"
        lines.append(
            f"| {result['mode']} | {mechanisms} | {result['mean_survival']:.1f} | {result['p10_survival']:.1f} | {result['success_rate']:.2f} | {result['max_survival']:.1f} |"
        )
    (out / "ablations.md").write_text("\n".join(lines), encoding="utf-8")
