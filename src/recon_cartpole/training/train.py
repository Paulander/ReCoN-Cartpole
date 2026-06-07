from __future__ import annotations

from typing import Any

from .evaluate import evaluate


def train_block(config: dict[str, Any]) -> dict[str, Any]:
    # Fast plasticity and bandit modes learn online during rollout. This block is
    # intentionally small until slow consolidation promotion rules are added.
    return evaluate(
        env_name=config.get("env", "cartpole_n"),
        mode=config.get("mode", "recon_fast_bandit"),
        n_poles=int(config.get("n_poles", 1)),
        episodes=int(config.get("episodes", 20)),
        horizon=int(config.get("horizon", 500)),
        seed=int(config.get("seed", 0)),
    )

