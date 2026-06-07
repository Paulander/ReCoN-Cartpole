from __future__ import annotations

from gymnasium.envs.registration import register


def register_envs() -> None:
    try:
        register(
            id="ReconCartPole/CartPoleN-v0",
            entry_point="recon_cartpole.envs:CartPoleNEnv",
        )
    except Exception:
        pass

