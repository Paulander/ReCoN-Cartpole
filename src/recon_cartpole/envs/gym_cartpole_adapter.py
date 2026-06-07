from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


class GymCartPoleAdapter:
    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.last_info: dict[str, Any] = {}

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def raw_state(self) -> np.ndarray:
        return np.asarray(getattr(self.env.unwrapped, "state", np.zeros(4)), dtype=float)

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        self.last_info = {**info, "raw_state": self.raw_state}
        return obs, self.last_info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.last_info = {**info, "raw_state": self.raw_state}
        return obs, reward, terminated, truncated, self.last_info

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

