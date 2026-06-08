from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recon_cartpole.control.sensors import StateFeatures


@dataclass
class MlpTerminalConfig:
    enabled: bool = False
    hidden_size: int = 16
    eta: float = 0.08
    eta_tick: float = 0.01
    sigma: float = 0.08
    blend: float = 0.35
    baseline_beta: float = 0.9
    max_update_norm: float = 0.5


@dataclass
class MlpTerminalState:
    input_size: int
    hidden_size: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    episode_noise: dict[str, Any] = field(default_factory=dict)
    last_features: list[float] = field(default_factory=list)
    last_hidden: list[float] = field(default_factory=list)
    last_force: float = 0.0
    baseline_return: float = 0.0
    episodes: int = 0
    last_update: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(cls, n_poles: int, hidden_size: int, seed: int = 17) -> "MlpTerminalState":
        input_size = 2 + 2 * n_poles + 1
        rng = np.random.default_rng(seed + n_poles * 1009 + hidden_size)
        w1 = rng.normal(0.0, 0.08, size=(hidden_size, input_size))
        b1 = np.zeros(hidden_size, dtype=float)
        w2 = np.zeros(hidden_size, dtype=float)
        return cls(input_size=input_size, hidden_size=hidden_size, w1=w1, b1=b1, w2=w2, b2=0.0)

    def features(self, state: StateFeatures) -> np.ndarray:
        raw = [state.x, state.x_dot]
        raw.extend(pole.theta for pole in state.poles)
        raw.extend(pole.theta_dot for pole in state.poles)
        raw.append(1.0)
        values = np.asarray(raw, dtype=float)
        scale = np.asarray([2.4, 5.0] + [0.21] * len(state.poles) + [5.0] * len(state.poles) + [1.0], dtype=float)
        return np.clip(values / scale, -3.0, 3.0)

    def force(self, state: StateFeatures, force_mag: float) -> tuple[float, dict[str, Any]]:
        x = self.features(state)
        hidden = np.tanh(self.w1 @ x + self.b1)
        out = float(np.tanh(float(self.w2 @ hidden + self.b2)) * force_mag)
        self.last_features = x.tolist()
        self.last_hidden = hidden.tolist()
        self.last_force = out
        return out, {
            "input": [round(v, 5) for v in x.tolist()],
            "hidden_size": self.hidden_size,
            "raw_correction": out,
            "tick_update_ready": bool(self.last_features),
            "baseline_return": self.baseline_return,
            "episodes": self.episodes,
            "last_update": dict(self.last_update),
        }

    def start_episode(self, config: MlpTerminalConfig, rng: np.random.Generator, learn: bool) -> None:
        self.last_update = {}
        if not config.enabled or not learn or config.sigma <= 0.0:
            self.episode_noise = {}
            return
        self.episode_noise = {
            "w1": rng.normal(0.0, config.sigma, size=self.w1.shape),
            "b1": rng.normal(0.0, config.sigma, size=self.b1.shape),
            "w2": rng.normal(0.0, config.sigma, size=self.w2.shape),
            "b2": float(rng.normal(0.0, config.sigma)),
        }
        self.w1 += self.episode_noise["w1"]
        self.b1 += self.episode_noise["b1"]
        self.w2 += self.episode_noise["w2"]
        self.b2 += self.episode_noise["b2"]



    def update_from_tick(self, reward: float, force_mag: float, config: MlpTerminalConfig) -> dict[str, float]:
        if not config.enabled or config.eta_tick <= 0.0 or not self.last_features or not self.last_hidden:
            return {}
        reward = float(np.clip(reward, -1.0, 1.0))
        if abs(reward) < 1e-9:
            return {}
        x = np.asarray(self.last_features, dtype=float)
        hidden = np.asarray(self.last_hidden, dtype=float)
        z = float(np.clip(self.last_force / max(force_mag, 1e-9), -0.999, 0.999))
        # Reinforce the emitted force when reward is positive; reduce it when reward is negative.
        grad_out = reward * np.sign(self.last_force if abs(self.last_force) > 1e-9 else 1.0) * (1.0 - z * z)
        step = config.eta_tick * grad_out
        dw2 = step * hidden
        db2 = step
        dhidden = step * self.w2 * (1.0 - hidden * hidden)
        dw1 = np.outer(dhidden, x)
        db1 = dhidden
        norm = float(np.sqrt(np.sum(dw2 ** 2) + db2 * db2 + np.sum(dw1 ** 2) + np.sum(db1 ** 2)))
        if norm > config.max_update_norm > 0.0:
            shrink = config.max_update_norm / max(norm, 1e-12)
            dw2 *= shrink
            db2 *= shrink
            dw1 *= shrink
            db1 *= shrink
            norm = config.max_update_norm
        self.w2 += dw2
        self.b2 += float(db2)
        self.w1 += dw1
        self.b1 += db1
        self.last_update = {"tick_reward": reward, "tick_update_norm": norm}
        return dict(self.last_update)

    def end_episode(self, total_return: float, horizon: int, config: MlpTerminalConfig) -> dict[str, float]:
        if self.episode_noise:
            self.w1 -= self.episode_noise["w1"]
            self.b1 -= self.episode_noise["b1"]
            self.w2 -= self.episode_noise["w2"]
            self.b2 -= self.episode_noise["b2"]
        outcome = float(total_return) / max(1, horizon)
        advantage = outcome - self.baseline_return
        self.baseline_return = config.baseline_beta * self.baseline_return + (1.0 - config.baseline_beta) * outcome
        self.episodes += 1
        if not config.enabled or not self.episode_noise or config.sigma <= 0.0:
            self.episode_noise = {}
            self.last_update = {}
            return {}
        scale = config.eta * advantage / config.sigma
        updates = {
            "w1": scale * self.episode_noise["w1"],
            "b1": scale * self.episode_noise["b1"],
            "w2": scale * self.episode_noise["w2"],
            "b2": scale * self.episode_noise["b2"],
        }
        norm = float(np.sqrt(sum(np.sum(np.asarray(value) ** 2) for value in updates.values())))
        if norm > config.max_update_norm > 0.0:
            shrink = config.max_update_norm / max(norm, 1e-12)
            updates = {key: value * shrink for key, value in updates.items()}
            norm = config.max_update_norm
        self.w1 += updates["w1"]
        self.b1 += updates["b1"]
        self.w2 += updates["w2"]
        self.b2 += float(updates["b2"])
        self.episode_noise = {}
        self.last_update = {"advantage": float(advantage), "update_norm": float(norm), "outcome": outcome}
        return dict(self.last_update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2,
            "baseline_return": self.baseline_return,
            "episodes": self.episodes,
            "last_update": dict(self.last_update),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MlpTerminalState":
        return cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            w1=np.asarray(data["w1"], dtype=float),
            b1=np.asarray(data["b1"], dtype=float),
            w2=np.asarray(data["w2"], dtype=float),
            b2=float(data.get("b2", 0.0)),
            baseline_return=float(data.get("baseline_return", 0.0)),
            episodes=int(data.get("episodes", 0)),
            last_update={key: float(value) for key, value in data.get("last_update", {}).items()},
        )
