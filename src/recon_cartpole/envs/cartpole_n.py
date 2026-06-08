from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class CartPoleNConfig:
    n_poles: int = 1
    action_mode: str = "discrete"
    horizon: int = 500
    dt: float = 0.02
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_masses: list[float] = field(default_factory=list)
    pole_lengths: list[float] = field(default_factory=list)
    damping: float = 0.01
    force_mag: float = 10.0
    force_noise: float = 0.0
    x_threshold: float = 2.4
    theta_threshold_radians: float = 12.0 * 2.0 * math.pi / 360.0
    initial_angle_range: float = 0.05
    seed: Optional[int] = None

    def masses(self) -> np.ndarray:
        return np.asarray(self.pole_masses or [0.1] * self.n_poles, dtype=float)

    def lengths(self) -> np.ndarray:
        return np.asarray(self.pole_lengths or [0.5] * self.n_poles, dtype=float)


class CartPoleNEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: Optional[CartPoleNConfig] = None,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = CartPoleNConfig(**kwargs)
        elif kwargs:
            config = CartPoleNConfig(**{**config.__dict__, **kwargs})
        if config.n_poles < 1:
            raise ValueError("n_poles must be >= 1")
        self.config = config
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(config.seed)
        self.state = np.zeros(2 + 2 * config.n_poles, dtype=float)
        obs_size = 2 + 3 * config.n_poles
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        if config.action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.asarray([-config.force_mag], dtype=np.float32),
                high=np.asarray([config.force_mag], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(2)
        self.steps = 0
        self._screen = None
        self._clock = None

    @property
    def raw_state(self) -> np.ndarray:
        return self.state.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        high = self.config.initial_angle_range
        self.state = self.np_random.uniform(low=-high, high=high, size=self.state.shape)
        self.state[0] = self.np_random.uniform(-0.05, 0.05)
        self.state[1] = self.np_random.uniform(-0.02, 0.02)
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action: Any):
        force = self._force_from_action(action)
        if self.config.force_noise > 0.0:
            noise = self.np_random.normal(0.0, self.config.force_noise * self.config.force_mag)
            force = float(np.clip(force + noise, -self.config.force_mag, self.config.force_mag))
        self._integrate(force)
        self.steps += 1
        terminated = self._terminated()
        truncated = self.steps >= self.config.horizon
        reward = 1.0 if not terminated else 0.0
        return self._get_obs(), reward, terminated, truncated, self._get_info(force)

    def _force_from_action(self, action: Any) -> float:
        if self.config.action_mode == "continuous":
            return float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.config.force_mag, self.config.force_mag))
        return self.config.force_mag if int(action) == 1 else -self.config.force_mag

    def _integrate(self, force: float) -> None:
        if self.config.n_poles == 1:
            x_acc, theta_acc = self._single_pole_accelerations(force)
            theta_accs = np.asarray([theta_acc])
        else:
            x_acc, theta_accs = self._coupled_accelerations(force)

        dt = self.config.dt
        n = self.config.n_poles
        self.state[1] += dt * x_acc
        self.state[0] += dt * self.state[1]
        self.state[2 + n :] += dt * theta_accs
        self.state[2 : 2 + n] += dt * self.state[2 + n :]
        self.state[2 : 2 + n] = (self.state[2 : 2 + n] + math.pi) % (2 * math.pi) - math.pi

    def _single_pole_accelerations(self, force: float) -> tuple[float, float]:
        cfg = self.config
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]
        mass_pole = cfg.masses()[0]
        length = cfg.lengths()[0]
        total_mass = cfg.cart_mass + mass_pole
        polemass_length = mass_pole * length
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sintheta - cfg.damping * x_dot) / total_mass
        theta_acc = (
            cfg.gravity * sintheta
            - costheta * temp
            - cfg.damping * theta_dot / max(mass_pole * length, 1e-9)
        ) / (length * (4.0 / 3.0 - mass_pole * costheta**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass
        return x_acc, theta_acc

    def _coupled_accelerations(self, force: float) -> tuple[float, np.ndarray]:
        cfg = self.config
        n = cfg.n_poles
        theta = self.state[2 : 2 + n]
        theta_dot = self.state[2 + n :]
        masses = cfg.masses()
        lengths = cfg.lengths()
        total_mass = cfg.cart_mass + float(np.sum(masses))
        base_acc = (force - cfg.damping * self.state[1]) / total_mass
        theta_acc = np.zeros(n, dtype=float)
        for i in range(n):
            neighbor = 0.0
            if i > 0:
                neighbor += theta[i - 1] - theta[i]
            if i + 1 < n:
                neighbor += theta[i + 1] - theta[i]
            theta_acc[i] = (
                cfg.gravity * math.sin(theta[i])
                - math.cos(theta[i]) * base_acc
                + 0.35 * neighbor
                - cfg.damping * theta_dot[i] / max(masses[i] * lengths[i], 1e-9)
            ) / (lengths[i] * 4.0 / 3.0)
        x_acc = (
            force
            + float(np.sum(masses * lengths * (theta_dot**2 * np.sin(theta) - theta_acc * np.cos(theta))))
            - cfg.damping * self.state[1]
        ) / total_mass
        return x_acc, theta_acc

    def _get_obs(self) -> np.ndarray:
        n = self.config.n_poles
        theta = self.state[2 : 2 + n]
        theta_dot = self.state[2 + n :]
        parts: list[float] = [self.state[0], self.state[1]]
        for angle, velocity in zip(theta, theta_dot):
            parts.extend([math.sin(angle), math.cos(angle), velocity])
        return np.asarray(parts, dtype=np.float32)

    def _get_info(self, force: float = 0.0) -> dict[str, Any]:
        return {
            "raw_state": self.raw_state,
            "steps": self.steps,
            "force": force,
            "energy": self.energy(),
        }

    def _terminated(self) -> bool:
        n = self.config.n_poles
        angles = np.abs(self.state[2 : 2 + n])
        return bool(
            abs(self.state[0]) > self.config.x_threshold
            or np.any(angles > self.config.theta_threshold_radians)
        )

    def energy(self) -> float:
        n = self.config.n_poles
        theta = self.state[2 : 2 + n]
        theta_dot = self.state[2 + n :]
        masses = self.config.masses()
        lengths = self.config.lengths()
        cart_energy = 0.5 * self.config.cart_mass * self.state[1] ** 2
        kinetic = 0.5 * masses * (lengths * theta_dot) ** 2
        potential = masses * self.config.gravity * lengths * (1.0 - np.cos(theta))
        return float(cart_energy + np.sum(kinetic + potential))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            self._render_rgb_array()
            return None
        return None

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        width, height = 800, 420
        if self._screen is None:
            pygame.init()
            self._screen = pygame.Surface((width, height))
        surf = self._screen
        surf.fill((248, 250, 252))
        rail_y = height - 95
        scale = width / (self.config.x_threshold * 2.4)
        cart_x = int(width / 2 + self.state[0] * scale)
        pygame.draw.line(surf, (30, 41, 59), (40, rail_y), (width - 40, rail_y), 3)
        pygame.draw.rect(surf, (37, 99, 235), pygame.Rect(cart_x - 42, rail_y - 24, 84, 40), border_radius=4)
        origin = np.asarray([cart_x, rail_y - 24], dtype=float)
        theta = self.state[2 : 2 + self.config.n_poles]
        lengths = self.config.lengths()
        for i, angle in enumerate(theta):
            rod_len = max(34.0, lengths[i] * 220.0)
            end = origin + np.asarray([math.sin(angle) * rod_len, -math.cos(angle) * rod_len])
            pygame.draw.line(surf, (15, 23, 42), origin, end, 7)
            pygame.draw.circle(surf, (239, 68, 68), end.astype(int), 8)
            pygame.draw.circle(surf, (2, 132, 199), origin.astype(int), 7)
            origin = end
        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2))

