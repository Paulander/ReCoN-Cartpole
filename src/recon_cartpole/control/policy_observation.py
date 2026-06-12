from __future__ import annotations

import math
from typing import Any

import numpy as np


def _padded_pole_count(mode: str, n_poles: int) -> int:
    if mode in (
        "normalized_raw4",
        "normalized_raw4_prev_force",
        "normalized_raw4_subchains",
        "normalized_raw4_subchains_prev_force",
    ):
        return 4
    return int(n_poles)


def _has_prev_force(mode: str) -> bool:
    return mode in ("normalized_raw_prev_force", "normalized_raw4_prev_force", "normalized_raw4_subchains_prev_force")


def _has_subchains(mode: str) -> bool:
    return mode in ("normalized_raw4_subchains", "normalized_raw4_subchains_prev_force")


def subchain_feature_size(padded_n: int = 4) -> int:
    return max(0, int(padded_n) - 1) * 4


def policy_observation_size(n_poles: int, mode: str) -> int:
    padded_n = _padded_pole_count(mode, n_poles)
    if mode in ("normalized_raw", "normalized_raw4", "normalized_raw4_subchains"):
        return 2 + 2 * padded_n + (subchain_feature_size(padded_n) if _has_subchains(mode) else 0)
    if _has_prev_force(mode):
        return 3 + 2 * padded_n + (subchain_feature_size(padded_n) if _has_subchains(mode) else 0)
    return 2 + 3 * int(n_poles)


def adjacent_subchain_features(theta: np.ndarray, theta_dot: np.ndarray, padded_n: int) -> list[float]:
    features: list[float] = []
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    theta_dot = np.asarray(theta_dot, dtype=np.float32).reshape(-1)
    if theta.size < padded_n:
        theta = np.pad(theta, (0, padded_n - theta.size))
    if theta_dot.size < padded_n:
        theta_dot = np.pad(theta_dot, (0, padded_n - theta_dot.size))
    for idx in range(max(0, int(padded_n) - 1)):
        a = float(theta[idx])
        b = float(theta[idx + 1])
        av = float(theta_dot[idx])
        bv = float(theta_dot[idx + 1])
        features.extend([
            b - a,
            bv - av,
            0.5 * (a + b),
            0.5 * (av + bv),
        ])
    return features


def policy_observation_from_state(
    observation: Any,
    raw_state: Any | None,
    n_poles: int,
    mode: str = "env",
    x_threshold: float = 2.4,
    theta_threshold: float = 12.0 * 2.0 * math.pi / 360.0,
    cart_velocity_scale: float = 5.0,
    pole_velocity_scale: float = 5.0,
    previous_force: float = 0.0,
    force_mag: float = 10.0,
) -> np.ndarray:
    normalized_modes = (
        "normalized_raw",
        "normalized_raw_prev_force",
        "normalized_raw4",
        "normalized_raw4_prev_force",
        "normalized_raw4_subchains",
        "normalized_raw4_subchains_prev_force",
    )
    if mode not in normalized_modes:
        return np.asarray(observation, dtype=np.float32).reshape(-1)
    raw = (
        np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw_state is not None
        else np.asarray([], dtype=np.float32)
    )
    n_poles = int(n_poles)
    needed = 2 + 2 * n_poles
    if raw.size < needed:
        raise ValueError("normalized_raw policy observation requires raw_state")
    padded_n = _padded_pole_count(mode, n_poles)
    theta = raw[2 : 2 + n_poles]
    theta_dot = raw[2 + n_poles : 2 + 2 * n_poles]
    if padded_n > n_poles:
        theta = np.pad(theta, (0, padded_n - n_poles))
        theta_dot = np.pad(theta_dot, (0, padded_n - n_poles))
    parts = [
        float(raw[0]) / max(float(x_threshold), 1e-9),
        float(raw[1]) / max(float(cart_velocity_scale), 1e-9),
    ]
    theta_norm = theta / max(float(theta_threshold), 1e-9)
    theta_dot_norm = theta_dot / max(float(pole_velocity_scale), 1e-9)
    parts.extend(theta_norm.tolist())
    parts.extend(theta_dot_norm.tolist())
    if _has_subchains(mode):
        parts.extend(adjacent_subchain_features(theta_norm, theta_dot_norm, padded_n))
    if _has_prev_force(mode):
        parts.append(float(previous_force) / max(float(force_mag), 1e-9))
    return np.asarray(parts, dtype=np.float32)
