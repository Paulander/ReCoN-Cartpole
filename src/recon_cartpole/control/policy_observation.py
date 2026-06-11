from __future__ import annotations

import math
from typing import Any

import numpy as np


def _padded_pole_count(mode: str, n_poles: int) -> int:
    if mode in ("normalized_raw4", "normalized_raw4_prev_force"):
        return 4
    return int(n_poles)


def policy_observation_size(n_poles: int, mode: str) -> int:
    padded_n = _padded_pole_count(mode, n_poles)
    if mode in ("normalized_raw", "normalized_raw4"):
        return 2 + 2 * padded_n
    if mode in ("normalized_raw_prev_force", "normalized_raw4_prev_force"):
        return 3 + 2 * padded_n
    return 2 + 3 * int(n_poles)


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
    parts.extend((theta / max(float(theta_threshold), 1e-9)).tolist())
    parts.extend((theta_dot / max(float(pole_velocity_scale), 1e-9)).tolist())
    if mode in ("normalized_raw_prev_force", "normalized_raw4_prev_force"):
        parts.append(float(previous_force) / max(float(force_mag), 1e-9))
    return np.asarray(parts, dtype=np.float32)
