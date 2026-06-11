from __future__ import annotations

import math
from typing import Any

import numpy as np


def policy_observation_size(n_poles: int, mode: str) -> int:
    if mode == "normalized_raw":
        return 2 + 2 * int(n_poles)
    if mode == "normalized_raw_prev_force":
        return 3 + 2 * int(n_poles)
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
    if mode not in ("normalized_raw", "normalized_raw_prev_force"):
        return np.asarray(observation, dtype=np.float32).reshape(-1)
    raw = (
        np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw_state is not None
        else np.asarray([], dtype=np.float32)
    )
    needed = 2 + 2 * int(n_poles)
    if raw.size < needed:
        raise ValueError("normalized_raw policy observation requires raw_state")
    theta = raw[2 : 2 + n_poles]
    theta_dot = raw[2 + n_poles : 2 + 2 * n_poles]
    parts = [
        float(raw[0]) / max(float(x_threshold), 1e-9),
        float(raw[1]) / max(float(cart_velocity_scale), 1e-9),
    ]
    parts.extend((theta / max(float(theta_threshold), 1e-9)).tolist())
    parts.extend((theta_dot / max(float(pole_velocity_scale), 1e-9)).tolist())
    if mode == "normalized_raw_prev_force":
        parts.append(float(previous_force) / max(float(force_mag), 1e-9))
    return np.asarray(parts, dtype=np.float32)
