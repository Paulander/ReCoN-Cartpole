from __future__ import annotations

from typing import Any

from .sensors import StateFeatures, features_from_state


def compute_cartpole_goal_vector(
    state: Any,
    limits: dict[str, float] | None = None,
    n_poles: int = 1,
    raw_state: Any | None = None,
    stage: str = "default",
) -> dict[str, float | int | str]:
    limits = limits or {
        "x_threshold": 2.4,
        "theta_threshold_radians": 12.0 * 2.0 * 3.141592653589793 / 360.0,
        "max_angular_velocity": 4.0,
    }
    features = state if isinstance(state, StateFeatures) else features_from_state(state, raw_state, n_poles)
    cart_centering_need = min(1.0, abs(features.x) / limits["x_threshold"])
    max_angle_pressure = min(1.0, features.max_angle_abs / limits["theta_threshold_radians"])
    max_velocity_pressure = min(1.0, features.max_angular_velocity_abs / limits["max_angular_velocity"])
    falling_pressure = max(
        0.0,
        max((abs(p.theta) * abs(p.theta_dot) for p in features.poles), default=0.0) / 1.5,
    )
    risk = min(
        1.0,
        0.42 * cart_centering_need
        + 0.40 * max_angle_pressure
        + 0.12 * max_velocity_pressure
        + 0.06 * falling_pressure,
    )
    urgency = min(
        1.0,
        0.50 * falling_pressure
        + 0.30 * max_angle_pressure
        + 0.20 * cart_centering_need,
    )
    return {
        "risk": risk,
        "urgency": urgency,
        "cart_centering_need": cart_centering_need,
        "max_angle_pressure": max_angle_pressure,
        "max_velocity_pressure": max_velocity_pressure,
        "pole_count": n_poles,
        "stage": stage,
    }

