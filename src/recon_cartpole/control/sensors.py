from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PoleFeatures:
    theta: float
    theta_dot: float
    angle_abs: float
    falling_direction: float
    energy: float


@dataclass
class StateFeatures:
    x: float
    x_dot: float
    poles: list[PoleFeatures]
    max_angle_abs: float
    max_angular_velocity_abs: float
    worst_pole_index: int


def features_from_state(
    observation: Any,
    raw_state: Any | None = None,
    n_poles: int | None = None,
    gravity: float = 9.8,
    lengths: list[float] | None = None,
) -> StateFeatures:
    obs = np.asarray(observation, dtype=float)
    if raw_state is not None:
        raw = np.asarray(raw_state, dtype=float)
        if n_poles is None:
            n_poles = max(1, (raw.size - 2) // 2)
        x = float(raw[0])
        x_dot = float(raw[1])
        theta = raw[2 : 2 + n_poles]
        theta_dot = raw[2 + n_poles : 2 + 2 * n_poles]
    elif obs.size == 4:
        n_poles = 1
        x, x_dot, angle, velocity = [float(v) for v in obs]
        theta = np.asarray([angle])
        theta_dot = np.asarray([velocity])
    else:
        n_poles = n_poles or max(1, (obs.size - 2) // 3)
        x = float(obs[0])
        x_dot = float(obs[1])
        theta = np.asarray([math.atan2(obs[2 + 3 * i], obs[3 + 3 * i]) for i in range(n_poles)])
        theta_dot = np.asarray([obs[4 + 3 * i] for i in range(n_poles)])

    lengths = lengths or [0.5] * int(n_poles)
    poles: list[PoleFeatures] = []
    for i in range(int(n_poles)):
        angle = float(theta[i])
        velocity = float(theta_dot[i])
        energy = 0.5 * (lengths[i] * velocity) ** 2 + gravity * lengths[i] * (1.0 - math.cos(angle))
        poles.append(
            PoleFeatures(
                theta=angle,
                theta_dot=velocity,
                angle_abs=abs(angle),
                falling_direction=math.copysign(1.0, angle * velocity) if angle * velocity else 0.0,
                energy=energy,
            )
        )

    worst = max(range(len(poles)), key=lambda idx: poles[idx].angle_abs + 0.15 * abs(poles[idx].theta_dot))
    return StateFeatures(
        x=x,
        x_dot=x_dot,
        poles=poles,
        max_angle_abs=max(p.angle_abs for p in poles),
        max_angular_velocity_abs=max(abs(p.theta_dot) for p in poles),
        worst_pole_index=worst,
    )

