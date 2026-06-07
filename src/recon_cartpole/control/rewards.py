from __future__ import annotations

from typing import Any

from .sensors import StateFeatures, features_from_state


def stability_score(
    observation: Any,
    raw_state: Any | None = None,
    n_poles: int = 1,
    terminated: bool = False,
) -> float:
    features = observation if isinstance(observation, StateFeatures) else features_from_state(observation, raw_state, n_poles)
    angle_penalty = sum((idx + 1) * pole.angle_abs for idx, pole in enumerate(features.poles))
    velocity_penalty = sum((idx + 1) * abs(pole.theta_dot) for idx, pole in enumerate(features.poles))
    score = (
        1.0
        - 0.18 * abs(features.x)
        - 0.04 * abs(features.x_dot)
        - 1.8 * angle_penalty / max(1, len(features.poles))
        - 0.08 * velocity_penalty / max(1, len(features.poles))
    )
    if terminated:
        score -= 2.0
    return float(score)


def reward_tick(
    before_observation: Any,
    after_observation: Any,
    before_raw_state: Any | None = None,
    after_raw_state: Any | None = None,
    n_poles: int = 1,
    terminated: bool = False,
) -> float:
    before = stability_score(before_observation, before_raw_state, n_poles)
    after = stability_score(after_observation, after_raw_state, n_poles, terminated)
    return after - before

