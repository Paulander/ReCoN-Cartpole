from __future__ import annotations

import random
from typing import Literal

from .sensors import StateFeatures

ControllerMode = Literal[
    "baseline_random",
    "baseline_heuristic",
    "static_recon",
    "recon_fast",
    "recon_bandit",
    "recon_fast_bandit",
    "recon_slow",
    "gain_search_only",
    "gain_search_recon_fast_bandit",
    "recon_learn_only",
    "recon_slow_no_gain_search",
    "recon_mlp_terminal",
]


def heuristic_force(features: StateFeatures, force_mag: float = 10.0) -> float:
    force = 0.35 * features.x + 0.75 * features.x_dot
    for idx, pole in enumerate(features.poles):
        weight = 1.0 + 0.25 * idx
        force += weight * (18.0 * pole.theta + 3.8 * pole.theta_dot)
    return max(-force_mag, min(force_mag, force))


def random_action() -> int:
    return random.choice([0, 1])


def force_to_discrete(force: float) -> int:
    return 1 if force >= 0.0 else 0

