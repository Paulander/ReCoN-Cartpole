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
    "recon_policy_terminal",
    "recon_recurrent_policy_terminal",
    "recon_feedforward_terminal_frozen",
    "recon_feedforward_terminal_plus_recon_learning",
    "recon_feedforward_terminal_with_pole1_fix",
    "recon_subchain_terminal",
    "recon_mingru_terminal",
    "recon_mingru_terminal_plus_recon_learning",
]


def heuristic_force(features: StateFeatures, force_mag: float = 10.0) -> float:
    force = 0.35 * features.x + 0.75 * features.x_dot
    for idx, pole in enumerate(features.poles):
        weight = 1.0 + 0.25 * idx
        force += weight * (18.0 * pole.theta + 3.8 * pole.theta_dot)
    return max(-force_mag, min(force_mag, force))


def random_action(discrete_action_bins: int = 2) -> int:
    return random.randrange(max(2, int(discrete_action_bins)))


def force_to_discrete(force: float, force_mag: float = 10.0, discrete_action_bins: int = 2) -> int:
    bins = max(2, int(discrete_action_bins))
    if bins == 2:
        return 1 if force >= 0.0 else 0
    low = -float(force_mag)
    high = float(force_mag)
    if high <= low:
        return 0
    scaled = (float(force) - low) / (high - low)
    return max(0, min(bins - 1, int(round(scaled * (bins - 1)))))

