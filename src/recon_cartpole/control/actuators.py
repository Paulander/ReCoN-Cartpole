from __future__ import annotations

from .controllers import force_to_discrete


def clamp_force(force: float, force_mag: float) -> float:
    return max(-force_mag, min(force_mag, force))


def action_from_force(force: float, action_mode: str = "discrete"):
    if action_mode == "continuous":
        return [force]
    return force_to_discrete(force)

