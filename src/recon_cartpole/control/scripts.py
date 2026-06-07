from __future__ import annotations

from dataclasses import dataclass

from .controllers import heuristic_force
from .sensors import StateFeatures


@dataclass
class ForceProposal:
    source_node: str
    force: float
    confidence: float
    urgency: float
    reason: str


REGIMES = [
    "avoid_rail",
    "damp_energy",
    "recover_worst_pole",
    "recover_base_pole",
    "stabilize_chain",
    "center_cart",
]


def propose_force_for_regime(regime: str, features: StateFeatures, force_mag: float = 10.0) -> ForceProposal:
    if regime == "avoid_rail":
        force = -force_mag if features.x > 0 else force_mag
        confidence = min(1.0, abs(features.x) / 2.4)
        return ForceProposal(regime, force, confidence, confidence, "rail margin")
    if regime == "center_cart":
        force = -force_mag * (0.75 * features.x + 0.25 * features.x_dot)
        return ForceProposal(regime, max(-force_mag, min(force_mag, force)), 0.45, 0.35, "cart centering")
    if regime == "damp_energy":
        worst = features.poles[features.worst_pole_index]
        force = force_mag if worst.theta_dot >= 0 else -force_mag
        return ForceProposal(regime, force, 0.40, min(1.0, abs(worst.theta_dot) / 4.0), "energy damping")
    if regime == "recover_base_pole":
        pole = features.poles[0]
        force = force_mag if pole.theta + 0.35 * pole.theta_dot >= 0 else -force_mag
        return ForceProposal(regime, force, 0.80, min(1.0, pole.angle_abs / 0.21), "base pole recovery")
    if regime == "stabilize_chain":
        force = heuristic_force(features, force_mag)
        return ForceProposal(regime, force, 0.65, min(1.0, features.max_angle_abs / 0.21), "chain heuristic")
    worst = features.poles[features.worst_pole_index]
    force = force_mag if worst.theta + 0.30 * worst.theta_dot >= 0 else -force_mag
    return ForceProposal(regime, force, 0.85, min(1.0, worst.angle_abs / 0.21), "worst pole recovery")

