from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .controllers import heuristic_force
from .sensors import StateFeatures


@dataclass
class ForceProposal:
    source_node: str
    force: float
    confidence: float
    urgency: float
    reason: str
    score: float = 0.0
    raw_confidence: float = 0.0
    raw_urgency: float = 0.0
    select_edge_weight: float = 1.0
    proposal_edge_weight: float = 1.0
    bandit_score: float = 1.0
    selection_multiplier: float = 1.0
    selected: bool = True
    suppressed: bool = False
    selection_mode: str = "soft_select"

    def __post_init__(self) -> None:
        if self.raw_confidence == 0.0:
            self.raw_confidence = self.confidence
        if self.raw_urgency == 0.0:
            self.raw_urgency = self.urgency
        if self.score == 0.0 and not self.suppressed:
            self.score = max(0.01, self.confidence) * (1.0 + self.urgency)


@dataclass
class ProposalGains:
    angle_gain: float = 18.0
    velocity_gain: float = 3.8
    cart_position_gain: float = 0.75
    cart_velocity_gain: float = 0.25
    worst_velocity_mix: float = 0.30
    base_velocity_mix: float = 0.35
    outer_link_weight: float = 0.25
    center_confidence: float = 0.45
    chain_confidence: float = 0.65
    recover_confidence: float = 0.85

    def clipped(self) -> "ProposalGains":
        bounds = gain_bounds()
        values = asdict(self)
        return ProposalGains(**{key: _clip(value, *bounds[key]) for key, value in values.items()})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProposalGains":
        if not data:
            return cls()
        valid = {key: float(value) for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**valid).clipped()


REGIMES = [
    "avoid_rail",
    "damp_energy",
    "recover_worst_pole",
    "recover_base_pole",
    "stabilize_chain",
    "center_cart",
]


def gain_bounds() -> dict[str, tuple[float, float]]:
    return {
        "angle_gain": (4.0, 40.0),
        "velocity_gain": (0.5, 10.0),
        "cart_position_gain": (0.05, 3.0),
        "cart_velocity_gain": (0.02, 2.0),
        "worst_velocity_mix": (0.0, 1.5),
        "base_velocity_mix": (0.0, 1.5),
        "outer_link_weight": (0.0, 1.5),
        "center_confidence": (0.05, 1.0),
        "chain_confidence": (0.05, 1.0),
        "recover_confidence": (0.05, 1.0),
    }


def propose_force_for_regime(
    regime: str,
    features: StateFeatures,
    force_mag: float = 10.0,
    gains: ProposalGains | None = None,
) -> ForceProposal:
    gains = gains or ProposalGains()
    if regime == "avoid_rail":
        force = -force_mag if features.x > 0 else force_mag
        confidence = min(1.0, abs(features.x) / 2.4)
        return ForceProposal(regime, force, confidence, confidence, "rail margin")
    if regime == "center_cart":
        force = -force_mag * (gains.cart_position_gain * features.x + gains.cart_velocity_gain * features.x_dot)
        return ForceProposal(
            regime,
            _clip(force, -force_mag, force_mag),
            gains.center_confidence,
            0.35,
            "cart centering",
        )
    if regime == "damp_energy":
        worst = features.poles[features.worst_pole_index]
        force = force_mag if worst.theta_dot >= 0 else -force_mag
        return ForceProposal(regime, force, 0.40, min(1.0, abs(worst.theta_dot) / 4.0), "energy damping")
    if regime == "recover_base_pole":
        pole = features.poles[0]
        force = force_mag if pole.theta + gains.base_velocity_mix * pole.theta_dot >= 0 else -force_mag
        return ForceProposal(
            regime,
            force,
            0.80,
            min(1.0, pole.angle_abs / 0.21),
            "base pole recovery",
        )
    if regime == "stabilize_chain":
        force = _chain_force(features, force_mag, gains)
        return ForceProposal(
            regime,
            force,
            gains.chain_confidence,
            min(1.0, features.max_angle_abs / 0.21),
            "chain gain blend",
        )
    worst = features.poles[features.worst_pole_index]
    force = force_mag if worst.theta + gains.worst_velocity_mix * worst.theta_dot >= 0 else -force_mag
    return ForceProposal(
        regime,
        force,
        gains.recover_confidence,
        min(1.0, worst.angle_abs / 0.21),
        "worst pole recovery",
    )


def _chain_force(features: StateFeatures, force_mag: float, gains: ProposalGains) -> float:
    force = gains.cart_position_gain * features.x + gains.cart_velocity_gain * features.x_dot
    for idx, pole in enumerate(features.poles):
        weight = 1.0 + gains.outer_link_weight * idx
        force += weight * (gains.angle_gain * pole.theta + gains.velocity_gain * pole.theta_dot)
    if abs(force) < 1e-9:
        return heuristic_force(features, force_mag)
    return _clip(force, -force_mag, force_mag)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
