from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from recon_cartpole.control.scripts import ForceProposal, REGIMES, _clip
from recon_cartpole.control.sensors import StateFeatures


@dataclass
class RegimeNodeParams:
    confidence_multiplier: float = 1.0
    urgency_multiplier: float = 1.0
    force_bias: float = 0.0
    angle_weight: float = 1.0
    velocity_weight: float = 1.0
    pole_priority_weight: float = 1.0
    cart_centering_weight: float = 1.0

    def clipped(self) -> "RegimeNodeParams":
        return RegimeNodeParams(
            confidence_multiplier=_clip(self.confidence_multiplier, 0.1, 3.0),
            urgency_multiplier=_clip(self.urgency_multiplier, 0.1, 3.0),
            force_bias=_clip(self.force_bias, -5.0, 5.0),
            angle_weight=_clip(self.angle_weight, 0.1, 3.0),
            velocity_weight=_clip(self.velocity_weight, 0.1, 3.0),
            pole_priority_weight=_clip(self.pole_priority_weight, 0.1, 3.0),
            cart_centering_weight=_clip(self.cart_centering_weight, 0.1, 3.0),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RegimeNodeParams":
        if not data:
            return cls()
        valid = {key: float(value) for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**valid).clipped()


@dataclass
class RegimeParamState:
    base: RegimeNodeParams = field(default_factory=RegimeNodeParams)
    current: RegimeNodeParams = field(default_factory=RegimeNodeParams)
    eligibility: float = 0.0
    delta_sum: dict[str, float] = field(default_factory=dict)

    def reset_episode(self) -> None:
        self.current = RegimeNodeParams.from_dict(self.base.to_dict())
        self.eligibility = 0.0
        self.delta_sum = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "current": self.current.to_dict(),
            "eligibility": round(self.eligibility, 4),
            "delta_sum": {key: round(value, 5) for key, value in self.delta_sum.items() if abs(value) > 1e-12},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegimeParamState":
        return cls(
            base=RegimeNodeParams.from_dict(data.get("base")),
            current=RegimeNodeParams.from_dict(data.get("current") or data.get("base")),
            eligibility=float(data.get("eligibility", 0.0)),
            delta_sum={key: float(value) for key, value in data.get("delta_sum", {}).items()},
        )


@dataclass
class NodeParamConfig:
    enabled: bool = False
    eta_fast: float = 0.01
    eta_consolidate: float = 0.05
    lambda_decay: float = 0.85
    min_episodes: int = 20
    max_delta_episode: float = 0.35


def init_regime_param_state() -> dict[str, RegimeParamState]:
    return {regime: RegimeParamState() for regime in REGIMES}


def apply_node_params(proposal: ForceProposal, params: RegimeNodeParams, features: StateFeatures, force_mag: float) -> ForceProposal:
    regime = proposal.source_node
    force = proposal.force + params.force_bias
    if regime in ("recover_worst_pole", "recover_base_pole"):
        idx = 0 if regime == "recover_base_pole" else features.worst_pole_index
        pole = features.poles[idx]
        signal = params.angle_weight * pole.theta + params.velocity_weight * pole.theta_dot
        force += 0.15 * force_mag * signal
    elif regime == "stabilize_chain":
        force = params.pole_priority_weight * force + params.cart_centering_weight * 0.15 * force_mag * features.x
    elif regime == "center_cart":
        force *= params.cart_centering_weight
    proposal.force = _clip(force, -force_mag, force_mag)
    proposal.confidence *= params.confidence_multiplier
    proposal.urgency *= params.urgency_multiplier
    proposal.reason = f"{proposal.reason}; node_params"
    return proposal


def update_regime_param_state(
    state: dict[str, RegimeParamState],
    regime: str | None,
    reward: float,
    force: float,
    force_mag: float,
    config: NodeParamConfig,
) -> dict[str, float]:
    if not config.enabled or not regime or regime not in state:
        return {}
    item = state[regime]
    item.eligibility = item.eligibility * config.lambda_decay + 1.0
    reward = max(-1.0, min(1.0, float(reward)))
    direction = 1.0 if force >= 0.0 else -1.0
    updates = {
        "confidence_multiplier": config.eta_fast * reward * item.eligibility,
        "urgency_multiplier": 0.5 * config.eta_fast * reward * item.eligibility,
        "force_bias": config.eta_fast * reward * direction * force_mag * 0.15 * item.eligibility,
        "angle_weight": 0.25 * config.eta_fast * reward * item.eligibility,
        "velocity_weight": 0.25 * config.eta_fast * reward * item.eligibility,
        "pole_priority_weight": 0.15 * config.eta_fast * reward * item.eligibility,
        "cart_centering_weight": 0.15 * config.eta_fast * reward * item.eligibility,
    }
    current = item.current.to_dict()
    base = item.base.to_dict()
    actual: dict[str, float] = {}
    for key, delta in updates.items():
        proposed = current[key] + delta
        proposed = max(base[key] - config.max_delta_episode, min(base[key] + config.max_delta_episode, proposed))
        next_params = RegimeNodeParams.from_dict({**current, key: proposed}).to_dict()
        actual_delta = next_params[key] - current[key]
        if abs(actual_delta) > 1e-12:
            current[key] = next_params[key]
            item.delta_sum[key] = item.delta_sum.get(key, 0.0) + actual_delta
            actual[key] = actual_delta
    item.current = RegimeNodeParams.from_dict(current)
    return actual


def consolidate_regime_params(
    state: dict[str, RegimeParamState],
    outcome_score: float,
    config: NodeParamConfig,
) -> dict[str, dict[str, float]]:
    if not config.enabled:
        return {}
    applied: dict[str, dict[str, float]] = {}
    outcome = max(-1.0, min(1.0, float(outcome_score)))
    for regime, item in state.items():
        if not item.delta_sum:
            continue
        base = item.base.to_dict()
        regime_applied: dict[str, float] = {}
        for key, delta_sum in item.delta_sum.items():
            delta = config.eta_consolidate * outcome * delta_sum
            next_params = RegimeNodeParams.from_dict({**base, key: base[key] + delta}).to_dict()
            actual = next_params[key] - base[key]
            if abs(actual) > 1e-12:
                base[key] = next_params[key]
                regime_applied[key] = actual
        if regime_applied:
            item.base = RegimeNodeParams.from_dict(base)
            item.current = RegimeNodeParams.from_dict(base)
            applied[regime] = regime_applied
        item.delta_sum = {}
        item.eligibility = 0.0
    return applied


def snapshot_regime_params(state: dict[str, RegimeParamState]) -> dict[str, Any]:
    return {regime: item.to_dict() for regime, item in state.items()}
