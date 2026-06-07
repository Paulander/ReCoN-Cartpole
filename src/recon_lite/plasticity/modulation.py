from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModulationConfig:
    alpha_risk: float = 0.75
    alpha_urgency: float = 0.75
    eta_base: float = 0.03
    c_explore_base: float = 1.0


@dataclass
class Modulators:
    risk: float
    urgency: float
    eta_tick_eff: float
    c_explore_eff: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "risk": round(self.risk, 4),
            "urgency": round(self.urgency, 4),
            "eta_tick_eff": round(self.eta_tick_eff, 4),
            "c_explore_eff": round(self.c_explore_eff, 4),
        }


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def compute_modulators(
    goal_vector: Optional[Dict[str, Any]],
    config: Optional[ModulationConfig] = None,
) -> Modulators:
    config = config or ModulationConfig()
    goal_vector = goal_vector or {}

    risk = _clamp01(
        0.45 * _clamp01(goal_vector.get("risk", 0.0))
        + 0.25 * _clamp01(goal_vector.get("max_angle_pressure", 0.0))
        + 0.20 * _clamp01(goal_vector.get("cart_centering_need", 0.0))
        + 0.10 * _clamp01(goal_vector.get("max_velocity_pressure", 0.0))
    )
    urgency = _clamp01(
        0.55 * _clamp01(goal_vector.get("urgency", 0.0))
        + 0.25 * _clamp01(goal_vector.get("max_angle_pressure", 0.0))
        + 0.20 * _clamp01(goal_vector.get("max_velocity_pressure", 0.0))
    )

    return Modulators(
        risk=risk,
        urgency=urgency,
        eta_tick_eff=config.eta_base * (1.0 + config.alpha_risk * risk),
        c_explore_eff=config.c_explore_base * (1.0 + config.alpha_urgency * urgency),
    )

