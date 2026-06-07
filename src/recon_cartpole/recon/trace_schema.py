from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepTrace:
    step: int
    observation: list[float]
    raw_state: list[float]
    action: Any
    force: float
    env_reward: float
    reward_tick: float
    return_so_far: float
    terminated: bool
    truncated: bool
    goal_vector: dict[str, Any]
    selected_regime: str
    proposal: dict[str, Any]
    fired_edges: list[dict[str, str]] = field(default_factory=list)
    plasticity: dict[str, Any] = field(default_factory=dict)
    bandit: dict[str, Any] = field(default_factory=dict)
    graph_nodes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

