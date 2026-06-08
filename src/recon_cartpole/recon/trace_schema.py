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
    proposals: list[dict[str, Any]] = field(default_factory=list)
    suppressed_proposals: list[dict[str, Any]] = field(default_factory=list)
    selection_mode: str = "soft_select"
    fired_edges: list[dict[str, str]] = field(default_factory=list)
    plasticity: dict[str, Any] = field(default_factory=dict)
    fast_deltas: dict[str, float] = field(default_factory=dict)
    node_params: dict[str, Any] = field(default_factory=dict)
    node_param_deltas: dict[str, float] = field(default_factory=dict)
    mlp_terminal: dict[str, Any] = field(default_factory=dict)
    bandit: dict[str, Any] = field(default_factory=dict)
    consolidation: dict[str, Any] = field(default_factory=dict)
    graph_nodes: dict[str, str] = field(default_factory=dict)
    graph_ticks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

