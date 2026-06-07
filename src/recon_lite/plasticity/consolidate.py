from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..graph import Graph, LinkType


@dataclass
class EpisodeSummary:
    edge_delta_sums: Dict[str, float] = field(default_factory=dict)
    avg_reward_tick: float = 0.0
    outcome_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationConfig:
    enabled: bool = False
    eta_consolidate: float = 0.005
    min_episodes: int = 50
    outcome_weight: float = 0.5
    max_base_delta: float = 0.25
    w_min: float = 0.05
    w_max: float = 5.0


@dataclass
class EdgeConsolidationState:
    edge_key: str
    w_base: float = 1.0
    w_init: float = 1.0
    accumulated_weighted_delta: float = 0.0
    episode_count: int = 0

    def mean_weighted_delta(self) -> float:
        return self.accumulated_weighted_delta / self.episode_count if self.episode_count else 0.0


class ConsolidationEngine:
    def __init__(self, config: Optional[ConsolidationConfig] = None):
        self.config = config or ConsolidationConfig()
        self.edge_states: Dict[str, EdgeConsolidationState] = {}
        self.total_episodes = 0
        self.episodes_since_apply = 0
        self.last_apply_time: Optional[str] = None

    def init_from_graph(self, graph: Graph, edge_whitelist: Optional[list[str]] = None) -> None:
        whitelist = set(edge_whitelist) if edge_whitelist else None
        for edge in graph.edges:
            if edge.ltype not in (LinkType.SUB, LinkType.POR):
                continue
            key = f"{edge.src}->{edge.dst}:{edge.ltype.name}"
            if whitelist is not None and key not in whitelist:
                continue
            try:
                weight = float(edge.w)
            except Exception:
                weight = 1.0
            self.edge_states.setdefault(key, EdgeConsolidationState(key, weight, weight))

    def accumulate_episode(self, summary: EpisodeSummary) -> None:
        if not self.config.enabled:
            return
        self.total_episodes += 1
        self.episodes_since_apply += 1
        episode_delta = (
            summary.outcome_score * self.config.outcome_weight
            + summary.avg_reward_tick * (1.0 - self.config.outcome_weight)
        )
        for key, edge_delta in summary.edge_delta_sums.items():
            state = self.edge_states.setdefault(key, EdgeConsolidationState(key))
            state.accumulated_weighted_delta += edge_delta * episode_delta
            state.episode_count += 1

    def should_apply(self) -> bool:
        return self.config.enabled and self.episodes_since_apply >= self.config.min_episodes

    def apply_to_graph(self, graph: Graph) -> Dict[str, float]:
        if not self.config.enabled:
            return {}
        applied: Dict[str, float] = {}
        for key, state in self.edge_states.items():
            if not state.episode_count:
                continue
            delta = max(
                -self.config.max_base_delta,
                min(self.config.max_base_delta, self.config.eta_consolidate * state.mean_weighted_delta()),
            )
            new_base = max(self.config.w_min, min(self.config.w_max, state.w_base + delta))
            actual = new_base - state.w_base
            if abs(actual) > 1e-9:
                state.w_base = new_base
                applied[key] = actual
                _apply_weight(graph, key, new_base)
            state.accumulated_weighted_delta = 0.0
            state.episode_count = 0
        self.episodes_since_apply = 0
        self.last_apply_time = datetime.now().isoformat()
        return applied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "total_episodes": self.total_episodes,
            "episodes_since_apply": self.episodes_since_apply,
            "last_apply_time": self.last_apply_time,
            "edge_states": {key: state.__dict__ for key, state in self.edge_states.items()},
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _apply_weight(graph: Graph, key: str, weight: float) -> bool:
    src, rest = key.split("->", 1)
    dst, ltype = rest.split(":", 1)
    for edge in graph.edges:
        if edge.src == src and edge.dst == dst and edge.ltype.name == ltype:
            edge.w = weight
            return True
    return False

