from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from ..graph import Edge, Graph, LinkType


def edge_key_from_parts(src: str, dst: str, ltype: LinkType | str) -> str:
    name = ltype.name if isinstance(ltype, LinkType) else str(ltype).upper()
    return f"{src}->{dst}:{name}"


def edge_key(edge: Edge) -> str:
    return edge_key_from_parts(edge.src, edge.dst, edge.ltype)


@dataclass
class EdgePlasticityState:
    src: str
    dst: str
    ltype: LinkType
    w_init: float = 1.0
    eligibility: float = 0.0
    delta_sum: float = 0.0

    def key(self) -> str:
        return edge_key_from_parts(self.src, self.dst, self.ltype)


@dataclass
class PlasticityConfig:
    enabled: bool = True
    eta_tick: float = 0.03
    r_max: float = 1.0
    w_min: float = 0.05
    w_max: float = 5.0
    lambda_decay: float = 0.85
    max_delta_episode: Optional[float] = 1.0


def _edge_weight(edge: Edge) -> float:
    try:
        return float(edge.w[0]) if hasattr(edge.w, "__len__") else float(edge.w)
    except Exception:
        return 1.0


def init_plasticity_state(
    graph: Graph,
    edge_whitelist: Optional[Iterable[Tuple[str, str, LinkType]]] = None,
) -> Dict[str, EdgePlasticityState]:
    whitelist: Optional[Set[Tuple[str, str, str]]] = None
    if edge_whitelist is not None:
        whitelist = {
            (src, dst, ltype.name if isinstance(ltype, LinkType) else str(ltype).upper())
            for src, dst, ltype in edge_whitelist
        }

    state: Dict[str, EdgePlasticityState] = {}
    for edge in graph.edges:
        if edge.ltype not in (LinkType.SUB, LinkType.POR):
            continue
        if whitelist is not None and (edge.src, edge.dst, edge.ltype.name) not in whitelist:
            continue
        state[edge_key(edge)] = EdgePlasticityState(
            src=edge.src,
            dst=edge.dst,
            ltype=edge.ltype,
            w_init=_edge_weight(edge),
        )
    return state


def update_eligibility(
    state: Dict[str, EdgePlasticityState],
    fired_edges: Iterable[Dict[str, str]],
    lambda_decay: float,
) -> None:
    for edge_state in state.values():
        edge_state.eligibility *= lambda_decay

    fired = {
        edge_key_from_parts(item["src"], item["dst"], item.get("ltype", item.get("type", "SUB")))
        for item in fired_edges
        if "src" in item and "dst" in item
    }
    for key in fired:
        if key in state:
            state[key].eligibility += 1.0


def apply_fast_update(
    state: Dict[str, EdgePlasticityState],
    graph: Graph,
    reward_tick: float,
    eta_eff: float,
    config: PlasticityConfig,
) -> Dict[str, float]:
    if not config.enabled:
        return {}

    reward = max(-config.r_max, min(config.r_max, float(reward_tick)))
    deltas: Dict[str, float] = {}
    for key, edge_state in state.items():
        if edge_state.eligibility == 0.0:
            continue
        proposed = eta_eff * reward * edge_state.eligibility
        for edge in graph.edges:
            if edge.src == edge_state.src and edge.dst == edge_state.dst and edge.ltype == edge_state.ltype:
                current = _edge_weight(edge)
                new_weight = max(config.w_min, min(config.w_max, current + proposed))
                if config.max_delta_episode is not None:
                    new_weight = max(
                        edge_state.w_init - config.max_delta_episode,
                        min(edge_state.w_init + config.max_delta_episode, new_weight),
                    )
                actual = new_weight - current
                edge.w = new_weight
                edge_state.delta_sum += actual
                deltas[key] = actual
                break
    return deltas


def reset_episode(state: Dict[str, EdgePlasticityState], graph: Graph) -> None:
    for edge_state in state.values():
        for edge in graph.edges:
            if edge.src == edge_state.src and edge.dst == edge_state.dst and edge.ltype == edge_state.ltype:
                edge.w = edge_state.w_init
                break
        edge_state.eligibility = 0.0
        edge_state.delta_sum = 0.0


def snapshot_plasticity(state: Dict[str, EdgePlasticityState]) -> Dict[str, Any]:
    return {
        key: {
            "eligibility": round(edge_state.eligibility, 4),
            "delta_sum": round(edge_state.delta_sum, 4),
            "w_init": round(edge_state.w_init, 4),
        }
        for key, edge_state in state.items()
        if edge_state.eligibility != 0.0 or edge_state.delta_sum != 0.0
    }

