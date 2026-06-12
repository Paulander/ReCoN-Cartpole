from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from recon_lite import Graph, LinkType, Node, NodeType

from recon_cartpole.control.scripts import REGIMES


@dataclass
class GraphConfig:
    action_mode: str = "discrete"
    force_mag: float = 10.0


def terminal(predicate_name: str) -> Callable:
    def _predicate(node, env):
        callbacks = env["callbacks"]
        return callbacks[predicate_name](node, env)

    return _predicate


def build_cartpole_graph(n_poles: int, config: GraphConfig | None = None) -> Graph:
    _ = config or GraphConfig()
    graph = Graph()

    def script(nid: str, **meta):
        graph.add_node(Node(nid, NodeType.SCRIPT, meta=meta))

    def term(nid: str, callback: str, **meta):
        graph.add_node(Node(nid, NodeType.TERMINAL, predicate=terminal(callback), meta=meta))

    script("root_balance")
    for nid in ["observe_state", "estimate_risk", "select_control_regime", "arbitrate_force", "apply_force"]:
        script(nid)
        graph.add_hierarchy_pair("root_balance", nid)

    graph.add_sequence_pair("observe_state", "estimate_risk")
    graph.add_sequence_pair("estimate_risk", "select_control_regime")
    graph.add_sequence_pair("select_control_regime", "arbitrate_force")
    graph.add_sequence_pair("arbitrate_force", "apply_force")

    term("observe_state_terminal", "observe_state")
    term("estimate_risk_terminal", "estimate_risk")
    term("arbitrate_force_terminal", "arbitrate_force")
    term("apply_force_terminal", "apply_force")
    graph.add_hierarchy_pair("observe_state", "observe_state_terminal")
    graph.add_hierarchy_pair("estimate_risk", "estimate_risk_terminal")
    graph.add_hierarchy_pair("arbitrate_force", "arbitrate_force_terminal")
    graph.add_hierarchy_pair("apply_force", "apply_force_terminal")

    for regime in REGIMES:
        script(regime, alt=True, trainable=True)
        term(f"{regime}_proposal", "proposal", regime=regime)
        graph.add_hierarchy_pair("select_control_regime", regime)
        graph.add_hierarchy_pair(regime, f"{regime}_proposal")

    for idx in range(n_poles):
        script(f"pole_{idx}_monitor")
        term(f"pole_{idx}_sensor", "pole_sensor", pole_index=idx)
        graph.add_hierarchy_pair("observe_state", f"pole_{idx}_monitor")
        graph.add_hierarchy_pair(f"pole_{idx}_monitor", f"pole_{idx}_sensor")

    for idx in range(max(0, n_poles - 1)):
        script(f"subchain_{idx}_{idx + 1}_monitor", subchain=True)
        term(
            f"subchain_{idx}_{idx + 1}_sensor",
            "subchain_sensor",
            start_pole=idx,
            end_pole=idx + 1,
        )
        graph.add_hierarchy_pair("observe_state", f"subchain_{idx}_{idx + 1}_monitor")
        graph.add_hierarchy_pair(
            f"subchain_{idx}_{idx + 1}_monitor",
            f"subchain_{idx}_{idx + 1}_sensor",
        )

    graph.nodes["root_balance"].meta["confirm_policy"] = "and"
    graph.nodes["select_control_regime"].meta["confirm_policy"] = "and"

    for edge in graph.edges:
        if edge.ltype in (LinkType.SUB, LinkType.POR):
            edge.meta["trainable"] = (
                edge.src == "select_control_regime"
                or edge.dst.endswith("_proposal")
                or edge.src in ("select_control_regime", "arbitrate_force")
            )

    graph.validate_article_compliance()
    return graph


def trainable_edge_whitelist(graph: Graph):
    return [(edge.src, edge.dst, edge.ltype) for edge in graph.edges if edge.meta.get("trainable")]

