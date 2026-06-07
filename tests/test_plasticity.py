from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite.plasticity import PlasticityConfig, apply_fast_update, init_plasticity_state, update_eligibility


def test_fast_plasticity_updates_whitelisted_edge():
    graph = Graph()
    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(Node("leaf", NodeType.TERMINAL))
    graph.add_hierarchy_pair("root", "leaf")
    state = init_plasticity_state(graph, [("root", "leaf", LinkType.SUB)])
    update_eligibility(state, [{"src": "root", "dst": "leaf", "ltype": "SUB"}], 0.85)
    deltas = apply_fast_update(state, graph, 1.0, 0.1, PlasticityConfig())
    assert deltas
    assert graph.edges[0].w > 1.0

