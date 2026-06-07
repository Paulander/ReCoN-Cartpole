from recon_cartpole.recon.graph_factory import build_cartpole_graph, trainable_edge_whitelist


def test_generated_graph_is_article_compliant_and_trainable():
    graph = build_cartpole_graph(3)
    graph.validate_article_compliance()
    assert "pole_2_monitor" in graph.nodes
    assert trainable_edge_whitelist(graph)

