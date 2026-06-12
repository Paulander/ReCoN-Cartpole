from recon_cartpole.recon.graph_factory import build_cartpole_graph, trainable_edge_whitelist


def test_generated_graph_is_article_compliant_and_trainable():
    graph = build_cartpole_graph(3)
    graph.validate_article_compliance()
    assert "pole_2_monitor" in graph.nodes
    assert trainable_edge_whitelist(graph)



def test_generated_graph_includes_adjacent_subchain_sensors():
    graph = build_cartpole_graph(4)

    for idx in range(3):
        monitor = f"subchain_{idx}_{idx + 1}_monitor"
        sensor = f"subchain_{idx}_{idx + 1}_sensor"
        assert monitor in graph.nodes
        assert sensor in graph.nodes
        assert graph.nodes[monitor].meta["subchain"] is True
        assert graph.nodes[sensor].meta["start_pole"] == idx
        assert graph.nodes[sensor].meta["end_pole"] == idx + 1

    graph_one = build_cartpole_graph(1)
    assert "subchain_0_1_monitor" not in graph_one.nodes
