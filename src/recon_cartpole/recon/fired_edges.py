from __future__ import annotations

from recon_lite import Graph, LinkType


def fired_edges_from_requests(graph: Graph, now_requested: dict[str, bool]) -> list[dict[str, str]]:
    fired: list[dict[str, str]] = []
    requested = {node_id for node_id, did_request in now_requested.items() if did_request}
    for edge in graph.edges:
        if edge.ltype in (LinkType.SUB, LinkType.POR) and edge.dst in requested:
            fired.append({"src": edge.src, "dst": edge.dst, "ltype": edge.ltype.name})
    return fired

