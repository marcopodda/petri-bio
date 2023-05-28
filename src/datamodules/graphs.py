import networkx as nx


def augment_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Given an input nx.DiGraph, adds additional backward edges from
    reactions to species representing negative influence.
    """
    G_copy = G.copy()

    for n1, n2 in G.edges:
        nt1 = G.nodes[n1]["shape"]
        nt2 = G.nodes[n2]["shape"]
        et = G.edges[(n1, n2)]["arrowhead"]

        if nt1 == "circle" and nt2 == "box" and et == "vee":
            G_copy.add_edge(n2, n1, arrowhead="vee")

    return nx.DiGraph(G_copy)


def extract_induced_subgraph(
    G: nx.DiGraph,
    input_node: str,
    output_node: str,
) -> nx.DiGraph | None:
    """
    Given an input nx.DiGraph, returns the set of nodes
    in any path from input_node to output_node.
    """
    G_augmented = augment_graph(G)
    path_nodes = set()

    if not nx.has_path(G_augmented, input_node, output_node):
        return None

    for node in G_augmented.nodes():
        has_input_node_path = nx.has_path(G_augmented, input_node, node)
        has_node_output_path = nx.has_path(G_augmented, node, output_node)
        if has_input_node_path and has_node_output_path:
            path_nodes.add(node)

    return nx.DiGraph(G.subgraph(path_nodes))
