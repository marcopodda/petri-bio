import torch
import networkx as nx


SHAPE2INT = {"box": 0, "circle": 1}
ARROW2INT = {"vee": 0, "tee": 1, "dot": 2}


def get_edge_features(G: nx.Graph) -> tuple[torch.Tensor, torch.Tensor]:
    node2int = {n: i for (i, n) in enumerate(G.nodes())}
    edge_index, edge_attrs = [], []

    for e1, e2, edge_data in G.edges(data=True):  # type: ignore
        e1, e2 = node2int[e1], node2int[e2]
        edge_index.append([e1, e2])
        edge_type = ARROW2INT[edge_data["arrowhead"]]  # type: ignore
        edge_attrs.append(edge_type)

    edge_index = torch.LongTensor(edge_index).t()
    edge_attrs = torch.LongTensor(edge_attrs)
    return edge_index, edge_attrs


def get_node_features(G: nx.Graph, input_node: str, output_node: str) -> torch.Tensor:
    node_matrix = []

    for node, node_data in G.nodes(data=True):
        node_features = []
        node_type = SHAPE2INT[node_data["shape"]]  # type: ignore
        node_features.append(node_type)
        node_features.append(int(node == input_node))
        node_features.append(int(node == output_node))
        node_matrix.append(node_features)

    return torch.Tensor(node_matrix)
