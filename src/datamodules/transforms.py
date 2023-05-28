import torch
from torch_geometric.data import Data


def baseline(data: Data) -> Data:
    return data


def connectivity(data: Data) -> Data:
    # remove all node features
    data.x = torch.ones(data.x.size(0), 1)  #  type: ignore

    # remove edge features
    data.edge_attr = torch.zeros_like(data.edge_attr)  # type: ignore

    return data


def source_dest(data: Data) -> Data:
    # remove node type feature
    data.x = data.x[:, 1:]  # type: ignore

    # remove edge features
    data.edge_attr = torch.zeros_like(data.edge_attr)  # type: ignore

    return data


def node_type(data: Data) -> Data:
    # remove edge features
    data.edge_attr = torch.zeros_like(data.edge_attr)  # type: ignore

    return data


def edge_aware(data: Data) -> Data:
    return data
