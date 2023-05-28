from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv as GC, global_add_pool


class GraphConv(nn.Module):
    """Graph convolution class."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        edge_types: List[int],
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.use_conv = use_conv
        self.edge_types = []
        self.convs = nn.ModuleList([])

        if use_conv:
            self.edge_types = edge_types
            for _ in edge_types:
                self.convs.append(GC(dim_input, dim_output))

        self.linear = nn.Linear(dim_input, dim_output)
        self.batchnorm = nn.BatchNorm1d(dim_output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        outputs = 0

        if self.use_conv:
            for et, conv in zip(self.edge_types, self.convs):
                outputs += conv((x, None), edge_index[:, edge_attr == et])

        outputs += self.linear(x)
        return self.batchnorm(F.relu(outputs))


class Embedder(nn.Module):
    """Graph embedder class."""

    def __init__(
        self,
        num_layers: int,
        dim_embed: int,
        dim_input: int = 3,
        edge_types: List[int] = [0, 1, 2],
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_embed = dim_embed
        self.dim_output = dim_embed * num_layers
        self.use_conv = use_conv

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_embed
            layer = GraphConv(
                dim_input, dim_embed, edge_types=edge_types, use_conv=use_conv
            )
            self.layers.append(layer)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        outputs = []
        for layer in self.layers:
            x = layer(
                x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
            )
            graph_repr = self.norm_pool(x, batch=data.batch)
            outputs.append(graph_repr)

        return torch.cat(outputs, dim=1)  # type: ignore

    @staticmethod
    def norm_pool(
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        graph_repr = global_add_pool(x, batch=batch)
        return graph_repr / torch.sqrt(batch.bincount())[:, None]
