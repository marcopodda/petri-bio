from typing import Callable

import torch
from torch_geometric.data import InMemoryDataset

import pandas as pd
import numpy as np

from src.datamodules.preprocess import (
    get_knockout_data,
    get_predict_data,
    load_subgraphs_df,
    preprocess_graphs,
)
from src.utils import paths
from src.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class PetriNetDataset(InMemoryDataset):
    def __init__(self, name: str, transform: Callable = lambda d: d):
        self.name = name

        root = str(paths.DATA_DIR / name)
        super().__init__(root=root, transform=transform)
        self.info, self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_file_names(self):
        return ["data.csv"]

    def process(self):
        subgraphs_df = load_subgraphs_df(self.raw_paths[0])
        subgraphs_data = preprocess_graphs(subgraphs_df)
        info, data_list = zip(*subgraphs_data)

        # shuffle the data once
        permutation = np.random.permutation(len(data_list))  #  type: ignore
        info = [info[i] for i in permutation]  #  type: ignore
        data_list = [data_list[i] for i in permutation]  #  type: ignore

        data, slices = self.collate(data_list)  # type: ignore
        torch.save((info, data, slices), self.processed_paths[0])

    def filter_by_num_nodes(self, n: int):
        indices = [i for (i, d) in enumerate(self) if d.num_nodes <= n]  # type: ignore
        filtered_dataset = self[indices]
        filtered_dataset.info = [self.info[i] for i in indices]  # type: ignore
        return filtered_dataset

    def as_dataframe(self):
        return pd.DataFrame(self.info)


class PetriNetKnockoutDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        pathway_id: str,
        input_species: str,
        output_species: str,
        transform: Callable = lambda d: d,
    ):
        self.name = name
        self.pathway_id = pathway_id
        self.input_species = input_species
        self.output_species = output_species

        root = str(paths.DATA_DIR / name)
        super().__init__(root=root, transform=transform, log=False)

        data_list, info, G = get_knockout_data(
            name=name,
            pathway_id=pathway_id,
            input_species=input_species,
            output_species=output_species,
        )

        self.data, self.slices = self.collate(data_list)  # type: ignore
        self.info = info
        self.G = G

    @property
    def processed_file_names(self):
        return []

    def process(self):
        return None

    def as_dataframe(self):
        return pd.DataFrame(self.info)

    @property
    def removed_arcs(self):
        return [i["arc"] for i in self.info]

    @property
    def target(self):
        return self.info[0]["property"]


class PetriNetPredictDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        pathway_id: str,
        input_species: str,
        output_species: str,
        transform: Callable = lambda d: d,
    ):
        self.name = name
        self.pathway_id = pathway_id
        self.input_species = input_species
        self.output_species = output_species

        root = str(paths.DATA_DIR / name)
        super().__init__(root=root, transform=transform, log=False)

        info, data = get_predict_data(
            name=name,
            pathway_id=pathway_id,
            input_species=input_species,
            output_species=output_species,
        )

        self.data, self.slices = self.collate([data])  # type: ignore
        self.info = [info]

    @property
    def processed_file_names(self):
        return []

    def process(self):
        return None

    def as_dataframe(self):
        return pd.DataFrame(self.info)
