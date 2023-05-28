from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets import (
    PetriNetDataset,
    PetriNetKnockoutDataset,
    PetriNetPredictDataset,
)
from src.datamodules.transforms import edge_aware


def get_class_weights(
    targets: np.ndarray,
    nclasses: int = 2,
) -> torch.Tensor:
    """Returns a tensor of weights for each sample based on class proportions."""
    targets = targets.astype(int)
    count = [0] * nclasses

    for item in targets:
        count[item] += 1

    weight_per_class = [0.0] * nclasses
    N = float(sum(count))

    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(targets)
    for idx, val in enumerate(targets):
        weight[idx] = weight_per_class[val]  # type: ignore

    return torch.FloatTensor(weight)


def split_data(dataset) -> list:
    indices = np.arange(len(dataset))
    targets = dataset.y.numpy()
    out_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    in_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    splits = []
    for dev_idx, test_idx in out_split.split(indices, targets):
        train_idx, val_idx = next(in_split.split(indices[dev_idx], targets[dev_idx]))
        splits.append({"train": train_idx, "val": val_idx, "test": test_idx})

    return splits


class PetriNetDataModule(LightningDataModule):
    """Datamodule for a Petri net dataset."""

    def __init__(
        self,
        name: str,
        fold: int,
        max_nodes: int = 200,
        transform: Callable = edge_aware,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.data_train: PetriNetDataset | None = None
        self.data_val: PetriNetDataset | None = None
        self.data_test: PetriNetDataset | None = None

        # load and split datasets only if not loaded already
        dataset = PetriNetDataset(name, transform=transform)
        self.dataset = dataset.filter_by_num_nodes(max_nodes)

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""

        if not self.data_train and not self.data_val and not self.data_test:
            splits = self.load_splits()
            fold = self.hparams.fold  # type: ignore
            self.data_train = self.dataset[splits[fold]["train"]]  # type: ignore
            self.data_val = self.dataset[splits[fold]["val"]]  # type: ignore
            self.data_test = self.dataset[splits[fold]["test"]]  # type: ignore

    def train_dataloader(self) -> DataLoader:
        """Returns DataLoader object for training set."""
        weights = get_class_weights(self.data_train.y.numpy())  # type: ignore
        sampler = WeightedRandomSampler(weights, len(weights))  # type: ignore
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader object for validation set."""
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns DataLoader object for test set."""
        return DataLoader(
            dataset=self.data_test,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=False,
        )

    def load_splits(self):
        assert self.dataset is not None
        base_dir = Path(self.dataset.processed_dir)
        splits_path = base_dir / f"split_{self.hparams.max_nodes}.pt"  #  type: ignore
        if not splits_path.exists():
            splits = split_data(self.dataset)
            torch.save(splits, splits_path)
        return torch.load(splits_path)


class PetriNetKnockoutDataModule(LightningDataModule):
    """Datamodule for a Petri net knockout dataset."""

    def __init__(
        self,
        name: str,
        pathway_id: str,
        input_species: str,
        output_species: str,
        transform: Callable = edge_aware,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.data_knockout: InMemoryDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_knockout`."""
        # load and split datasets only if not loaded already
        if self.data_knockout is None:
            self.data_knockout = PetriNetKnockoutDataset(
                name=self.hparams.name,  # type: ignore
                pathway_id=self.hparams.pathway_id,  # type: ignore
                input_species=self.hparams.input_species,  # type: ignore
                output_species=self.hparams.output_species,  # type: ignore
                transform=self.hparams.transform,  # type: ignore
            )

    def predict_dataloader(self) -> DataLoader:
        """Returns DataLoader object for predictions."""
        return DataLoader(
            dataset=self.data_knockout,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=False,
        )


class PetriNetPredictDataModule(LightningDataModule):
    """Datamodule for a Petri net single prediction dataset."""

    def __init__(
        self,
        name: str,
        pathway_id: str,
        input_species: str,
        output_species: str,
        transform: Callable = edge_aware,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.dataset: InMemoryDataset | None = None

    def setup(self, stage: str | None = None):
        """Load data. Set variables: `self.dataset`."""
        # load and split datasets only if not loaded already
        if self.dataset is None:
            self.dataset = PetriNetPredictDataset(
                name=self.hparams.name,  # type: ignore
                pathway_id=self.hparams.pathway_id,  # type: ignore
                input_species=self.hparams.input_species,  # type: ignore
                output_species=self.hparams.output_species,  # type: ignore
                transform=self.hparams.transform,  # type: ignore
            )

    def predict_dataloader(self) -> DataLoader:
        """Returns DataLoader object for single prediction."""
        return DataLoader(
            dataset=self.dataset,  # type: ignore
            batch_size=1,  # self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=False,
        )
