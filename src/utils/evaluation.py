from typing import Optional, List
from pathlib import Path

import yaml
import torch
import pandas as pd

from src.utils import paths
from src.datamodules.datasets import PetriNetDataset
from src.utils.pylogger import get_pylogger
from src.utils.misc import iter_dir


log = get_pylogger(__name__)


class NotFoundError(Exception):
    pass


def load_config(exp_root):
    config_path = exp_root / "logs" / "hparams.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def dump_config(config, path="logs/hparams.yaml"):
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, Dumper=yaml.Dumper)


def extract_from_config(config, dotted_path):
    parts = dotted_path.split(".")
    if len(parts) == 1:
        return config[parts[0]]
    return extract_from_config(config[parts[0]], ".".join(parts[1:]))


def infer_ckpt_filename(hparam_dir: Path) -> Optional[Path]:
    ckpt_dir = hparam_dir / "checkpoints"
    try:
        ckpt_paths = sorted(ckpt_dir.glob("epoch_*.ckpt"))
        return ckpt_paths[-1]
    except IndexError:
        return None


def collect_fold_results(dataset_name: str, experiment_name: str, fold_index: int) -> pd.DataFrame:
    experiment_dir = paths.EXPS_DIR / dataset_name / experiment_name / "train"
    fold_dir = experiment_dir / f"fold_{fold_index}"

    rows = []
    for hparam_dir in iter_dir(fold_dir, "L=*"):
        csv_filename = hparam_dir / "logs" / "metrics.csv"  # type: ignore
        if ckpt_filename := infer_ckpt_filename(hparam_dir):
            auroc = pd.read_csv(csv_filename)["val/auroc"].max()
            rows.append({"Fold": fold_index, "AUROC": auroc, "Ckpt": ckpt_filename})

    if rows == []:
        msg = f"{dataset_name} - {experiment_name}: "
        raise Exception(msg + f"No data to collect from fold {fold_index}!")

    return pd.DataFrame(rows)


def collect_cv_results(dataset_name: str, experiment_name: str) -> pd.DataFrame:
    dfs = []

    for fold_index in range(5):
        dfs.append(collect_fold_results(dataset_name, experiment_name, fold_index))

    return pd.concat(dfs, axis=0, ignore_index=True)


def find_best_model(
    dataset_name: str,
    experiment_name: str,
    fold_index: int,
) -> tuple[Path, float]:
    fold_results = collect_fold_results(dataset_name, experiment_name, fold_index)
    best_index = fold_results.AUROC.idxmax()
    best = fold_results.iloc[best_index]  # type: ignore
    return best.Ckpt, best.AUROC


def dump_best_config(path):
    config = load_config(path.parent.parent)
    dump_config(config)


def load_checkpoints(dataset_name: str, experiment_name: str) -> List[Path]:
    checkpoints = []
    for fold_index in range(5):
        ckpt = find_best_model(dataset_name, experiment_name, fold_index)
        checkpoints.append(ckpt)
    return checkpoints


def retrieve_test_fold_index(name: str, pathway_id: str, input_species: str, output_species: str, max_nodes: int = 200):
    df = PetriNetDataset(name).filter_by_num_nodes(max_nodes).as_dataframe()  #  type: ignore
    match = df[(df.pathway_id == pathway_id) & (df.input == input_species) & (df.output == output_species)]

    if match.empty:
        raise ValueError(f"No data for {pathway_id} ({input_species}, {output_species})")

    index = match.index[0]

    splits = torch.load(paths.DATA_DIR / name / "processed" / f"split_{max_nodes}.pt")
    for fold_idx, fold in enumerate(splits):
        if index in fold["test"]:
            return fold_idx

    raise ValueError(f"test fold {index} not found!")
