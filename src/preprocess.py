import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from src.utils import paths
from src.datamodules.datasets import PetriNetDataset
from src.datamodules.datamodules import PetriNetDataModule


def preprocess(cfg: DictConfig):
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    choices = ("monotonicity", "sensitivity", "robustness")
    if cfg.dataset not in ("monotonicity", "sensitivity", "robustness"):
        raise ValueError(f"cfg.dataset should be one of {choices}")

    _ = PetriNetDataset(cfg.dataset)
    datamodule = PetriNetDataModule(cfg.dataset, fold=0)
    datamodule.setup()


@hydra.main(
    version_base="1.3",
    config_path=str(paths.CONFIG_DIR),
    config_name="preprocess.yaml",
)
def main(cfg: DictConfig) -> None:
    preprocess(cfg)


if __name__ == "__main__":
    main()
