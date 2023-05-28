import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src.utils import paths
from src.utils.misc import (
    task_wrapper,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from src.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


@task_wrapper
def train(cfg: DictConfig) -> None:
    """Trains the model."""

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {"cfg": cfg, "model": model, "trainer": trainer}
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(
    version_base="1.3",
    config_path=str(paths.CONFIG_DIR),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> None:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
