import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import Logger

from src.utils import paths
from src.models.model import Model
from src.utils.misc import task_wrapper, instantiate_loggers
from src.utils.evaluation import find_best_model, dump_best_config
from src.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


@task_wrapper
def evaluate(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    ckpt_path, score = find_best_model(cfg.data.name, cfg.experiment.name, cfg.data.fold)
    log.info(f"Loaded best checkpoint: {ckpt_path} ({score=})")

    model = Model.load_from_checkpoint(ckpt_path)
    log.info("Model loaded from checkpoint correctly!")

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    dump_best_config(ckpt_path)


@hydra.main(
    version_base="1.3",
    config_path=str(paths.CONFIG_DIR),
    config_name="eval.yaml",
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
