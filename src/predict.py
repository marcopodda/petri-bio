import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer

from src.utils import paths
from src.models.model import Model
from src.utils.pylogger import get_pylogger
from src.utils.evaluation import find_best_model


log = get_pylogger(__name__)


def predict(cfg: DictConfig):
    log.info("Predict started.")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Using fold 0 (this is just to record the time it takes to predict a sample)")

    ckpt_path, score = find_best_model(cfg.data.name, cfg.experiment.name, 0)
    log.info(f"Loaded best checkpoint: {ckpt_path} ({score=})")

    model = Model.load_from_checkpoint(ckpt_path)  #  type: ignore
    log.info("Model loaded from checkpoint correctly!")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)

    log.info("Starting testing!")
    output = trainer.predict(
        model=model,
        dataloaders=datamodule,
        ckpt_path=ckpt_path,  #  type: ignore
    )
    log.info(output)
    log.info("Predictions computed!")

    with open("elapsed.txt", "w") as file:
        print(f"{output[0]['elapsed']}", file=file)  # type: ignore


@hydra.main(
    version_base="1.3",
    config_path=str(paths.CONFIG_DIR),
    config_name="predict.yaml",
)
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    main()
