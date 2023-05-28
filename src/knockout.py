import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer

from src.utils import paths
from src.models.model import Model
from src.utils.pylogger import get_pylogger
from src.utils.evaluation import find_best_model, retrieve_test_fold_index
from src.utils.plots import print_colored_subgraph


log = get_pylogger(__name__)


def knockout(cfg: DictConfig):
    log.info("Knockout started.")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    fold_index = retrieve_test_fold_index(
        cfg.data.name,
        cfg.data.pathway_id,
        cfg.data.input_species,
        cfg.data.output_species,
    )
    log.info(f"Found test fold for the pathway: {fold_index}")

    log.info(f"Processing fold {fold_index}.")
    ckpt_path, score = find_best_model(cfg.data.name, cfg.experiment.name, fold_index)
    log.info(f"Loaded best checkpoint: {ckpt_path} ({score=})")

    model = Model.load_from_checkpoint(ckpt_path, map_location="cpu")
    log.info("Model loaded from checkpoint correctly!")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)

    log.info("Starting testing!")
    trainer.predict(model=model, dataloaders=datamodule)
    log.info("Predictions computed!")

    #  type: ignore
    dataset = datamodule.data_knockout  # type: ignore
    df = dataset.as_dataframe()
    df["prediction"] = trainer.model.cat_outputs("proba")  #  type: ignore
    df.to_csv("predictions.csv", index=False)

    print_colored_subgraph(
        G=dataset.G,
        name=dataset.name.capitalize(),
        input_species=dataset.input_species,
        output_species=dataset.output_species,
        prop=dataset.target,
        prediction=df.prediction,  # type: ignore
        edges=dataset.removed_arcs,
        output_path="knockout_graph.dot",  # type: ignore
    )


@hydra.main(
    version_base="1.3",
    config_path=str(paths.CONFIG_DIR),
    config_name="knockout.yaml",
)
def main(cfg: DictConfig) -> None:
    knockout(cfg)


if __name__ == "__main__":
    main()
