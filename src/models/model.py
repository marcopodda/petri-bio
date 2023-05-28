import time
from typing import Any, List

import torch
import pandas as pd
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.accuracy import Accuracy

from src.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class Model(LightningModule):
    def __init__(
        self,
        embedder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()

        self.embedder = embedder
        self.classifier = nn.Linear(embedder.dim_output, 1)  # type: ignore

        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_auroc = AUROC(task="binary", pos_label=1)
        self.val_auroc = AUROC(task="binary", pos_label=1)
        self.test_auroc = AUROC(task="binary", pos_label=1)

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.outputs = []

    def forward(self, batch: Any):
        embedding = self.embedder(batch)
        logits = self.classifier(embedding).view(-1)
        return torch.sigmoid(logits)

    def log_metric(self, key, val):
        self.log(key, val, on_step=False, on_epoch=True, prog_bar=True)

    def step(self, batch: Any):
        embedding = self.embedder(batch)
        logits = self.classifier(embedding).view(-1)
        loss = self.criterion(logits, batch.y)
        proba = torch.sigmoid(logits.detach())
        preds = proba.round().long()
        targets = batch.y.long()
        return (loss, proba, preds, targets)

    def training_step(self, batch: Any, batch_idx: int):
        loss, proba, preds, targets = self.step(batch)

        self.train_acc.update(preds, targets)
        self.train_auroc.update(proba, targets)

        # log train metrics
        self.log_metric("train/loss", loss.detach())
        return loss

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()
        train_acc = self.train_acc.compute()
        train_auroc = self.train_auroc.compute()

        self.log_metric("train/acc", train_acc)
        self.log_metric("train/auroc", train_auroc)

        self.train_acc.reset()
        self.train_auroc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, proba, preds, targets = self.step(batch)

        self.val_acc.update(preds, targets)
        self.val_auroc.update(proba, targets)

        # log val metrics
        self.log_metric("val/loss", loss.detach())

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_auroc = self.val_auroc.compute()

        self.log_metric("val/acc", val_acc)
        self.log_metric("val/auroc", val_auroc)

        self.val_acc.reset()
        self.val_auroc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, proba, preds, targets = self.step(batch)

        self.outputs.append({"proba": proba, "target": targets})

        self.test_acc.update(preds, targets)
        self.test_auroc.update(proba, targets)

        # log test metrics
        self.log_metric("test/loss", loss.detach())

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_auroc = self.test_auroc.compute()

        self.log_metric("test/acc", test_acc)
        self.log_metric("test/auroc", test_auroc)

        self.save_test_predictions()

        self.test_acc.reset()
        self.test_auroc.reset()

    def cat_outputs(self, key):
        tensor = torch.cat([o[key] for o in self.outputs], dim=0).cpu()
        return tensor.numpy()

    def save_test_predictions(self):
        proba = self.cat_outputs("proba")
        targets = self.cat_outputs("target")
        df = pd.DataFrame({"Prediction": proba, "Target": targets})
        df.to_csv("logs/test_predictions.csv", index=False)

    def predict_step(self, batch: Any, batch_idx: int):
        start = time.time()
        proba = self.forward(batch)
        elapsed = torch.FloatTensor([time.time() - start], device=proba.device)
        return_dict = {"proba": proba, "elapsed": elapsed}
        self.outputs.append(return_dict)
        return return_dict

    def get_predictions(self):
        proba = self.cat_outputs("proba")
        elapsed = self.cat_outputs("elapsed")
        return pd.DataFrame({"Prediction": proba, "Elapsed": elapsed})

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())  #  type: ignore
        scheduler = self.hparams.scheduler(optimizer=optimizer)  # type: ignore
        return [optimizer], [scheduler]
