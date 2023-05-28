from typing import Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    matthews_corrcoef,
    f1_score,
)


def round_score(metric):
    def wrapper(y_true: np.ndarray, y_score: np.ndarray):
        y_pred = y_score.round().astype(int)
        return metric(y_true, y_pred)

    return wrapper


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


@round_score
def accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_score))


@round_score
def mcc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_score))


@round_score
def wf1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(f1_score(y_true, y_score, average="weighted"))


@round_score
def sensitivity(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(recall_score(y_true, y_score, pos_label=1))


@round_score
def specificity(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(recall_score(y_true, y_score, pos_label=0))


def score_predictions(predictions: pd.DataFrame) -> Dict[str, float]:
    y_true = np.array(predictions.Target)
    y_score = np.array(predictions.Prediction)

    return {
        "AUROC": auroc(y_true, y_score),
        "Accuracy": accuracy(y_true, y_score),
        "WF1": wf1(y_true, y_score),
        "MCC": mcc(y_true, y_score),
        "Sensitivity": sensitivity(y_true, y_score),
        "Specificity": specificity(y_true, y_score),
    }
