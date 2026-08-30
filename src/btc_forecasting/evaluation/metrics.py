from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    true_epsilon: float = 1e-12,
) -> float:
    """Compute sign agreement; zero predictions count as misses."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(truth) > true_epsilon
    if not np.any(mask):
        return float("nan")
    truth_sign = np.sign(truth[mask])
    pred_sign = np.sign(pred[mask])
    return float(np.mean(truth_sign == pred_sign))


def mse_skill_vs_zero_return(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute 1 - model MSE / zero-return MSE."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    model_mse = mean_squared_error(truth, pred)
    baseline_mse = mean_squared_error(truth, np.zeros_like(truth))
    if baseline_mse == 0:
        return float("nan")
    return float(1.0 - model_mse / baseline_mse)


def _safe_correlation(y_true: np.ndarray, y_pred: np.ndarray, *, rank: bool) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(truth) < 2 or np.all(truth == truth[0]) or np.all(pred == pred[0]):
        return float("nan")
    result = spearmanr(truth, pred) if rank else pearsonr(truth, pred)
    return float(result.statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return the primary regression metric set in target space."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if truth.size == 0:
        raise ValueError("metric inputs must not be empty")

    mse = mean_squared_error(truth, pred)
    return {
        "n": int(truth.size),
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(truth, pred)),
        "pearson_ic": _safe_correlation(truth, pred, rank=False),
        "spearman_rank_ic": _safe_correlation(truth, pred, rank=True),
        "directional_accuracy": directional_accuracy(truth, pred),
        "mse_skill_vs_zero_return": mse_skill_vs_zero_return(truth, pred),
    }
