"""
Feature: ret_vol_corr_30
Contract version: 1.0.0
Category: volume_price_relation
Normalization mode: signed

Purpose:
    Computes the 30-observation rolling Pearson correlation between absolute completed log returns and completed-bar volume.

Formula:
    corr_30(|ln(close[t-1]/close[t-2])|, volume[t-1])

Inputs:
    close, volume

Parameters:
    window=30, degenerate_window_value=0.0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, 1.0]
    Interpretation: Positive values associate larger price moves with higher volume; negative values indicate the opposite.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 31.

Volume contract:
    same_volume_definition_in_training_and_metatrader

Notes:
    A mature zero-variance or numerically degenerate window is assigned neutral correlation 0. Volume definition must be consistent.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "ret_vol_corr_30"
MODE = "signed"
VOLUME_REQUIREMENT = "same_volume_definition_in_training_and_metatrader"


def compute(g: pd.DataFrame, window: int = 30) -> pd.Series:
    ret_abs = np.log(g["close"].shift(1) / g["close"].shift(2)).abs()
    volume = g["volume"].shift(1)
    correlation = ret_abs.rolling(window=window, min_periods=window).corr(volume)

    count_x = ret_abs.rolling(window=window, min_periods=window).count()
    count_y = volume.rolling(window=window, min_periods=window).count()
    std_x = ret_abs.rolling(window=window, min_periods=window).std(ddof=0)
    std_y = volume.rolling(window=window, min_periods=window).std(ddof=0)
    mature = (count_x == window) & (count_y == window)
    degenerate = mature & ((std_x <= EPS) | (std_y <= EPS) | ~np.isfinite(correlation))

    return correlation.mask(degenerate, 0.0).replace([np.inf, -np.inf], np.nan)
