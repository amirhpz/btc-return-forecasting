"""
Feature: ret_autocorr_1_30
Contract version: 1.0.0
Category: serial_dependence
Normalization mode: signed

Purpose:
    Computes the 30-observation rolling lag-1 Pearson autocorrelation of completed log returns.

Formula:
    corr_30(return[t], return[t-1])

Inputs:
    close

Parameters:
    window=30, lag=1, degenerate_window_value=0.0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, 1.0]
    Interpretation: Positive values indicate return persistence; negative values indicate mean-reverting alternation.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 32.

Volume contract:
    none

Notes:
    A mature zero-variance window is assigned neutral correlation 0 instead of NaN.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "ret_autocorr_1_30"
MODE = "signed"


def compute(g: pd.DataFrame, window: int = 30) -> pd.Series:
    ret = np.log(g["close"].shift(1) / g["close"].shift(2))
    lagged = ret.shift(1)
    correlation = ret.rolling(window=window, min_periods=window).corr(lagged)

    count_x = ret.rolling(window=window, min_periods=window).count()
    count_y = lagged.rolling(window=window, min_periods=window).count()
    std_x = ret.rolling(window=window, min_periods=window).std(ddof=0)
    std_y = lagged.rolling(window=window, min_periods=window).std(ddof=0)
    mature = (count_x == window) & (count_y == window)
    degenerate = mature & ((std_x <= EPS) | (std_y <= EPS) | ~np.isfinite(correlation))

    return correlation.mask(degenerate, 0.0).replace([np.inf, -np.inf], np.nan)
