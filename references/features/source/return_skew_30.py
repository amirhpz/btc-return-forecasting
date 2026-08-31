"""
Feature: return_skew_30
Contract version: 1.0.0
Category: return_distribution
Normalization mode: signed

Purpose:
    Computes rolling sample skewness of the latest 30 completed log returns.

Formula:
    skew_30(ln(close[t-1]/close[t-2]))

Inputs:
    close

Parameters:
    window=30, degenerate_window_value=0.0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Positive values indicate a longer right tail; negative values indicate a longer left tail.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 31.

Volume contract:
    none

Notes:
    A mature near-zero-variance window is assigned neutral skewness 0 instead of NaN.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "return_skew_30"
MODE = "signed"


def compute(g: pd.DataFrame, window: int = 30) -> pd.Series:
    ret = np.log(g["close"].shift(1) / g["close"].shift(2))
    skewness = ret.rolling(window=window, min_periods=window).skew()

    count = ret.rolling(window=window, min_periods=window).count()
    std = ret.rolling(window=window, min_periods=window).std(ddof=0)
    mature = count == window
    degenerate = mature & ((std <= EPS) | ~np.isfinite(skewness))

    return skewness.mask(degenerate, 0.0).replace([np.inf, -np.inf], np.nan)
