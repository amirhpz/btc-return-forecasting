"""
Feature: upside_semivol_20
Contract version: 1.0.0
Category: volatility
Normalization mode: positive

Purpose:
    Computes the root mean square of positive log returns over the latest 20 completed return observations.

Formula:
    sqrt(mean_20(max(ln(close[t-1]/close[t-2]), 0)^2))

Inputs:
    close

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate stronger upside return dispersion.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

NAME = "upside_semivol_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:

    ret = np.log((g["close"].shift(1)) / (g["close"].shift(2)))

    ret_up = ret.clip(lower=0.0)

    mse_up = (ret_up.pow(2)).rolling(window=20, min_periods=20).mean()

    return np.sqrt(mse_up)
