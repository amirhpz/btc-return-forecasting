"""
Feature: downside_semivol_20
Contract version: 1.0.0
Category: volatility
Normalization mode: positive

Purpose:
    Computes the root mean square of negative log returns over the latest 20 completed return observations.

Formula:
    sqrt(mean_20(min(ln(close[t-1]/close[t-2]), 0)^2))

Inputs:
    close

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate stronger downside return dispersion.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd
from feature_utils import EPS

NAME = "downside_semivol_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:

    ret = np.log((g["close"].shift(1)) / (g["close"].shift(2)))

    ret_dn = ret.clip(upper=0.0)

    mse_dn = (ret_dn.pow(2)).rolling(window=20, min_periods=20).mean()

    return np.sqrt(mse_dn)
