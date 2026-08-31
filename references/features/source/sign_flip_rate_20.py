"""
Feature: sign_flip_rate_20
Contract version: 1.0.0
Category: market_regime
Normalization mode: positive

Purpose:
    Measures how frequently the sign of completed log returns changes over the latest 20 observations.

Formula:
    mean_20(I[sign(return[t]) * sign(return[t-1]) < 0])

Inputs:
    close

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Higher values indicate choppier direction changes; lower values indicate directional persistence.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 19.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd
from feature_utils import EPS

NAME = "sign_flip_rate_20"
MODE = "positive"


def compute(g: pd.DataFrame, window: int = 20) -> pd.Series:

    ret = np.log((g["close"].shift(1)) / (g["close"].shift(2)))

    sign = np.sign(ret)

    flip = ((sign * sign.shift(1)) < 0).astype(float)

    return flip.rolling(window=window, min_periods=window).mean()
