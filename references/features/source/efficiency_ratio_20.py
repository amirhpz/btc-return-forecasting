"""
Feature: efficiency_ratio_20
Contract version: 1.0.0
Category: trend_quality
Normalization mode: positive

Purpose:
    Measures directional efficiency as net movement divided by total path length over 20 completed intervals.

Formula:
    |close[t-1] - close[t-21]| / (sum_20(|delta close|) + EPS)

Inputs:
    close

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Values near 1 indicate a smooth directional path; values near 0 indicate choppy movement.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS

NAME = "efficiency_ratio_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 20

    close = g["close"].shift(1)
    net_move = (close - close.shift(window)).abs()
    path_len = close.diff(1).abs().rolling(window=window, min_periods=window).sum()

    raw = net_move / (path_len + EPS)
    return raw.clip(lower=0.0, upper=1.0)
