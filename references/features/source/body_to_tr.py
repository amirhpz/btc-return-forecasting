"""
Feature: body_to_tr
Contract version: 1.0.0
Category: candlestick_structure
Normalization mode: positive

Purpose:
    Measures the absolute body of the last completed candle relative to its True Range.

Formula:
    |close[t-1] - open[t-1]| / (TR[t-1] + EPS)

Inputs:
    open, high, low, close

Parameters:
    none

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Values near 1 indicate a body-dominated candle; values near 0 indicate wick- or gap-dominated range.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 1.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, true_range

NAME = "body_to_tr"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    return (g["close"].shift(1) - g["open"].shift(1)).abs() / (true_range(g) + EPS)
