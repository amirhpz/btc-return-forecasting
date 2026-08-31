"""
Feature: upper_wick_to_tr
Contract version: 1.0.0
Category: candlestick_structure
Normalization mode: positive

Purpose:
    Measures the upper wick of the last completed candle relative to its True Range.

Formula:
    (high[t-1] - max(open[t-1], close[t-1])) / (TR[t-1] + EPS)

Inputs:
    open, high, low, close

Parameters:
    none

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Higher values indicate stronger upper-wick rejection relative to the total true range.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 1.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, true_range


NAME = "upper_wick_to_tr"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    upper_wick = g["high"].shift(1) - pd.concat(
        [g["open"].shift(1), g["close"].shift(1)], axis=1
    ).max(axis=1)
    return upper_wick / (true_range(g) + EPS)
