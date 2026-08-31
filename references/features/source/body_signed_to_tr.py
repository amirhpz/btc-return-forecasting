"""
Feature: body_signed_to_tr
Contract version: 1.0.0
Category: candlestick_structure
Normalization mode: signed

Purpose:
    Measures the signed body of the last completed candle relative to its True Range.

Formula:
    clip((close[t-1] - open[t-1]) / (TR[t-1] + EPS), -1, 1)

Inputs:
    open, high, low, close

Parameters:
    none

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, 1.0]
    Interpretation: Positive values indicate bullish bodies, negative values indicate bearish bodies, and magnitude reflects body dominance within True Range.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 1.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, true_range

NAME = "body_signed_to_tr"
MODE = "signed"


def compute(g: pd.DataFrame) -> pd.Series:
    tr = true_range(g)  # >= 0
    raw = (g["close"].shift(1) - g["open"].shift(1)) / (tr + EPS)

    # By construction, |close-open| <= TR, so raw should lie in [-1,1]
    return raw.clip(-1.0, 1.0)
