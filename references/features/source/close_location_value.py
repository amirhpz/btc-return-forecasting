"""
Feature: close_location_value
Contract version: 1.0.0
Category: candlestick_structure
Normalization mode: signed

Purpose:
    Computes the classical signed Close Location Value for the last completed candle.

Formula:
    ((close[t-1]-low[t-1]) - (high[t-1]-close[t-1])) / (high[t-1]-low[t-1])

Inputs:
    high, low, close

Parameters:
    none

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, 1.0]
    Interpretation: One means close at the high, minus one means close at the low, and zero is the midpoint or a flat candle.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 1.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "close_location_value"
MODE = "signed"


def compute(g: pd.DataFrame) -> pd.Series:
    close = g["close"].shift(1)
    low = g["low"].shift(1)
    high = g["high"].shift(1)
    candle_range = high - low

    out = pd.Series(np.nan, index=g.index, dtype=float)
    valid = candle_range > EPS
    out.loc[valid] = (
        (close.loc[valid] - low.loc[valid])
        - (high.loc[valid] - close.loc[valid])
    ) / candle_range.loc[valid]
    flat = candle_range.notna() & ~valid
    out.loc[flat] = 0.0
    return out.clip(-1.0, 1.0)
