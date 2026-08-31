"""
Feature: channel_pos_20
Contract version: 1.0.0
Category: price_position
Normalization mode: positive

Purpose:
    Locates the last completed close within the high-low channel of the latest 20 completed bars.

Formula:
    (close[t-1] - min(low[t-20:t])) / (max(high[t-20:t]) - min(low[t-20:t]) + EPS)

Inputs:
    high, low, close

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Zero is the channel bottom and one is the channel top.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    none

Notes:
    None.
"""

from __future__ import annotations

import pandas as pd
from feature_utils import EPS

NAME = "channel_pos_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 20
    past_high = g["high"].shift(1)
    past_low = g["low"].shift(1)
    past_close = g["close"].shift(1)

    high_w = past_high.rolling(window=window, min_periods=window).max()
    low_w = past_low.rolling(window=window, min_periods=window).min()

    raw = (past_close - low_w) / ((high_w - low_w) + EPS)
    return raw.clip(lower=0.0, upper=1.0)
