"""
Feature: range_compression_20_100
Contract version: 1.0.0
Category: volatility_regime
Normalization mode: positive

Purpose:
    Compares the 20-bar high-low range with the 100-bar high-low range.

Formula:
    (max_20(high)-min_20(low)) / (max_100(high)-min_100(low)+EPS)

Inputs:
    high, low

Parameters:
    short_window=20, long_window=100

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Low values indicate short-term compression; values near 1 indicate the short range spans most of the long range.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 100.

Volume contract:
    none

Notes:
    None.
"""

from __future__ import annotations

import pandas as pd
from feature_utils import EPS

NAME = "range_compression_20_100"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    short_window = 20
    long_window = 100

    past_high = g["high"].shift(1)
    past_low = g["low"].shift(1)

    high_short = past_high.rolling(window=short_window, min_periods=short_window).max()
    low_short = past_low.rolling(window=short_window, min_periods=short_window).min()

    high_long = past_high.rolling(window=long_window, min_periods=long_window).max()
    low_long = past_low.rolling(window=long_window, min_periods=long_window).min()

    raw = (high_short - low_short) / ((high_long - low_long) + EPS)
    return raw.clip(lower=0.0, upper=1.0)
