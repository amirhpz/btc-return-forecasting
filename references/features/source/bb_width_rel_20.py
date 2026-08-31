"""
Feature: bb_width_rel_20
Contract version: 1.0.0
Category: volatility
Normalization mode: positive

Purpose:
    Measures Bollinger Band width relative to the 20-period middle band.

Formula:
    (upper_20_2[t] - lower_20_2[t]) / middle_20[t]

Inputs:
    close

Parameters:
    period=20, standard_deviation_multiplier=2.0, std_ddof=0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate wider bands and greater recent dispersion.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd

from feature_utils import EPS

NAME = "bb_width_rel_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 20
    multiplier = 2.0

    past_close = g["close"].shift(1)
    middle = past_close.rolling(period, min_periods=period).mean()
    std = past_close.rolling(period, min_periods=period).std(ddof=0)
    width = 2.0 * multiplier * std
    return width.div(middle.where(middle.abs() > EPS))
