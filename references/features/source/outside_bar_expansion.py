"""
Feature: outside_bar_expansion
Contract version: 1.0.0
Category: candlestick_pattern
Normalization mode: sparse_positive

Purpose:
    Scores a completed outside bar by its full candle range relative to ATR.

Formula:
    I[high[t-1] >= high[t-2] and low[t-1] <= low[t-2]] * (high[t-1]-low[t-1]) / (ATR_14_shift1[t] + EPS)

Inputs:
    high, low, close

Parameters:
    atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Zero means no outside-bar event; larger values indicate stronger range expansion.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "outside_bar_expansion"
MODE = "sparse_positive"


def compute(g: pd.DataFrame) -> pd.Series:
    atr_period = 14

    prev_high = g["high"].shift(1)
    prev_low = g["low"].shift(1)
    prev2_high = g["high"].shift(2)
    prev2_low = g["low"].shift(2)

    outside = ((prev_high >= prev2_high) & (prev_low <= prev2_low)).astype(float)
    raw = outside * (prev_high - prev_low) / (atr(g, period=atr_period) + EPS)
    return raw
