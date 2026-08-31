"""
Feature: breakdown_strength_20
Contract version: 1.0.0
Category: breakout
Normalization mode: sparse_positive

Purpose:
    Measures how far the last completed close broke below the preceding 20-bar low, scaled by ATR.

Formula:
    max(min(low[t-21:t-1]) - close[t-1], 0) / (ATR_14_shift1[t] + EPS)

Inputs:
    low, close, high

Parameters:
    lookback=20, atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Zero means no downside breakdown; larger values indicate a deeper ATR-scaled breakdown.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "breakdown_strength_20"
MODE = "sparse_positive"


def compute(g: pd.DataFrame) -> pd.Series:
    lookback = 20
    atr_period = 14

    prev_low = g["low"].rolling(window=lookback, min_periods=lookback).min().shift(2)

    return (prev_low - g["close"].shift(1)).clip(lower=0.0) / (atr(g, atr_period) + EPS)
