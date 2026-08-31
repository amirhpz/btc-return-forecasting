"""
Feature: up_close_ratio_5
Contract version: 1.0.0
Category: momentum
Normalization mode: positive

Purpose:
    Measures the fraction of the latest five completed close-to-close changes that were positive.

Formula:
    mean_5(I[close[t-1] > close[t-2]])

Inputs:
    close

Parameters:
    window=5

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: One means all five changes were positive; zero means none were positive.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 4.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd

NAME = "up_close_ratio_5"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 5

    prev_close = g["close"].shift(1)
    prev2_close = g["close"].shift(2)

    up_past_only = (prev_close > prev2_close).astype(float)
    raw = up_past_only.rolling(window=window, min_periods=window).mean()

    return raw.clip(0.0, 1.0)
