"""
Feature: bearish_engulf_score
Contract version: 1.0.0
Category: candlestick_pattern
Normalization mode: sparse_positive

Purpose:
    Scores a completed bearish engulfing pattern by its body size relative to ATR.

Formula:
    I[bearish_engulfing(t-2,t-1)] * |close[t-1]-open[t-1]| / (ATR_14_shift1[t] + EPS)

Inputs:
    open, high, low, close

Parameters:
    atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Zero means no bearish engulfing event; larger values indicate a stronger ATR-scaled pattern.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "bearish_engulf_score"
MODE = "sparse_positive"


def compute(g: pd.DataFrame) -> pd.Series:
    atr_period = 14

    curr_open = g["open"].shift(1)
    curr_close = g["close"].shift(1)
    prev_open = g["open"].shift(2)
    prev_close = g["close"].shift(2)

    is_engulf = (
        (prev_close > prev_open)  # t-2 bullish
        & (curr_close < curr_open)  # t-1 bearish
        & (curr_open >= prev_close)  # t-1 open engulfs t-2 close
        & (curr_close <= prev_open)  # t-1 close engulfs t-2 open
    )

    body = (curr_close - curr_open).abs()
    atr_14 = atr(g, period=atr_period)

    return is_engulf.astype(float) * body / (atr_14 + EPS)
