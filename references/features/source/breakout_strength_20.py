"""
Feature: breakout_strength_20
Contract version: 1.0.0
Category: breakout
Normalization mode: sparse_positive

Purpose:
    Measures how far the last completed close broke above the preceding 20-bar high, scaled by ATR.

Formula:
    max(close[t-1] - max(high[t-21:t-1]), 0) / (ATR_14_shift1[t] + EPS)

Inputs:
    high, close, low

Parameters:
    lookback=20, atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Zero means no upside breakout; larger values indicate a stronger ATR-scaled breakout.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "breakout_strength_20"
MODE = "sparse_positive"


def compute(g: pd.DataFrame) -> pd.Series:
    lookback = 20
    atr_period = 14

    prev_high = g["high"].rolling(window=lookback, min_periods=lookback).max().shift(2)

    return (g["close"].shift(1) - prev_high).clip(lower=0.0) / (
        atr(g, atr_period) + EPS
    )
