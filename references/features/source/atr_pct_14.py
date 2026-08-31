"""
Feature: atr_pct_14
Contract version: 1.0.0
Category: volatility
Normalization mode: positive

Purpose:
    Expresses the native MT5-compatible 14-bar ATR as a fraction of the last completed close.

Formula:
    ATR_14_shift1[t] / (close[t-1] + EPS)

Inputs:
    high, low, close

Parameters:
    atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate greater recent true-range volatility relative to price.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "atr_pct_14"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 14
    return atr(g, period) / (g["close"].shift(1) + EPS)
