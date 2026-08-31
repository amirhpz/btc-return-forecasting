"""
Feature: tr_to_atr_14
Contract version: 1.0.0
Category: volatility_shock
Normalization mode: positive

Purpose:
    Compares the last completed candle True Range with its 14-bar ATR.

Formula:
    TR[t-1] / (ATR_14_shift1[t] + EPS)

Inputs:
    high, low, close

Parameters:
    atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate the last completed true range exceeded the recent ATR baseline.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr, true_range

NAME = "tr_to_atr_14"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 14
    return true_range(g) / (atr(g, period) + EPS)
