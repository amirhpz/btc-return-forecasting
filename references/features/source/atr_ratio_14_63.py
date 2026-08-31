"""
Feature: atr_ratio_14_63
Contract version: 1.0.0
Category: volatility_regime
Normalization mode: positive

Purpose:
    Compares short-horizon ATR with long-horizon ATR to identify volatility regime expansion or contraction.

Formula:
    ATR_14_shift1[t] / (ATR_63_shift1[t] + EPS)

Inputs:
    high, low, close

Parameters:
    fast_atr_period=14, slow_atr_period=63

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate short-term volatility exceeds the longer-term baseline.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 64.

Volume contract:
    none

Notes:
    None.
"""

from __future__ import annotations

import pandas as pd
from feature_utils import EPS, atr


NAME = "atr_ratio_14_63"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    return atr(g, period=14) / (atr(g, period=63) + EPS)
