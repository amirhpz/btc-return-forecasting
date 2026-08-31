"""
Feature: open_gap_atr_14
Contract version: 1.0.0
Category: market_open_state
Normalization mode: sparse_signed

Purpose:
    Measures the current bar opening gap from the previous close in ATR units.

Formula:
    (open[t] - close[t-1]) / (ATR_14_shift1[t] + EPS)

Inputs:
    open, high, low, close

Parameters:
    atr_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses current-bar field(s) available at bar open: open; all remaining inputs are completed-bar values.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Positive values indicate an upward opening gap; negative values indicate a downward gap.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    This is the only feature intentionally allowed to use a current-bar field because open[t] is known at bar open.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "open_gap_atr_14"
MODE = "sparse_signed"


def compute(g: pd.DataFrame, atr_period: int = 14) -> pd.Series:

    gap = g["open"] - g["close"].shift(1)

    atr_values = atr(g, period=atr_period)

    return gap / (atr_values + EPS)
