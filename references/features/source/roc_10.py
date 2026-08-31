"""
Feature: roc_10
Contract version: 1.0.0
Category: momentum
Normalization mode: signed

Purpose:
    Computes the 10-bar rate of change of completed closing prices.

Formula:
    close[t-1] / close[t-11] - 1

Inputs:
    close

Parameters:
    lookback=10

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, unbounded]
    Interpretation: Positive values indicate price appreciation over ten bars; negative values indicate depreciation.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 11.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd

NAME = "roc_10"
MODE = "signed"


def compute(g: pd.DataFrame) -> pd.Series:
    close_lag = g["close"].shift(1)
    return close_lag / close_lag.shift(10) - 1.0
