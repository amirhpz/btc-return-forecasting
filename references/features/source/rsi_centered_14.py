"""
Feature: rsi_centered_14
Contract version: 1.0.0
Category: momentum
Normalization mode: signed

Purpose:
    Centers the 14-period Wilder RSI around zero for signed model input.

Formula:
    RSI_14(close.shift(1))[t] / 100 - 0.5

Inputs:
    close

Parameters:
    rsi_period=14

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-0.5, 0.5]
    Interpretation: Positive values indicate RSI above 50; negative values indicate RSI below 50.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    None.
"""

from __future__ import annotations

import pandas as pd
from feature_utils import rsi

NAME = "rsi_centered_14"
MODE = "signed"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 14
    raw_rsi = rsi(g["close"].shift(1), period=period)  # expected in [0, 100]
    out = raw_rsi / 100.0 - 0.5  # expected in [-0.5, 0.5]
    return out
