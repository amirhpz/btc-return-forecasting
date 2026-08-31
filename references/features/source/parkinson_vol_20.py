"""
Feature: parkinson_vol_20
Contract version: 1.0.0
Category: volatility
Normalization mode: positive

Purpose:
    Computes the 20-bar Parkinson high-low volatility estimator from completed candles.

Formula:
    sqrt(mean_20(ln(high/low)^2) / (4*ln(2)))

Inputs:
    high, low

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate greater intrabar high-low volatility.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd
from feature_utils import EPS

NAME = "parkinson_vol_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    log_hl = np.log((g["high"] + EPS) / (g["low"] + EPS))

    term = (log_hl**2).shift(1).rolling(window=20, min_periods=20).mean()

    return np.sqrt(term / (4.0 * np.log(2.0)))
