"""
Feature: realized_vol_20
Contract version: 1.0.0
Category: volatility
Normalization mode: positive

Purpose:
    Computes the population standard deviation of 20 completed one-bar log returns.

Formula:
    std_20(ln(close[t-1]/close[t-2]), ddof=0)

Inputs:
    close

Parameters:
    window=20, std_ddof=0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate greater close-to-close return volatility.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

NAME = "realized_vol_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 20

    log_ret = np.log(g["close"] / g["close"].shift(1))
    # Use shifted returns so window is strictly prior bars.
    return log_ret.shift(1).rolling(window=period, min_periods=period).std(ddof=0)
