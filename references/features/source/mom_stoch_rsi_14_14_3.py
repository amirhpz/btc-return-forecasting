"""
Feature: mom_stoch_rsi_14_14_3
Contract version: 1.0.0
Category: momentum
Normalization mode: positive

Purpose:
    Computes a 3-bar smoothed Stochastic RSI from a 14-period Wilder RSI and a 14-period stochastic window.

Formula:
    SMA_3((RSI_14[t] - min_14(RSI_14)) / (max_14(RSI_14) - min_14(RSI_14)))

Inputs:
    close

Parameters:
    rsi_period=14, stochastic_period=14, smoothing_period=3

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Values near 1 indicate RSI near its recent maximum; values near 0 indicate RSI near its recent minimum.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 30.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS, wilder_rsi

NAME = "mom_stoch_rsi_14_14_3"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    rsi_period = 14
    stoch_period = 14
    smooth_period = 3

    rsi_value = wilder_rsi(g["close"].shift(1), rsi_period)
    rsi_min = rsi_value.rolling(stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi_value.rolling(stoch_period, min_periods=stoch_period).max()
    spread = rsi_max - rsi_min

    stoch = pd.Series(np.nan, index=g.index, dtype=float)
    valid = spread > EPS
    stoch.loc[valid] = (
        rsi_value.loc[valid] - rsi_min.loc[valid]
    ) / spread.loc[valid]
    flat = spread.notna() & ~valid
    stoch.loc[flat] = 0.5

    return stoch.rolling(smooth_period, min_periods=smooth_period).mean().clip(0.0, 1.0)
