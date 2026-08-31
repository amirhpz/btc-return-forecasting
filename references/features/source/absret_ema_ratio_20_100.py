"""
Feature: absret_ema_ratio_20_100
Contract version: 1.0.0
Category: volatility_regime
Normalization mode: positive

Purpose:
    Measures short-term volatility acceleration as the ratio of fast and slow EMAs of absolute log returns.

Formula:
    EMA_20(|ln(close[t-1] / close[t-2])|) / (EMA_100(|ln(close[t-1] / close[t-2])|) + EPS)

Inputs:
    close

Parameters:
    fast_span=20, slow_span=100, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate recent absolute-return volatility is elevated relative to its longer-term baseline.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 101.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd
from feature_utils import EPS

NAME = "absret_ema_ratio_20_100"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:

    ret_abs = np.log((g["close"].shift(1)) / (g["close"].shift(2))).abs()

    ema_fast = ret_abs.ewm(span=20, adjust=False, min_periods=20).mean()

    ema_slow = ret_abs.ewm(span=100, adjust=False, min_periods=100).mean()

    return ema_fast / (ema_slow + EPS)
