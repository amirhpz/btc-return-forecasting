"""
Feature: vol_of_vol_ratio_20
Contract version: 1.0.0
Category: volatility_regime
Normalization mode: positive

Purpose:
    Measures variation in the TR-to-ATR ratio relative to its 20-span EMA level.

Formula:
    std_20(TR/ATR, ddof=0) / (EMA_20(TR/ATR) + EPS)

Inputs:
    high, low, close

Parameters:
    atr_period=14, window=20, std_ddof=0, ema_span=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate a less stable short-term volatility regime.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 34.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr, true_range


NAME = "vol_of_vol_ratio_20"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    atr_period = 14
    window = 20

    tr_over_atr = true_range(g) / (atr(g, atr_period) + EPS)

    # true_range/atr are already bar-open aligned; no extra shift here.
    rolling_std = tr_over_atr.rolling(window=window, min_periods=window).std(ddof=0)
    rolling_ema = tr_over_atr.ewm(span=window, adjust=False, min_periods=window).mean()

    return rolling_std / (rolling_ema + EPS)
