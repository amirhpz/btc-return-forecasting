"""
Feature: ema_slope_atr_20_5
Contract version: 1.0.0
Category: trend
Normalization mode: signed

Purpose:
    Measures the five-bar slope of the 20-span EMA in average per-bar ATR units.

Formula:
    (EMA_20[t] - EMA_20[t-5]) / (5 * ATR_14_shift1[t] + EPS)

Inputs:
    close, high, low

Parameters:
    ema_span=20, slope_lag=5, atr_period=14, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Positive values indicate an upward EMA slope; negative values indicate a downward slope.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 25.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "ema_slope_atr_20_5"
MODE = "signed"


def ema(series: pd.Series, span: int) -> pd.Series:
    # Causal EMA with warm-up NaNs
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute(g: pd.DataFrame) -> pd.Series:
    close_lag = g["close"].shift(1)
    e20 = ema(close_lag, span=20)
    a14 = atr(g, period=14)

    raw = (e20 - e20.shift(5)) / (5.0 * a14 + EPS)
    return raw
