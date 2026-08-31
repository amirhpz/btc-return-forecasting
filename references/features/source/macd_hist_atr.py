"""
Feature: macd_hist_atr
Contract version: 1.0.0
Category: momentum
Normalization mode: signed

Purpose:
    Scales the standard 12/26/9 MACD histogram of completed closes by ATR.

Formula:
    (MACD_12_26[t] - EMA_9(MACD_12_26)[t]) / (ATR_14_shift1[t] + EPS)

Inputs:
    close, high, low

Parameters:
    fast_ema_span=12, slow_ema_span=26, signal_ema_span=9, atr_period=14, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Positive values indicate bullish momentum acceleration; negative values indicate bearish acceleration.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 34.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "macd_hist_atr"
MODE = "signed"


def ema(series: pd.Series, span: int) -> pd.Series:
    # Causal EMA with warm-up NaNs
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute(g: pd.DataFrame) -> pd.Series:
    close_lag = g["close"].shift(1)

    e12 = ema(close_lag, span=12)
    e26 = ema(close_lag, span=26)
    macd_line = e12 - e26

    signal9 = ema(macd_line, span=9)
    hist = macd_line - signal9  # MACD histogram

    a14 = atr(g, period=14)
    raw = hist / (a14 + EPS)
    return raw
