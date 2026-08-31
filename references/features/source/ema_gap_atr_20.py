"""
Feature: ema_gap_atr_20
Contract version: 1.0.0
Category: trend
Normalization mode: signed

Purpose:
    Measures the distance between the last completed close and its 20-span EMA in ATR units.

Formula:
    (close[t-1] - EMA_20(close.shift(1))[t]) / (ATR_14_shift1[t] + EPS)

Inputs:
    close, high, low

Parameters:
    ema_span=20, atr_period=14, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Positive values place price above the EMA; negative values place it below.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import EPS, atr

NAME = "ema_gap_atr_20"
MODE = "signed"


def ema(series: pd.Series, span: int) -> pd.Series:
    # Causal EMA; keep warm-up NaNs.
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute(g: pd.DataFrame) -> pd.Series:
    close_lag = g["close"].shift(1)
    e20 = ema(close_lag, span=20)
    a14 = atr(g, period=14)  # atr already uses shift(1) internally

    raw = (close_lag - e20) / (a14 + EPS)
    return raw
