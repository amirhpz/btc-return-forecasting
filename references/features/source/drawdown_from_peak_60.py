"""
Feature: drawdown_from_peak_60
Contract version: 1.0.0
Category: risk_state
Normalization mode: positive

Purpose:
    Measures the fractional drawdown of the last completed close from the highest completed close in the latest 60 bars.

Formula:
    (max_60(close[t-1]) - close[t-1]) / (max_60(close[t-1]) + EPS)

Inputs:
    close

Parameters:
    window=60

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Zero indicates the rolling peak; larger values indicate a deeper drawdown.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 60.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd
from feature_utils import EPS

NAME = "drawdown_from_peak_60"
MODE = "positive"


def compute(g: pd.DataFrame, window: int = 60) -> pd.Series:
    close_lagged = g["close"].shift(1)

    rolling_peak = close_lagged.rolling(window=window, min_periods=window).max()

    drawdown = (rolling_peak - close_lagged) / (rolling_peak + EPS)

    return drawdown.clip(lower=0.0)
