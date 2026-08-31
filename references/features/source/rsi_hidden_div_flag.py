"""
Feature: rsi_hidden_div_flag
Contract version: 1.0.0
Category: divergence
Normalization mode: ternary

Purpose:
    Emits a ternary hidden RSI divergence state from confirmed price pivots and a trend filter.

Formula:
    1 for bullish hidden divergence, -1 for bearish hidden divergence, otherwise 0

Inputs:
    high, low, close, datetime

Parameters:
    rsi_period=14, pivot_k=3, maximum_pivot_age=60, minimum_pivot_separation=5, trend_filter=true, trend_length=50

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, 1.0]
    Interpretation: One is bullish hidden divergence, minus one is bearish hidden divergence, and zero is neutral.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 0.

Volume contract:
    none

Notes:
    None.
"""

import pandas as pd
from feature_utils import RSIDivParams, get_rsi_hidden_div_flags

NAME = "rsi_hidden_div_flag"
MODE = "ternary"


def compute(g: pd.DataFrame) -> pd.Series:
    p = RSIDivParams()

    flag, g_work = get_rsi_hidden_div_flags(g, p)

    out_series = pd.Series(flag, index=g_work["_orig_idx"].to_numpy(), name=NAME)
    return out_series.sort_index()
