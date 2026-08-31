"""
Feature: rsi_div_persistence
Contract version: 1.0.0
Category: divergence
Normalization mode: sparse_signed

Purpose:
    Counts the signed consecutive duration of the current hidden RSI divergence state.

Formula:
    signed_run_length(hidden_rsi_divergence_flag)

Inputs:
    high, low, close, datetime

Parameters:
    rsi_period=14, pivot_k=3, maximum_pivot_age=60, minimum_pivot_separation=5, trend_filter=true, trend_length=50

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Positive integers count persistent bullish hidden divergence; negative integers count bearish persistence; zero means no active state.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 0.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

from feature_utils import RSIDivParams, get_rsi_hidden_div_flags

NAME = "rsi_div_persistence"
MODE = "sparse_signed"


def compute(g: pd.DataFrame) -> pd.Series:
    flag, work = get_rsi_hidden_div_flags(g, RSIDivParams())
    persistence = np.zeros(len(flag), dtype=float)

    for t, direction in enumerate(flag):
        if direction == 0:
            continue
        if t > 0 and np.sign(persistence[t - 1]) == direction:
            persistence[t] = direction * (abs(persistence[t - 1]) + 1.0)
        else:
            persistence[t] = float(direction)

    result = pd.Series(
        persistence,
        index=work["_orig_idx"].to_numpy(),
        name=NAME,
    )
    return result.sort_index()
