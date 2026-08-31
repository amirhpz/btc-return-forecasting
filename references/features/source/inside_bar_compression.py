"""
Feature: inside_bar_compression
Contract version: 1.0.0
Category: candlestick_pattern
Normalization mode: sparse_positive

Purpose:
    Scores how tightly the last completed inside bar is compressed within its mother bar.

Formula:
    I[high[t-1] <= high[t-2] and low[t-1] >= low[t-2]] * (1 - range[t-1]/range[t-2])

Inputs:
    high, low

Parameters:
    none

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Zero means no valid inside bar; larger values indicate tighter compression.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 0.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "inside_bar_compression"
MODE = "sparse_positive"


def compute(g: pd.DataFrame) -> pd.Series:
    child_high = g["high"].shift(1)
    child_low = g["low"].shift(1)
    mother_high = g["high"].shift(2)
    mother_low = g["low"].shift(2)

    inside = (child_high <= mother_high) & (child_low >= mother_low)
    child_range = child_high - child_low
    mother_range = mother_high - mother_low

    ratio = child_range.div(mother_range.where(mother_range > EPS))
    compression = (1.0 - ratio).clip(lower=0.0, upper=1.0)
    return compression.where(inside & ratio.notna(), 0.0).astype(float)
