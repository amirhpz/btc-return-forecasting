"""
Feature: band_bb_percB_20_2
Contract version: 1.0.0
Category: price_position
Normalization mode: signed

Purpose:
    Computes unbounded Bollinger Percent-B from completed closes while preserving outside-band magnitude.

Formula:
    (close[t-1] - lower_20_2[t]) / (upper_20_2[t] - lower_20_2[t])

Inputs:
    close

Parameters:
    period=20, standard_deviation_multiplier=2.0, std_ddof=0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Values between 0 and 1 lie inside the bands; values below 0 or above 1 indicate lower- or upper-band excursions.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "band_bb_percB_20_2"
MODE = "signed"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 20
    multiplier = 2.0

    past_close = g["close"].shift(1)
    middle = past_close.rolling(period, min_periods=period).mean()
    std = past_close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + multiplier * std
    lower = middle - multiplier * std
    width = upper - lower

    out = pd.Series(np.nan, index=g.index, dtype=float)
    valid = width.abs() > EPS
    out.loc[valid] = (past_close.loc[valid] - lower.loc[valid]) / width.loc[valid]
    flat = width.notna() & ~valid
    out.loc[flat] = 0.5
    return out
