"""
Feature: bb_excursion_20_2
Contract version: 1.0.0
Category: breakout
Normalization mode: sparse_signed

Purpose:
    Measures only the signed portion of a completed close that lies outside its 20-period, 2-sigma Bollinger Bands.

Formula:
    min(percent_b, 0) + max(percent_b - 1, 0)

Inputs:
    close

Parameters:
    period=20, standard_deviation_multiplier=2.0, std_ddof=0

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [unbounded, unbounded]
    Interpretation: Zero means price is inside the bands; positive and negative values measure upper- and lower-band excursions in band-width units.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    none

Notes:
    Replaces the legacy percent_b_20 feature to remove a clipped linear duplicate of band_bb_percB_20_2.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "bb_excursion_20_2"
MODE = "sparse_signed"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 20
    multiplier = 2.0

    past_close = g["close"].shift(1)
    middle = past_close.rolling(period, min_periods=period).mean()
    std = past_close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + multiplier * std
    lower = middle - multiplier * std
    width = upper - lower

    percent_b = pd.Series(np.nan, index=g.index, dtype=float)
    valid = width.abs() > EPS
    percent_b.loc[valid] = (
        past_close.loc[valid] - lower.loc[valid]
    ) / width.loc[valid]
    flat = width.notna() & ~valid
    percent_b.loc[flat] = 0.5

    below = percent_b.where(percent_b < 0.0, 0.0)
    above = (percent_b - 1.0).where(percent_b > 1.0, 0.0)
    excursion = below + above
    return excursion.where(percent_b.notna())
