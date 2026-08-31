"""
Feature: vol_over_median20
Contract version: 1.0.0
Category: volume
Normalization mode: positive

Purpose:
    Measures the last completed volume relative to the rolling median of the latest 20 completed bars.

Formula:
    volume[t-1] / (median_20(volume.shift(1))[t] + EPS)

Inputs:
    volume

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate volume above its recent median.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    same_volume_definition_in_training_and_metatrader

Notes:
    Volume definition must be identical in training and live inference.
"""

import pandas as pd
from feature_utils import EPS

NAME = "vol_over_median20"
MODE = "positive"
VOLUME_REQUIREMENT = "same_volume_definition_in_training_and_metatrader"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 20

    past_vol = g["volume"].shift(1)
    median_v = past_vol.rolling(window=window, min_periods=window).median()

    return past_vol / (median_v + EPS)
