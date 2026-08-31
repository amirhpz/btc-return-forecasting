"""
Feature: cand_up_down_vol_ratio_20
Contract version: 1.0.0
Category: volume_flow
Normalization mode: positive

Purpose:
    Compares volume on up-closing bars with volume on down-closing bars over the latest 20 completed bars.

Formula:
    sum_20(volume[t-1] where close[t-1] > close[t-2]) / (sum_20(volume[t-1] where close[t-1] < close[t-2]) + EPS)

Inputs:
    close, volume

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate greater volume on rising closes than on falling closes.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 19.

Volume contract:
    same_volume_definition_in_training_and_metatrader

Notes:
    Volume definition must be identical in training and live inference.
"""

import pandas as pd
from feature_utils import EPS


NAME = "cand_up_down_vol_ratio_20"
MODE = "positive"
VOLUME_REQUIREMENT = "same_volume_definition_in_training_and_metatrader"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 20

    close_p1 = g["close"].shift(1)
    vol_p1 = g["volume"].shift(1)

    up_vol = (
        vol_p1.where(close_p1 > close_p1.shift(1), 0.0)
        .rolling(window=window, min_periods=window)
        .sum()
    )
    down_vol = (
        vol_p1.where(close_p1 < close_p1.shift(1), 0.0)
        .rolling(window=window, min_periods=window)
        .sum()
    )

    ratio = up_vol / (down_vol + EPS)
    return ratio
