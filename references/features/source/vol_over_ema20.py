"""
Feature: vol_over_ema20
Contract version: 1.0.0
Category: volume
Normalization mode: positive

Purpose:
    Measures the last completed volume relative to its 20-span EMA.

Formula:
    volume[t-1] / (EMA_20(volume.shift(1))[t] + EPS)

Inputs:
    volume

Parameters:
    ema_span=20, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate above-EMA volume.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    same_volume_definition_in_training_and_metatrader

Notes:
    Volume definition must be identical in training and live inference.
"""

import pandas as pd
from feature_utils import EPS

NAME = "vol_over_ema20"
MODE = "positive"
VOLUME_REQUIREMENT = "same_volume_definition_in_training_and_metatrader"


def compute(g: pd.DataFrame) -> pd.Series:
    span = 20

    past_vol = g["volume"].shift(1)
    ema_v = past_vol.ewm(span=span, adjust=False, min_periods=span).mean()

    return past_vol / (ema_v + EPS)
