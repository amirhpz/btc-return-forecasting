"""
Feature: vol_ratio_20_100
Contract version: 1.0.0
Category: volume_regime
Normalization mode: positive

Purpose:
    Compares the 20-span EMA of completed volume with its 100-span EMA.

Formula:
    EMA_20(volume.shift(1))[t] / (EMA_100(volume.shift(1))[t] + EPS)

Inputs:
    volume

Parameters:
    fast_ema_span=20, slow_ema_span=100, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate short-term volume is elevated relative to the long-term baseline.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 100.

Volume contract:
    same_volume_definition_in_training_and_metatrader

Notes:
    Volume definition must be identical in training and live inference.
"""

import pandas as pd
from feature_utils import EPS


NAME = "vol_ratio_20_100"
MODE = "positive"
VOLUME_REQUIREMENT = "same_volume_definition_in_training_and_metatrader"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute(g: pd.DataFrame) -> pd.Series:
    volume_lag = g["volume"].shift(1)

    v20 = ema(volume_lag, span=20)
    v100 = ema(volume_lag, span=100)

    return v20 / (v100 + EPS)
