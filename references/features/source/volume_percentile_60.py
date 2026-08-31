"""
Feature: volume_percentile_60
Contract version: 1.0.0
Category: volume_regime
Normalization mode: positive

Purpose:
    Ranks the last completed volume against the preceding 60 completed volume observations.

Formula:
    count(volume[t-61:t-1] <= volume[t-1]) / 60

Inputs:
    volume

Parameters:
    window=60, tie_policy=weak

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Values near 1 indicate unusually high volume relative to the preceding 60 bars.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 61.

Volume contract:
    same_volume_definition_in_training_and_metatrader

Notes:
    Volume definition must be identical in training and live inference.
"""

import numpy as np
import pandas as pd

NAME = "volume_percentile_60"
MODE = "positive"
VOLUME_REQUIREMENT = "same_volume_definition_in_training_and_metatrader"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 60
    volume = g["volume"].shift(1).to_numpy(dtype=float)
    result = np.full(len(volume), np.nan, dtype=float)

    for i, current in enumerate(volume):
        if not np.isfinite(current) or i < window:
            continue
        past = volume[i - window : i]
        valid = past[np.isfinite(past)]
        if len(valid) != window:
            continue
        result[i] = np.count_nonzero(valid <= current) / float(len(valid))

    return pd.Series(result, index=g.index)
