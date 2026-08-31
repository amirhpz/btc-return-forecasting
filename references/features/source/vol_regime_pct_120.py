"""
Feature: vol_regime_pct_120
Contract version: 1.0.0
Category: volatility_regime
Normalization mode: positive

Purpose:
    Ranks current 20-bar realized volatility against the preceding 120 fully formed volatility observations.

Formula:
    count(past_volatility_120 <= current_volatility) / 120

Inputs:
    close

Parameters:
    volatility_window=20, percentile_window=120, std_ddof=1, tie_policy=weak

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: Values near 1 indicate a high-volatility regime relative to recent history.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 141.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "vol_regime_pct_120"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    volatility_window = 20
    percentile_window = 120

    log_close = np.log(g["close"] + EPS)
    returns = log_close.diff().shift(1)
    volatility = returns.rolling(
        volatility_window, min_periods=volatility_window
    ).std(ddof=1)

    values = volatility.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)

    for i, current in enumerate(values):
        if not np.isfinite(current) or i < percentile_window:
            continue
        past = values[i - percentile_window : i]
        valid = past[np.isfinite(past)]
        if len(valid) != percentile_window:
            continue
        result[i] = np.count_nonzero(valid <= current) / float(len(valid))

    return pd.Series(result, index=g.index)
