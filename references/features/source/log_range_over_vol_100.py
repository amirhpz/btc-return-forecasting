"""
Feature: log_range_over_vol_100
Contract version: 1.0.0
Category: trend_quality
Normalization mode: positive

Purpose:
    Compares the 100-bar log-price range with the standard deviation of one-bar log returns.

Formula:
    (max_100(log_close) - min_100(log_close)) / (std_100(log_return, ddof=1) + EPS)

Inputs:
    close

Parameters:
    window=100, std_ddof=1

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate a large directional range relative to local return noise.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 101.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd
from feature_utils import EPS

NAME = "log_range_over_vol_100"
MODE = "positive"


def compute(g: pd.DataFrame) -> pd.Series:
    window = 100

    log_c = np.log(g["close"] + EPS)
    ret1 = log_c.diff(1)

    past_log_c = log_c.shift(1)
    past_ret1 = ret1.shift(1)

    max_log = past_log_c.rolling(window=window, min_periods=window).max()
    min_log = past_log_c.rolling(window=window, min_periods=window).min()
    log_range = max_log - min_log

    ret_vol = past_ret1.rolling(window=window, min_periods=window).std(ddof=1)

    return log_range / (ret_vol + EPS)
