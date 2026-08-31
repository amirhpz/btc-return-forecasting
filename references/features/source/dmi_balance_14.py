"""
Feature: dmi_balance_14
Contract version: 1.0.0
Category: trend
Normalization mode: signed

Purpose:
    Computes a normalized directional-movement balance between positive and negative directional indicators.

Formula:
    (DI_plus_14[t] - DI_minus_14[t]) / (DI_plus_14[t] + DI_minus_14[t] + EPS)

Inputs:
    high, low, close

Parameters:
    period=14, dm_smoothing=ewm_alpha_1_over_period, atr_alignment=native_mt5_sma_shift1

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [-1.0, 1.0]
    Interpretation: Positive values favor upward directional movement; negative values favor downward directional movement.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 15.

Volume contract:
    none

Notes:
    This is a documented custom balance feature and is not claimed to equal the native MT5 ADX/DI buffers.
"""

import pandas as pd
from feature_utils import dmi_balance

NAME = "dmi_balance_14"
MODE = "signed"


def compute(g: pd.DataFrame) -> pd.Series:
    period = 14
    raw = dmi_balance(g, period=period)  # expected ~[-1, 1]
    return raw.clip(-1.0, 1.0)
