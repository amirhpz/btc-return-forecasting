"""
Feature: dollar_vol_rel_20
Contract version: 1.0.0
Category: liquidity
Normalization mode: positive

Purpose:
    Measures last-bar quote notional volume relative to its 20-span EMA.

Formula:
    (close[t-1] * volume[t-1]) / (EMA_20(close*volume)[t] + EPS)

Inputs:
    close, volume

Parameters:
    ema_span=20, ema_adjust=false

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Values above 1 indicate above-baseline traded notional.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 20.

Volume contract:
    base_asset_traded_volume_consistent_between_train_and_live

Notes:
    Requires base-asset traded volume so close multiplied by volume represents quote notional.
"""

import pandas as pd
from feature_utils import EPS

NAME = "dollar_vol_rel_20"
MODE = "positive"
VOLUME_REQUIREMENT = "base_asset_traded_volume_consistent_between_train_and_live"


def compute(g: pd.DataFrame) -> pd.Series:
    span = 20

    dv = g["volume"].shift(1) * g["close"].shift(1)
    ema_dv = dv.ewm(span=span, adjust=False, min_periods=span).mean()

    return dv / (ema_dv + EPS)
