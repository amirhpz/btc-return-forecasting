"""
Feature: amihud_illiquidity_20
Contract version: 1.0.0
Category: liquidity
Normalization mode: positive

Purpose:
    Computes the 20-bar Amihud illiquidity estimate using absolute log return per unit of quote notional volume.

Formula:
    mean_20(|ln(close[t-1] / close[t-2])| / (close[t-1] * volume[t-1]))

Inputs:
    close, volume

Parameters:
    window=20

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, unbounded]
    Interpretation: Higher values indicate larger price movement per unit of traded notional and therefore lower liquidity.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 21.

Volume contract:
    base_asset_traded_volume_consistent_between_train_and_live

Notes:
    Requires strictly positive base-asset traded volume and consistent volume semantics between training and live inference.
"""

import numpy as np
import pandas as pd

from feature_utils import EPS

NAME = "amihud_illiquidity_20"
MODE = "positive"
VOLUME_REQUIREMENT = "base_asset_traded_volume_consistent_between_train_and_live"


def compute(g: pd.DataFrame) -> pd.Series:
    close_1 = g["close"].shift(1)
    close_2 = g["close"].shift(2)
    absolute_return = np.log(close_1 / close_2).abs()

    quote_notional = close_1 * g["volume"].shift(1)
    one_bar_illiquidity = absolute_return.div(quote_notional.where(quote_notional > EPS))

    return one_bar_illiquidity.rolling(window=20, min_periods=20).mean()
