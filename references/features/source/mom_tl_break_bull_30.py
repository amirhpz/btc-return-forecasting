"""
Feature: mom_tl_break_bull_30
Contract version: 1.0.0
Category: breakout
Normalization mode: binary

Purpose:
    Emits a bullish event when the last completed close crosses above a descending resistance line formed by the latest two confirmed pivot highs.

Formula:
    I[close[t-2] <= projected_line[t-2] and close[t-1] > projected_line[t-1]]

Inputs:
    high, close, datetime

Parameters:
    pivot_left_bars=3, pivot_right_bars=1, maximum_pivot_age=30, minimum_pivot_separation=3

Timing and causality:
    Output row t is intended for inference at the open of bar t. Uses completed bars only; no current-bar OHLCV values are consumed.
    The implementation is prefix-causal and future-perturbation invariant.

Output:
    Native range: [0.0, 1.0]
    Interpretation: One marks a confirmed bullish trendline breakout; zero means no event.

Warm-up:
    Expected leading warm-up rows on continuous finite data: 0.

Volume contract:
    none

Notes:
    None.
"""

import numpy as np
import pandas as pd

NAME = "mom_tl_break_bull_30"
MODE = "binary"


def _confirmed_pivot_highs(
    high: np.ndarray,
    left: int,
    right: int,
) -> list[list[int]]:
    n = len(high)
    confirmed_at: list[list[int]] = [[] for _ in range(n)]

    for confirmation_bar in range(left + right, n):
        pivot = confirmation_bar - right
        start = pivot - left
        stop = confirmation_bar + 1
        window = high[start:stop]
        center = high[pivot]
        if not np.isfinite(center) or not np.isfinite(window).all():
            continue

        neighbours = np.concatenate((window[:left], window[left + 1 :]))
        if neighbours.size and center > np.max(neighbours):
            confirmed_at[confirmation_bar].append(pivot)

    return confirmed_at


def compute(g: pd.DataFrame) -> pd.Series:
    left = 3
    right = 1
    max_pivot_age = 30
    min_pivot_separation = 3

    time_col = "datetime" if "datetime" in g.columns else None
    if time_col is None and "datetime_utc" in g.columns:
        time_col = "datetime_utc"

    work = g.sort_values(time_col) if time_col is not None else g.copy()
    work = work.copy()
    work["_orig_idx"] = work.index
    work = work.reset_index(drop=True)

    high = work["high"].to_numpy(dtype=float)
    close = work["close"].to_numpy(dtype=float)
    n = len(work)
    confirmed_at = _confirmed_pivot_highs(high, left, right)

    output = np.zeros(n, dtype=np.int8)
    active_pivots: list[int] = []

    # signal_bar s is candle t-1; its event belongs to output row t=s+1.
    for signal_bar in range(n - 1):
        active_pivots.extend(confirmed_at[signal_bar])
        oldest_allowed = signal_bar - max_pivot_age
        active_pivots = [p for p in active_pivots if p >= oldest_allowed]

        if len(active_pivots) < 2 or signal_bar < 1:
            continue

        first, second = active_pivots[-2], active_pivots[-1]
        if second - first < min_pivot_separation or signal_bar <= second:
            continue

        high_first = high[first]
        high_second = high[second]
        if not (high_second < high_first):
            continue

        slope = (high_second - high_first) / float(second - first)
        line_previous = high_first + slope * ((signal_bar - 1) - first)
        line_current = high_first + slope * (signal_bar - first)

        if close[signal_bar - 1] <= line_previous and close[signal_bar] > line_current:
            output[signal_bar + 1] = 1

    result = pd.Series(output, index=work["_orig_idx"].to_numpy(), name=NAME)
    return result.sort_index()
