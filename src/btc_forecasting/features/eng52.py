from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import numpy as np
import pandas as pd

EPS: Final = 1e-12
ONE_HOUR: Final = timedelta(hours=1)
REQUIRED_COLUMNS: Final = ("open_time", "open", "high", "low", "close", "volume")

ENG52_FEATURE_NAMES: Final = (
    "absret_ema_ratio_20_100", "amihud_illiquidity_20", "atr_pct_14",
    "atr_ratio_14_63", "band_bb_percB_20_2", "bb_excursion_20_2",
    "bb_width_rel_20", "bearish_engulf_score", "body_signed_to_tr", "body_to_tr",
    "breakdown_strength_20", "breakout_strength_20", "bullish_engulf_score",
    "cand_up_down_vol_ratio_20", "channel_pos_20", "close_location_value",
    "dmi_balance_14", "dollar_vol_rel_20", "downside_semivol_20",
    "drawdown_from_peak_60", "efficiency_ratio_20", "ema_gap_atr_20",
    "ema_slope_atr_20_5", "inside_bar_compression", "log_range_over_vol_100",
    "lower_wick_to_tr", "macd_hist_atr", "mom_stoch_rsi_14_14_3",
    "mom_tl_break_bull_30", "open_gap_atr_14", "outside_bar_expansion",
    "parkinson_vol_20", "range_compression_20_100", "realized_vol_20",
    "ret_autocorr_1_30", "ret_vol_corr_30", "return_skew_30", "roc_10",
    "rsi_centered_14", "rsi_div_persistence", "rsi_hidden_div_flag",
    "sign_flip_rate_20", "tr_to_atr_14", "up_close_ratio_5", "upper_wick_to_tr",
    "upside_semivol_20", "vol_of_vol_ratio_20", "vol_over_ema20",
    "vol_over_median20", "vol_ratio_20_100", "vol_regime_pct_120",
    "volume_percentile_60",
)

_FIRST_VALID_SEGMENT_ROW: Final = dict(zip(ENG52_FEATURE_NAMES, (
    101, 21, 15, 64, 20, 20, 20, 15, 1, 1, 21, 21, 15, 21, 20, 1, 15,
    20, 21, 60, 21, 20, 25, 2, 101, 1, 34, 30, 0, 15, 15, 20, 100, 21,
    32, 31, 31, 11, 15, 0, 0, 22, 15, 6, 1, 21, 34, 20, 20, 100, 141, 61,
), strict=True))


@dataclass(frozen=True)
class _RsiDivParams:
    rsi_period: int = 14
    pivot_k: int = 3
    max_pivot_age: int = 60
    min_pivot_sep: int = 5
    trend_filter: bool = True
    trend_len: int = 50


def _validated_input(hourly: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(hourly.columns))
    if missing:
        raise ValueError(f"Missing required ENG52 columns: {missing}")
    open_time = pd.to_datetime(hourly["open_time"], utc=True, errors="coerce")
    if open_time.isna().any():
        raise ValueError("open_time must contain valid UTC timestamps")
    if open_time.duplicated().any() or not open_time.is_monotonic_increasing:
        raise ValueError("open_time must be strictly ordered and unique")
    if not open_time.dt.floor("h").eq(open_time).all():
        raise ValueError("open_time must lie on UTC hour boundaries")
    numeric = hourly[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("ENG52 source columns must contain finite numeric values")
    if (numeric[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("ENG52 price columns must be strictly positive")
    if (numeric["volume"] < 0.0).any():
        raise ValueError("ENG52 volume must be non-negative")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("ENG52 high must be at least open, close, and low")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("ENG52 low must be at most open, close, and high")
    numeric.insert(0, "open_time", open_time)
    return numeric.reset_index(drop=True)


def _bar_true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat((
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ), axis=1).max(axis=1)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    return _bar_true_range(frame).shift(1)


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    by_bar = _bar_true_range(frame).rolling(period, min_periods=period).mean()
    by_bar.iloc[:period] = np.nan
    return by_bar.shift(1)


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    average_gain = np.full(len(values), np.nan)
    average_loss = np.full(len(values), np.nan)
    delta = np.full(len(values), np.nan)
    finite_pair = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    delta[1:][finite_pair] = values[1:][finite_pair] - values[:-1][finite_pair]
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)
    last_seed_or_value = -1
    for index in range(1, len(values)):
        if (last_seed_or_value == index - 1 and np.isfinite(average_gain[index - 1])
                and np.isfinite(delta[index])):
            average_gain[index] = (average_gain[index - 1] * (period - 1) + gain[index]) / period
            average_loss[index] = (average_loss[index - 1] * (period - 1) + loss[index]) / period
        elif index >= period:
            gain_window = gain[index - period + 1:index + 1]
            loss_window = loss[index - period + 1:index + 1]
            if not (np.isfinite(gain_window).all() and np.isfinite(loss_window).all()):
                continue
            average_gain[index] = gain_window.mean()
            average_loss[index] = loss_window.mean()
        else:
            continue
        last_seed_or_value = index
        if average_loss[index] > EPS:
            rs = average_gain[index] / average_loss[index]
            output[index] = 100.0 - 100.0 / (1.0 + rs)
        elif average_gain[index] > EPS:
            output[index] = 100.0
        else:
            output[index] = 50.0
    return pd.Series(output, index=close.index, dtype=float)


def _dmi_balance(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_lag, low_lag = frame["high"].shift(1), frame["low"].shift(1)
    up_move, down_move = high_lag.diff(), -low_lag.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    atr_value = _atr(frame, period)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr_value + EPS)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr_value + EPS)
    return (plus_di - minus_di) / (plus_di + minus_di + EPS)


def _find_pivots(high: pd.Series, low: pd.Series, pivot_k: int, min_sep: int) -> tuple[list[int], list[int]]:
    window = 2 * pivot_k + 1
    raw_low = low == low.rolling(window, center=True, min_periods=window).min()
    raw_high = high == high.rolling(window, center=True, min_periods=window).max()
    def filtered(mask: pd.Series) -> list[int]:
        kept: list[int] = []
        last = -(10**9)
        for value in np.flatnonzero(mask.to_numpy()):
            index = int(value)
            if index - last >= min_sep:
                kept.append(index)
                last = index
        return kept
    return filtered(raw_low), filtered(raw_high)


def _rsi_hidden_div_flags(frame: pd.DataFrame) -> np.ndarray:
    p = _RsiDivParams()
    close_lag, high, low = frame["close"].shift(1), frame["high"].shift(1), frame["low"].shift(1)
    rsi = _wilder_rsi(close_lag, p.rsi_period).to_numpy()
    pivot_lows, pivot_highs = _find_pivots(high, low, p.pivot_k, p.min_pivot_sep)
    size = len(frame)
    def latest_two(pivots: list[int]) -> tuple[np.ndarray, np.ndarray]:
        latest = previous = -(10**9)
        arr_latest = np.full(size, -(10**9), dtype=int)
        arr_previous = np.full(size, -(10**9), dtype=int)
        pointer = 0
        for row in range(size):
            while pointer < len(pivots) and pivots[pointer] <= row - p.pivot_k:
                previous, latest = latest, pivots[pointer]
                pointer += 1
            if latest < row - p.max_pivot_age:
                latest = -(10**9)
            if previous < row - p.max_pivot_age:
                previous = -(10**9)
            arr_latest[row], arr_previous[row] = latest, previous
        return arr_latest, arr_previous
    latest_low, previous_low = latest_two(pivot_lows)
    latest_high, previous_high = latest_two(pivot_highs)
    sma = close_lag.rolling(p.trend_len, min_periods=p.trend_len).mean().to_numpy()
    close_values = close_lag.to_numpy()
    trend_bull, trend_bear = close_values > sma, close_values < sma
    trend_bull[np.isnan(sma)] = False
    trend_bear[np.isnan(sma)] = False
    flags = np.zeros(size, dtype=np.int8)
    for row in range(size):
        i2, i1 = latest_low[row], previous_low[row]
        j2, j1 = latest_high[row], previous_high[row]
        bullish = i1 >= 0 and i2 >= 0 and low.iloc[i2] > low.iloc[i1] and rsi[i2] < rsi[i1] and trend_bull[row]
        bearish = j1 >= 0 and j2 >= 0 and high.iloc[j2] < high.iloc[j1] and rsi[j2] > rsi[j1] and trend_bear[row]
        if bullish and not bearish:
            flags[row] = 1
        elif bearish and not bullish:
            flags[row] = -1
        elif bullish and bearish:
            if i2 > j2:
                flags[row] = 1
            elif j2 > i2:
                flags[row] = -1
    return flags


def _confirmed_pivot_highs(high: np.ndarray, left: int, right: int) -> list[list[int]]:
    confirmed: list[list[int]] = [[] for _ in range(len(high))]
    for confirmation in range(left + right, len(high)):
        pivot = confirmation - right
        window = high[pivot - left:confirmation + 1]
        center = high[pivot]
        if not np.isfinite(center) or not np.isfinite(window).all():
            continue
        neighbours = np.concatenate((window[:left], window[left + 1:]))
        if neighbours.size and center > np.max(neighbours):
            confirmed[confirmation].append(pivot)
    return confirmed


def _trendline_breakout(frame: pd.DataFrame) -> pd.Series:
    high, close = frame["high"].to_numpy(float), frame["close"].to_numpy(float)
    confirmed = _confirmed_pivot_highs(high, 3, 1)
    output = np.zeros(len(frame), dtype=np.int8)
    active: list[int] = []
    for signal_bar in range(len(frame) - 1):
        active.extend(confirmed[signal_bar])
        active = [pivot for pivot in active if pivot >= signal_bar - 30]
        if len(active) < 2 or signal_bar < 1:
            continue
        first, second = active[-2], active[-1]
        if second - first < 3 or signal_bar <= second or not high[second] < high[first]:
            continue
        slope = (high[second] - high[first]) / float(second - first)
        previous_line = high[first] + slope * ((signal_bar - 1) - first)
        current_line = high[first] + slope * (signal_bar - first)
        if close[signal_bar - 1] <= previous_line and close[signal_bar] > current_line:
            output[signal_bar + 1] = 1
    return pd.Series(output, index=frame.index, dtype=float)


def _percentile_of_previous(values: pd.Series, window: int) -> pd.Series:
    array = values.to_numpy(float)
    result = np.full(len(array), np.nan)
    for index, current in enumerate(array):
        if not np.isfinite(current) or index < window:
            continue
        previous = array[index - window:index]
        finite = previous[np.isfinite(previous)]
        if len(finite) == window:
            result[index] = np.count_nonzero(finite <= current) / float(window)
    return pd.Series(result, index=values.index)


def _compute_segment(frame: pd.DataFrame) -> pd.DataFrame:
    feature: dict[str, pd.Series] = {}
    open_price, high, low = frame["open"], frame["high"], frame["low"]
    close, volume = frame["close"], frame["volume"]
    close_1, close_2 = close.shift(1), close.shift(2)
    return_1 = np.log(close_1 / close_2)
    absolute_return = return_1.abs()
    true_range, atr_14 = _true_range(frame), _atr(frame, 14)

    feature["absret_ema_ratio_20_100"] = absolute_return.ewm(
        span=20, adjust=False, min_periods=20
    ).mean() / (
        absolute_return.ewm(span=100, adjust=False, min_periods=100).mean() + EPS
    )
    quote_notional = close_1 * volume.shift(1)
    illiquidity = absolute_return.div(quote_notional.where(quote_notional > EPS))
    feature["amihud_illiquidity_20"] = illiquidity.rolling(20, min_periods=20).mean()
    feature["atr_pct_14"] = atr_14 / (close_1 + EPS)
    feature["atr_ratio_14_63"] = atr_14 / (_atr(frame, 63) + EPS)

    middle = close_1.rolling(20, min_periods=20).mean()
    bb_std = close_1.rolling(20, min_periods=20).std(ddof=0)
    upper, lower = middle + 2.0 * bb_std, middle - 2.0 * bb_std
    width = upper - lower
    percent_b = pd.Series(np.nan, index=frame.index, dtype=float)
    valid_width = width.abs() > EPS
    percent_b.loc[valid_width] = (
        close_1.loc[valid_width] - lower.loc[valid_width]
    ) / width.loc[valid_width]
    percent_b.loc[width.notna() & ~valid_width] = 0.5
    feature["band_bb_percB_20_2"] = percent_b
    below = percent_b.where(percent_b < 0.0, 0.0)
    above = (percent_b - 1.0).where(percent_b > 1.0, 0.0)
    feature["bb_excursion_20_2"] = (below + above).where(percent_b.notna())
    feature["bb_width_rel_20"] = (4.0 * bb_std).div(
        middle.where(middle.abs() > EPS)
    )

    current_open, previous_open = open_price.shift(1), open_price.shift(2)
    previous_close = close.shift(2)
    bearish = (
        (previous_close > previous_open)
        & (close_1 < current_open)
        & (current_open >= previous_close)
        & (close_1 <= previous_open)
    )
    bullish = (
        (previous_close < previous_open)
        & (close_1 > current_open)
        & (current_open <= previous_close)
        & (close_1 >= previous_open)
    )
    body = (close_1 - current_open).abs()
    feature["bearish_engulf_score"] = bearish.astype(float) * body / (atr_14 + EPS)
    feature["body_signed_to_tr"] = (
        (close_1 - current_open) / (true_range + EPS)
    ).clip(-1.0, 1.0)
    feature["body_to_tr"] = body / (true_range + EPS)
    previous_low = low.rolling(20, min_periods=20).min().shift(2)
    previous_high = high.rolling(20, min_periods=20).max().shift(2)
    feature["breakdown_strength_20"] = (previous_low - close_1).clip(
        lower=0.0
    ) / (atr_14 + EPS)
    feature["breakout_strength_20"] = (close_1 - previous_high).clip(
        lower=0.0
    ) / (atr_14 + EPS)
    feature["bullish_engulf_score"] = bullish.astype(float) * body / (atr_14 + EPS)

    past_volume = volume.shift(1)
    up_volume = past_volume.where(close_1 > close_1.shift(1), 0.0).rolling(
        20, min_periods=20
    ).sum()
    down_volume = past_volume.where(close_1 < close_1.shift(1), 0.0).rolling(
        20, min_periods=20
    ).sum()
    feature["cand_up_down_vol_ratio_20"] = up_volume / (down_volume + EPS)
    past_high, past_low = high.shift(1), low.shift(1)
    channel_high = past_high.rolling(20, min_periods=20).max()
    channel_low = past_low.rolling(20, min_periods=20).min()
    feature["channel_pos_20"] = (
        (close_1 - channel_low) / (channel_high - channel_low + EPS)
    ).clip(0.0, 1.0)

    candle_range = past_high - past_low
    close_location = pd.Series(np.nan, index=frame.index, dtype=float)
    valid_range = candle_range > EPS
    close_location.loc[valid_range] = (
        (close_1.loc[valid_range] - past_low.loc[valid_range])
        - (past_high.loc[valid_range] - close_1.loc[valid_range])
    ) / candle_range.loc[valid_range]
    close_location.loc[candle_range.notna() & ~valid_range] = 0.0
    feature["close_location_value"] = close_location.clip(-1.0, 1.0)
    feature["dmi_balance_14"] = _dmi_balance(frame, 14).clip(-1.0, 1.0)

    dollar_volume = past_volume * close_1
    feature["dollar_vol_rel_20"] = dollar_volume / (
        dollar_volume.ewm(span=20, adjust=False, min_periods=20).mean() + EPS
    )
    downside = return_1.clip(upper=0.0)
    feature["downside_semivol_20"] = np.sqrt(
        downside.pow(2).rolling(20, min_periods=20).mean()
    )
    rolling_peak = close_1.rolling(60, min_periods=60).max()
    feature["drawdown_from_peak_60"] = (
        (rolling_peak - close_1) / (rolling_peak + EPS)
    ).clip(lower=0.0)
    net_move = (close_1 - close_1.shift(20)).abs()
    path_length = close_1.diff().abs().rolling(20, min_periods=20).sum()
    feature["efficiency_ratio_20"] = (net_move / (path_length + EPS)).clip(
        0.0, 1.0
    )

    ema_20 = close_1.ewm(span=20, adjust=False, min_periods=20).mean()
    feature["ema_gap_atr_20"] = (close_1 - ema_20) / (atr_14 + EPS)
    feature["ema_slope_atr_20_5"] = (ema_20 - ema_20.shift(5)) / (
        5.0 * atr_14 + EPS
    )
    mother_high, mother_low = high.shift(2), low.shift(2)
    inside = (past_high <= mother_high) & (past_low >= mother_low)
    mother_range = mother_high - mother_low
    inside_ratio = candle_range.div(mother_range.where(mother_range > EPS))
    feature["inside_bar_compression"] = (1.0 - inside_ratio).clip(
        0.0, 1.0
    ).where(inside & inside_ratio.notna(), 0.0).astype(float)

    log_close = np.log(close + EPS)
    past_log_close, past_log_return = log_close.shift(1), log_close.diff().shift(1)
    log_range = (
        past_log_close.rolling(100, min_periods=100).max()
        - past_log_close.rolling(100, min_periods=100).min()
    )
    return_volatility = past_log_return.rolling(100, min_periods=100).std(ddof=1)
    feature["log_range_over_vol_100"] = log_range / (return_volatility + EPS)
    lower_wick = (
        pd.concat([current_open, close_1], axis=1).min(axis=1) - past_low
    )
    feature["lower_wick_to_tr"] = lower_wick / (true_range + EPS)

    ema_12 = close_1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close_1.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    feature["macd_hist_atr"] = (macd - signal) / (atr_14 + EPS)
    rsi = _wilder_rsi(close.shift(1), 14)
    rsi_min = rsi.rolling(14, min_periods=14).min()
    rsi_max = rsi.rolling(14, min_periods=14).max()
    rsi_spread = rsi_max - rsi_min
    stoch_rsi = pd.Series(np.nan, index=frame.index, dtype=float)
    valid_rsi = rsi_spread > EPS
    stoch_rsi.loc[valid_rsi] = (
        rsi.loc[valid_rsi] - rsi_min.loc[valid_rsi]
    ) / rsi_spread.loc[valid_rsi]
    stoch_rsi.loc[rsi_spread.notna() & ~valid_rsi] = 0.5
    feature["mom_stoch_rsi_14_14_3"] = stoch_rsi.rolling(
        3, min_periods=3
    ).mean().clip(0.0, 1.0)
    feature["mom_tl_break_bull_30"] = _trendline_breakout(frame)
    feature["open_gap_atr_14"] = (open_price - close_1) / (atr_14 + EPS)

    previous_two_high, previous_two_low = high.shift(2), low.shift(2)
    outside = (
        (past_high >= previous_two_high) & (past_low <= previous_two_low)
    ).astype(float)
    feature["outside_bar_expansion"] = outside * candle_range / (atr_14 + EPS)
    log_high_low = np.log((high + EPS) / (low + EPS))
    feature["parkinson_vol_20"] = np.sqrt(
        log_high_low.pow(2).shift(1).rolling(20, min_periods=20).mean()
        / (4.0 * np.log(2.0))
    )
    short_range = (
        past_high.rolling(20, min_periods=20).max()
        - past_low.rolling(20, min_periods=20).min()
    )
    long_range = (
        past_high.rolling(100, min_periods=100).max()
        - past_low.rolling(100, min_periods=100).min()
    )
    feature["range_compression_20_100"] = (
        short_range / (long_range + EPS)
    ).clip(0.0, 1.0)
    feature["realized_vol_20"] = np.log(close / close.shift(1)).shift(1).rolling(
        20, min_periods=20
    ).std(ddof=0)

    lagged_return = return_1.shift(1)
    autocorrelation = return_1.rolling(30, min_periods=30).corr(lagged_return)
    autocorrelation_degenerate = (
        return_1.rolling(30, min_periods=30).count().eq(30)
        & lagged_return.rolling(30, min_periods=30).count().eq(30)
        & (
            return_1.rolling(30, min_periods=30).std(ddof=0).le(EPS)
            | lagged_return.rolling(30, min_periods=30).std(ddof=0).le(EPS)
            | ~np.isfinite(autocorrelation)
        )
    )
    feature["ret_autocorr_1_30"] = autocorrelation.mask(
        autocorrelation_degenerate, 0.0
    ).replace([np.inf, -np.inf], np.nan)

    return_volume_correlation = absolute_return.rolling(
        30, min_periods=30
    ).corr(past_volume)
    return_volume_degenerate = (
        absolute_return.rolling(30, min_periods=30).count().eq(30)
        & past_volume.rolling(30, min_periods=30).count().eq(30)
        & (
            absolute_return.rolling(30, min_periods=30).std(ddof=0).le(EPS)
            | past_volume.rolling(30, min_periods=30).std(ddof=0).le(EPS)
            | ~np.isfinite(return_volume_correlation)
        )
    )
    feature["ret_vol_corr_30"] = return_volume_correlation.mask(
        return_volume_degenerate, 0.0
    ).replace([np.inf, -np.inf], np.nan)
    return_skew = return_1.rolling(30, min_periods=30).skew()
    skew_degenerate = (
        return_1.rolling(30, min_periods=30).count().eq(30)
        & (
            return_1.rolling(30, min_periods=30).std(ddof=0).le(EPS)
            | ~np.isfinite(return_skew)
        )
    )
    feature["return_skew_30"] = return_skew.mask(
        skew_degenerate, 0.0
    ).replace([np.inf, -np.inf], np.nan)
    feature["roc_10"] = close_1 / close_1.shift(10) - 1.0
    feature["rsi_centered_14"] = rsi / 100.0 - 0.5
    divergence_flags = _rsi_hidden_div_flags(frame)
    persistence = np.zeros(len(frame), dtype=float)
    for index, direction in enumerate(divergence_flags):
        if direction == 0:
            continue
        if index > 0 and np.sign(persistence[index - 1]) == direction:
            persistence[index] = direction * (abs(persistence[index - 1]) + 1.0)
        else:
            persistence[index] = float(direction)
    feature["rsi_div_persistence"] = pd.Series(persistence, index=frame.index)
    feature["rsi_hidden_div_flag"] = pd.Series(
        divergence_flags, index=frame.index, dtype=float
    )
    signs = np.sign(return_1)
    flips = ((signs * signs.shift(1)) < 0).astype(float)
    feature["sign_flip_rate_20"] = flips.rolling(20, min_periods=20).mean()
    feature["tr_to_atr_14"] = true_range / (atr_14 + EPS)
    up_close = (close_1 > close_2).astype(float)
    feature["up_close_ratio_5"] = up_close.rolling(
        5, min_periods=5
    ).mean().clip(0.0, 1.0)
    upper_wick = past_high - pd.concat(
        [current_open, close_1], axis=1
    ).max(axis=1)
    feature["upper_wick_to_tr"] = upper_wick / (true_range + EPS)
    upside = return_1.clip(lower=0.0)
    feature["upside_semivol_20"] = np.sqrt(
        upside.pow(2).rolling(20, min_periods=20).mean()
    )
    tr_over_atr = true_range / (atr_14 + EPS)
    feature["vol_of_vol_ratio_20"] = tr_over_atr.rolling(
        20, min_periods=20
    ).std(ddof=0) / (
        tr_over_atr.ewm(span=20, adjust=False, min_periods=20).mean() + EPS
    )
    feature["vol_over_ema20"] = past_volume / (
        past_volume.ewm(span=20, adjust=False, min_periods=20).mean() + EPS
    )
    feature["vol_over_median20"] = past_volume / (
        past_volume.rolling(20, min_periods=20).median() + EPS
    )
    feature["vol_ratio_20_100"] = past_volume.ewm(
        span=20, adjust=False, min_periods=20
    ).mean() / (
        past_volume.ewm(span=100, adjust=False, min_periods=100).mean() + EPS
    )
    realized_volatility = log_close.diff().shift(1).rolling(
        20, min_periods=20
    ).std(ddof=1)
    feature["vol_regime_pct_120"] = _percentile_of_previous(
        realized_volatility, 120
    )
    feature["volume_percentile_60"] = _percentile_of_previous(past_volume, 60)

    result = pd.DataFrame(feature, index=frame.index)
    for name, first_valid_row in _FIRST_VALID_SEGMENT_ROW.items():
        if first_valid_row:
            result.loc[result.index < first_valid_row, name] = np.nan
    return result.loc[:, ENG52_FEATURE_NAMES].astype(float)


def compute_eng52_features(hourly: pd.DataFrame) -> pd.DataFrame:
    work = _validated_input(hourly)
    segment_id = work["open_time"].diff().ne(ONE_HOUR).cumsum()
    parts: list[pd.DataFrame] = []
    for _, segment in work.groupby(segment_id, sort=False):
        start = int(segment.index[0])
        computed = _compute_segment(segment.reset_index(drop=True))
        computed.index = range(start, start + len(segment))
        parts.append(computed)
    features = pd.concat(parts).sort_index()
    result = pd.concat([work[["open_time"]], features], axis=1)
    result.index = hourly.index
    return result.loc[:, ("open_time", *ENG52_FEATURE_NAMES)]
