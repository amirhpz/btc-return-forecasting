"""
Shared utilities for feature computation and the new quantile-based normalization.

Normalization policy (per symbol, per feature):
- Compute `abs_min` = 20th percentile of train.
- Compute `abs_max` = 80th percentile of train.
- Positive features: map [abs_min -> 0, abs_max -> 1], clip outside to {0,1}.
- Signed features (can be negative): map [abs_min -> -1, abs_max -> 1], clip outside.
- Sparse positive features (mostly zeros): keep zero at zero; normalize non-zero
  values with log1p using non-zero train quantiles (recommended q50/q95).
- Binary features: leave unchanged.
- Ternary {-1, 0, 1} features: leave unchanged.

All bounds must come from train only; test reuses the stored metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12

# Modes for normalization
Mode = Literal[
    "positive", "signed", "binary", "ternary", "sparse_positive", "sparse_signed"
]


@dataclass
class RSIDivParams:
    """Shared parameters for RSI Divergence features"""

    rsi_period: int = 14
    pivot_k: int = 3
    max_pivot_age: int = 60
    min_pivot_sep: int = 5
    trend_filter: bool = True
    trend_len: int = 50


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Return RSI using the native MetaTrader/Wilder seed and recurrence.

    The first valid value is seeded with the arithmetic mean of the first
    ``period`` finite gains and losses. Later values use Wilder's recurrence.
    Leading or internal missing values remain unavailable until a fresh full
    seed window exists.
    """
    if period <= 0:
        raise ValueError("period must be greater than zero")

    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    output = np.full(n, np.nan, dtype=float)
    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)

    delta = np.full(n, np.nan, dtype=float)
    finite_pair = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    delta[1:][finite_pair] = values[1:][finite_pair] - values[:-1][finite_pair]
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)

    last_seed_or_value = -1
    for i in range(1, n):
        if (
            last_seed_or_value == i - 1
            and np.isfinite(avg_gain[i - 1])
            and np.isfinite(delta[i])
        ):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
        elif i >= period:
            window_gain = gain[i - period + 1 : i + 1]
            window_loss = loss[i - period + 1 : i + 1]
            if np.isfinite(window_gain).all() and np.isfinite(window_loss).all():
                avg_gain[i] = window_gain.mean()
                avg_loss[i] = window_loss.mean()
            else:
                continue
        else:
            continue

        last_seed_or_value = i
        if avg_loss[i] > EPS:
            relative_strength = avg_gain[i] / avg_loss[i]
            output[i] = 100.0 - 100.0 / (1.0 + relative_strength)
        elif avg_gain[i] > EPS:
            output[i] = 100.0
        else:
            output[i] = 50.0

    return pd.Series(output, index=close.index, dtype=float)


def find_pivots(
    high: pd.Series, low: pd.Series, pivot_k: int, min_pivot_sep: int
) -> Tuple[List[int], List[int]]:
    """Return lists of pivot low indices and pivot high indices."""
    window = 2 * pivot_k + 1
    low_min = low.rolling(window=window, center=True, min_periods=window).min()
    high_max = high.rolling(window=window, center=True, min_periods=window).max()
    raw_pivot_low = low == low_min
    raw_pivot_high = high == high_max

    def filter_sep(idx_bool: pd.Series) -> List[int]:
        idx = np.flatnonzero(idx_bool.to_numpy())
        kept = []
        last = -(10**9)
        for i in idx:
            if i - last < min_pivot_sep:
                continue
            kept.append(i)
            last = i
        return kept

    return filter_sep(raw_pivot_low), filter_sep(raw_pivot_high)


def get_rsi_hidden_div_flags(
    g: pd.DataFrame, p: RSIDivParams
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Shared base logic to compute RSI hidden divergence flag.
    Returns the temporal flag array and the chronologically sorted working dataframe.
    """
    time_col = "datetime"
    if time_col not in g.columns and "datetime_utc" in g.columns:
        time_col = "datetime_utc"

    g_work = g.sort_values(time_col) if time_col in g.columns else g.copy()
    g_work = g_work.copy()
    g_work["_orig_idx"] = g_work.index
    g_work = g_work.reset_index(drop=True)

    n = len(g_work)
    close_lag = g_work["close"].shift(1)
    high = g_work["high"].shift(1)
    low = g_work["low"].shift(1)

    rsi = wilder_rsi(close_lag, p.rsi_period).to_numpy()
    piv_low, piv_high = find_pivots(high, low, p.pivot_k, p.min_pivot_sep)

    def latest_two(pivots: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        last1, last2 = -(10**9), -(10**9)
        arr1, arr2 = np.full(n, -(10**9), dtype=int), np.full(n, -(10**9), dtype=int)
        ptr, m = 0, len(pivots)
        for t in range(n):
            limit = t - p.pivot_k
            while ptr < m and pivots[ptr] <= limit:
                last2, last1 = last1, pivots[ptr]
                ptr += 1
            if last1 < t - p.max_pivot_age:
                last1 = -(10**9)
            if last2 < t - p.max_pivot_age:
                last2 = -(10**9)
            arr1[t], arr2[t] = last1, last2
        return arr1, arr2

    low_latest, low_prev = latest_two(piv_low)
    high_latest, high_prev = latest_two(piv_high)

    trend_ok_bull = np.ones(n, dtype=bool)
    trend_ok_bear = np.ones(n, dtype=bool)
    if p.trend_filter:
        sma = (
            close_lag.rolling(window=p.trend_len, min_periods=p.trend_len)
            .mean()
            .to_numpy()
        )
        close_arr = close_lag.to_numpy()
        trend_ok_bull = close_arr > sma
        trend_ok_bear = close_arr < sma
        trend_ok_bull[np.isnan(sma)] = False
        trend_ok_bear[np.isnan(sma)] = False

    flag = np.zeros(n, dtype=np.int8)

    for t in range(n):
        i2, i1 = low_latest[t], low_prev[t]
        j2, j1 = high_latest[t], high_prev[t]

        bullish = (
            i1 >= 0
            and i2 >= 0
            and low.iloc[i2] > low.iloc[i1]
            and rsi[i2] < rsi[i1]
            and trend_ok_bull[t]
        )
        bearish = (
            j1 >= 0
            and j2 >= 0
            and high.iloc[j2] < high.iloc[j1]
            and rsi[j2] > rsi[j1]
            and trend_ok_bear[t]
        )

        if bullish and not bearish:
            flag[t] = 1
        elif bearish and not bullish:
            flag[t] = -1
        elif bullish and bearish:
            if i2 > j2:
                flag[t] = 1
            elif j2 > i2:
                flag[t] = -1
            else:
                flag[t] = 0
        else:
            flag[t] = 0

    return flag, g_work


@dataclass
class Bounds:
    abs_min: float
    abs_max: float


def _time_column(df: pd.DataFrame) -> str | None:
    if "datetime" in df.columns:
        return "datetime"
    if "datetime_utc" in df.columns:
        return "datetime_utc"
    return None


def find_time_gaps(
    df: pd.DataFrame,
    expected_interval: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Return rows that start a larger-than-expected timestamp gap.

    The check is symbol-local. If ``expected_interval`` is omitted, the most
    common positive interval of each symbol is used. The function reports gaps
    but does not mutate the input or fill missing candles.
    """
    time_col = _time_column(df)
    columns = ["symbol", "previous_time", "current_time", "delta", "expected"]
    if time_col is None or df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    if work[time_col].isna().any():
        raise ValueError(f"Column {time_col!r} contains invalid timestamps")

    groups = work.groupby("symbol", sort=False) if "symbol" in work.columns else [(None, work)]
    records: list[dict] = []

    for symbol, group in groups:
        times = group[time_col].reset_index(drop=True)
        deltas = times.diff()
        positive = deltas[deltas > pd.Timedelta(0)]
        if expected_interval is None:
            if positive.empty:
                continue
            expected = positive.mode().iloc[0]
        else:
            expected = pd.Timedelta(expected_interval)
            if expected <= pd.Timedelta(0):
                raise ValueError("expected_interval must be positive")

        gap_positions = np.flatnonzero((deltas > expected).to_numpy())
        for position in gap_positions:
            records.append(
                {
                    "symbol": symbol,
                    "previous_time": times.iloc[position - 1],
                    "current_time": times.iloc[position],
                    "delta": deltas.iloc[position],
                    "expected": expected,
                }
            )

    return pd.DataFrame.from_records(records, columns=columns)


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate structural, numerical, OHLC, and chronological invariants."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("OHLCV input is empty")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric = df[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV columns must contain only finite numeric values")

    prices = numeric[["open", "high", "low", "close"]]
    if (prices <= 0.0).any().any():
        raise ValueError("OHLC prices must be strictly positive")
    if (numeric["volume"] < 0.0).any():
        raise ValueError("volume must be non-negative")

    row_max = numeric[["open", "close"]].max(axis=1)
    row_min = numeric[["open", "close"]].min(axis=1)
    invalid_high = numeric["high"] < row_max
    invalid_low = numeric["low"] > row_min
    invalid_range = numeric["high"] < numeric["low"]
    if invalid_high.any() or invalid_low.any() or invalid_range.any():
        bad = df.index[invalid_high | invalid_low | invalid_range].tolist()[:10]
        raise ValueError(f"Invalid OHLC relationships at rows: {bad}")

    time_col = _time_column(df)
    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if parsed_time.isna().any():
            raise ValueError(f"Column {time_col!r} contains invalid timestamps")

        if "symbol" in df.columns:
            check = pd.DataFrame({"symbol": df["symbol"], "time": parsed_time})
            if check.duplicated(["symbol", "time"]).any():
                raise ValueError("Duplicate timestamps found within at least one symbol")
            monotonic = check.groupby("symbol", sort=False)["time"].apply(
                lambda s: s.is_monotonic_increasing
            )
            if not bool(monotonic.all()):
                raise ValueError("Timestamps must be sorted ascending within each symbol")
        else:
            if parsed_time.duplicated().any():
                raise ValueError("Duplicate timestamps found")
            if not parsed_time.is_monotonic_increasing:
                raise ValueError("Timestamps must be sorted ascending")


def _bar_true_range(df: pd.DataFrame) -> pd.Series:
    """Return per-candle True Range on the candle's own row."""
    previous_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - previous_close).abs()
    low_close = (df["low"] - previous_close).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def true_range(df: pd.DataFrame) -> pd.Series:
    """Return bar-open causal True Range.

    Output at row ``t`` is the True Range of completed candle ``t-1``. The
    first candle uses its high-low range because no earlier close exists.
    """
    return _bar_true_range(df).shift(1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return ATR aligned to native MetaTrader 5 ``iATR`` at shift 1.

    MetaQuotes' reference ATR implementation initializes the first ATR from
    TR[1:period+1] and then maintains the simple moving average of the latest
    ``period`` True Range values. At output row ``t`` this function returns the
    native ATR value of completed candle ``t-1``. Consequently, MQL5 may use
    ``CopyBuffer(atr_handle, 0, 1, 1, buffer)`` for the matching value.
    """
    if period <= 0:
        raise ValueError("period must be greater than zero")

    tr_by_bar = _bar_true_range(df)
    atr_by_bar = tr_by_bar.rolling(window=period, min_periods=period).mean()

    # Native ATR leaves bars 0..period-1 undefined and seeds at bar `period`
    # from TR values 1..period. Masking removes the earlier pandas window that
    # would otherwise include bar 0 and would not match MetaTrader.
    atr_by_bar.iloc[:period] = np.nan
    return atr_by_bar.shift(1)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Alias for the shared causal Wilder RSI implementation."""
    return wilder_rsi(close, period)


def dmi_balance(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute normalized DMI balance:
    (DI+ - DI-) / (DI+ + DI- + eps), approximately in [-1, 1].

    Bar-open causal alignment: output at index t is based on directional
    movement up to candle t-1.
    """
    high_lag = df["high"].shift(1)
    low_lag = df["low"].shift(1)
    up_move = high_lag.diff()
    down_move = -low_lag.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr_n = atr(df, period=period)
    plus_di = (
        100.0
        * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / (atr_n + EPS)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / (atr_n + EPS)
    )
    return (plus_di - minus_di) / (plus_di + minus_di + EPS)


def add_feature_per_symbol(
    df: pd.DataFrame,
    feature_col: str,
    compute_fn,
) -> pd.DataFrame:
    """Append one causal feature while preserving the caller's row order.

    Computation is always performed chronologically within each symbol. The
    original row order is restored after feature generation.
    """
    out = df.copy()
    out["_orig_idx"] = np.arange(len(out))

    time_col = _time_column(out)
    if time_col is not None:
        sort_columns = ["symbol", time_col] if "symbol" in out.columns else [time_col]
        out = out.sort_values(sort_columns, kind="stable")

    validate_ohlcv(out)

    if "symbol" in out.columns:
        parts = []
        for _, group in out.groupby("symbol", sort=False, group_keys=False):
            computed = group.copy()
            computed[feature_col] = compute_fn(computed)
            parts.append(computed)
        merged = pd.concat(parts, axis=0)
    else:
        out[feature_col] = compute_fn(out)
        merged = out

    return merged.sort_values("_orig_idx").drop(columns=["_orig_idx"])


# -------- Quantile bounds and normalization helpers -------- #


def quantile_bounds(
    series: pd.Series, lower_q: float = 0.2, upper_q: float = 0.8
) -> Bounds:
    """Return Bounds(abs_min, abs_max) using specified quantiles."""
    abs_min = float(series.quantile(lower_q))
    abs_max = float(series.quantile(upper_q))
    return Bounds(abs_min=abs_min, abs_max=abs_max)


def sparse_nonzero_bounds(
    series: pd.Series, lower_q_nonzero: float = 0.5, upper_q_nonzero: float = 0.95
) -> Bounds:
    """
    Return bounds for sparse positive features using non-zero values only.

    If a series has no non-zero values, both bounds are returned as 0.0.
    """
    non_zero = series[(series != 0) & series.notna()]
    if non_zero.empty:
        return Bounds(abs_min=0.0, abs_max=0.0)
    abs_min = float(non_zero.quantile(lower_q_nonzero))
    abs_max = float(non_zero.quantile(upper_q_nonzero))
    return Bounds(abs_min=abs_min, abs_max=abs_max)


def sparse_signed_bounds(series: pd.Series, upper_q_nonzero: float = 0.95) -> Bounds:
    """
    Return bounds for sparse signed features using absolute non-zero values.
    abs_min is set to 0.0 to ensure symmetric mapping around zero.
    """
    non_zero_abs = series[(series != 0) & series.notna()].abs()
    if non_zero_abs.empty:
        return Bounds(abs_min=0.0, abs_max=0.0)
    abs_max = float(non_zero_abs.quantile(upper_q_nonzero))
    return Bounds(abs_min=0.0, abs_max=abs_max)


def normalize_feature(series: pd.Series, bounds: Bounds, mode: Mode) -> pd.Series:
    """Normalize one feature while preserving missing-value positions."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    missing = values.isna()
    abs_min = float(bounds.abs_min)
    abs_max = float(bounds.abs_max)

    if mode in ("binary", "ternary"):
        return values

    if not np.isfinite(abs_min) or not np.isfinite(abs_max):
        raise ValueError("Normalization bounds must be finite")

    if mode == "sparse_signed":
        if abs_max <= EPS:
            output = pd.Series(0.0, index=values.index, dtype=float)
        else:
            output = (values / abs_max).clip(-1.0, 1.0)
        return output.mask(missing)

    width = abs_max - abs_min
    if width <= EPS:
        if mode in ("positive", "signed"):
            output = pd.Series(0.0, index=values.index, dtype=float)
        elif mode == "sparse_positive":
            output = pd.Series(0.0, index=values.index, dtype=float)
            output.loc[(values != 0.0) & ~missing] = 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return output.mask(missing)

    if mode == "positive":
        output = ((values - abs_min) / width).clip(0.0, 1.0)
        return output.mask(missing)

    if mode == "signed":
        output = (-1.0 + 2.0 * (values - abs_min) / width).clip(-1.0, 1.0)
        return output.mask(missing)

    if mode == "sparse_positive":
        zero_mask = values == 0.0
        lo = np.log1p(max(abs_min, 0.0))
        hi = np.log1p(max(abs_max, 0.0))
        log_width = hi - lo
        if log_width <= EPS:
            output = pd.Series(0.0, index=values.index, dtype=float)
            output.loc[(~zero_mask) & ~missing] = 1.0
        else:
            output = (
                (np.log1p(values.clip(lower=0.0)) - lo) / log_width
            ).clip(0.0, 1.0)
            output = output.mask(zero_mask, 0.0)
        return output.mask(missing)

    raise ValueError(f"Unknown mode: {mode}")

def load_bounds_json(path: str | Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load bounds metadata JSON (symbol -> feature -> {abs_min, abs_max})."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_bounds_per_symbol(
    df: pd.DataFrame,
    feature_col: str,
    bounds_map: Dict[str, Dict[str, Dict[str, float]]],
    mode: Mode,
) -> pd.DataFrame:
    """
    Apply normalization per symbol using provided bounds map.
    bounds_map schema: {symbol: {feature_col: {abs_min, abs_max}}}
    """
    if "symbol" not in df.columns:
        raise ValueError("symbol column is required for per-symbol normalization")

    parts = []
    for sym, g in df.groupby("symbol", sort=False, group_keys=False):
        meta = bounds_map.get(sym, {}).get(feature_col)
        if meta is None:
            raise KeyError(f"Missing bounds for symbol={sym}, feature={feature_col}")
        bounds = Bounds(abs_min=float(meta["abs_min"]), abs_max=float(meta["abs_max"]))
        g2 = g.copy()
        g2[feature_col] = normalize_feature(g2[feature_col], bounds, mode)
        parts.append(g2)
    return pd.concat(parts, axis=0)


def compute_bounds_table(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    lower_q: float = 0.2,
    upper_q: float = 0.8,
    feature_modes: Dict[str, Mode] | None = None,
    sparse_lower_q_nonzero: float = 0.5,
    sparse_upper_q_nonzero: float = 0.95,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute bounds for each symbol and feature.
    Returns dict: {symbol: {feature: {abs_min, abs_max}}}
    """
    if "symbol" not in df.columns:
        raise ValueError("symbol column is required to compute per-symbol bounds")

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sym, g in df.groupby("symbol", sort=False, group_keys=False):
        out[sym] = {}
        for col in feature_cols:
            mode = feature_modes.get(col) if feature_modes else None
            if mode == "sparse_positive":
                bounds = sparse_nonzero_bounds(
                    g[col],
                    lower_q_nonzero=sparse_lower_q_nonzero,
                    upper_q_nonzero=sparse_upper_q_nonzero,
                )
            elif mode == "sparse_signed":
                bounds = sparse_signed_bounds(
                    g[col],
                    upper_q_nonzero=sparse_upper_q_nonzero,
                )
            else:
                bounds = quantile_bounds(g[col], lower_q=lower_q, upper_q=upper_q)
            out[sym][col] = {"abs_min": bounds.abs_min, "abs_max": bounds.abs_max}
    return out


def compute_bounds_single_symbol(
    df: pd.DataFrame,
    symbol: str,
    feature_cols: Iterable[str],
    lower_q: float = 0.2,
    upper_q: float = 0.8,
    feature_modes: Dict[str, Mode] | None = None,
    sparse_lower_q_nonzero: float = 0.5,
    sparse_upper_q_nonzero: float = 0.95,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute bounds for a single symbol dataset directly without groupby overhead.
    Returns dict: {symbol: {feature: {abs_min, abs_max}}}
    """
    out_symbol_bounds = {}

    for col in feature_cols:
        mode = feature_modes.get(col) if feature_modes else None

        if mode == "sparse_positive":
            bounds = sparse_nonzero_bounds(
                df[col],
                lower_q_nonzero=sparse_lower_q_nonzero,
                upper_q_nonzero=sparse_upper_q_nonzero,
            )
        elif mode == "sparse_signed":
            bounds = sparse_signed_bounds(
                df[col],
                upper_q_nonzero=sparse_upper_q_nonzero,
            )
        # -----------------------------------
        else:
            bounds = quantile_bounds(df[col], lower_q=lower_q, upper_q=upper_q)

        out_symbol_bounds[col] = {"abs_min": bounds.abs_min, "abs_max": bounds.abs_max}

    return {symbol: out_symbol_bounds}


def apply_bounds_single_symbol(
    df: pd.DataFrame,
    symbol: str,
    bounds_map: Dict[str, Dict[str, Dict[str, float]]],
    feature_modes: Dict[str, Mode],
) -> pd.DataFrame:
    """
    Apply normalization to a single symbol dataset efficiently.
    Applies to all requested features without requiring a 'symbol' column or groupby.
    """
    if symbol not in bounds_map:
        raise KeyError(f"No bounds found for symbol: {symbol}")

    out = df.copy()
    symbol_bounds = bounds_map[symbol]

    for col, mode in feature_modes.items():
        if col not in out.columns:
            continue

        meta = symbol_bounds.get(col)
        if meta is None:
            raise KeyError(f"Missing bounds for symbol={symbol}, feature={col}")

        bounds = Bounds(abs_min=float(meta["abs_min"]), abs_max=float(meta["abs_max"]))
        out[col] = normalize_feature(out[col], bounds, mode)

    return out


def update_and_save_bounds_json(
    new_bounds: Dict[str, Dict[str, Dict[str, float]]], path: str | Path
) -> None:
    """
    Safely load existing bounds.json (if exists), merge with new_bounds,
    and save back to disk. Prevents overwriting previous symbols.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    existing_bounds = {}

    # Load existing data if available
    if path_obj.exists():
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                existing_bounds = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {path} was empty or corrupted. Creating a new one.")

    # Merge new bounds
    for sym, features_dict in new_bounds.items():
        if sym not in existing_bounds:
            existing_bounds[sym] = {}
        existing_bounds[sym].update(features_dict)

    # Save merged data
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(existing_bounds, f, indent=4)
