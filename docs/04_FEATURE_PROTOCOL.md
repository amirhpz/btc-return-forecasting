# Feature Protocol

## Core Rule

Feature windows are defined by real-time duration, not an equal number of bars. For example, `return_6h` uses 6 bars on 1h data and 72 bars on 5m data.

This prevents an accidental change of economic meaning when moving between timeframes.

## F0: Minimal

Ten features covering returns, candle structure, volatility, and volume:

```text
log_return_1h
log_return_3h
log_return_6h
candle_log_return
high_low_range_pct
close_position_in_range
rolling_volatility_6h
rolling_volatility_24h
log_volume_change_1h
relative_volume_24h
```

## F1: Core

F0 plus thirteen documented indicators, for 23 total features. See `configs/features/f1_core.yaml`.

## F2: Existing Full 52

F2 is deliberately blocked until the existing 52-feature catalog is imported and frozen. Do not invent formulas. Each feature requires:

- exact name,
- family,
- formula,
- source columns,
- duration/window,
- causal availability rule,
- missing/warm-up policy.

E07 must fail fast unless the catalog contains exactly 52 unique enabled features.

## Causality

A feature value at anchor `t` must be reproducible using only completed rows up to `t`. No centered window, negative shift, future interpolation, or backward fill is allowed.

## Missing and Warm-Up Values

Warm-up rows are excluded only after the reason and count are recorded. Missing values are not imputed across time without an explicit train-only policy and test.

## Feature Selection

There is no combinatorial feature search. The controlled comparison is F0 versus F1 versus F2. Only the feature set changes between E05, E06, and E07.
