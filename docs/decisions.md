# Decision Log

## D001 - 2026-08-30 - Single-Asset Scope

Use BTCUSDT only for the initial benchmark.

## D002 - 2026-08-30 - Target

Predict one-hour-ahead log return, not raw price level. Reconstructed price is secondary.

## D003 - 2026-08-30 - Timeframes

Develop on 1h bars resampled from the same 5m master dataset; compare later with 5m using a common hourly decision grid.

## D004 - 2026-08-30 - Lookback

Use 24 real hours for both timeframes: 24 bars on 1h and 288 bars on 5m.

## D005 - 2026-08-30 - Complexity Control

No attention, multi-asset data, hyperparameter search, XAI, or trading optimization in v0.1.

## D006 - 2026-08-30 - Dependency Manager

Use uv with Python 3.12 and mutually exclusive PyTorch CPU/CUDA 12.6 extras.

## D007 - 2026-08-30 - Test Governance

Final test is locked until E10. E01-E09 use training and validation only.

## D008 - 2026-08-30 - E00B Raw Data Policy

- The 241 off-grid rows in 2018-02 are a deterministic timestamp displacement.
- In derived data only, map these 241 `open_time` values to their unique nearest 5-minute grid positions.
- Raw Binance archives remain unchanged.
- After this mapping, treat 1,703 grid positions as genuinely missing.
- Do not fill or interpolate missing candles.
- Retain all 17 `close_time`-anomaly rows.
- Downstream chronology and resampling use `open_time`, not source `close_time`.
- Do not repair source `close_time`.
- Retain zero-volume rows.
- Do not filter rows using Binance's `ignore` field, and do not use `ignore` as a feature.
