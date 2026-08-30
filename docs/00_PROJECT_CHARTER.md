# Project Charter

## Objective

Build a reproducible benchmark that answers one narrow question:

> Can a simple CNN-LSTM forecast the one-hour-ahead BTCUSDT log return better than naive and simpler model baselines under a leakage-safe protocol?

## Frozen Scope

| Item | Decision |
|---|---|
| Market | Binance cryptocurrency spot |
| Symbol | BTCUSDT only |
| Raw master data | 5-minute OHLCV |
| Development timeframe | 1-hour bars resampled from the same 5-minute master |
| Comparison timeframe | 5-minute bars |
| Forecast target | One-hour-ahead log return |
| Lookback | 24 real hours |
| Core model | Simple CNN-LSTM |
| Baselines | Zero return, previous return, Ridge, LSTM, CNN |
| Split | Chronological, 70/15/15, frozen after E00 |
| Final test | Locked until E10 |
| Hyperparameter search | Not allowed in the initial benchmark |

## Success Criteria

The benchmark succeeds when:

1. data provenance and integrity are documented;
2. target and window alignment are verified by tests;
3. all models use the same split and evaluation timestamps;
4. train-only scaling is proven;
5. the final test is opened once after configs are frozen;
6. positive, neutral, or negative results are reported honestly.

The CNN-LSTM does not need to outperform every baseline for the project to be useful.

## Out of Scope

Multi-asset learning, classification labels, attention, XAI, feature-subset optimization, trading optimization, sentiment, order books, and multiple horizons are excluded from this initial repository version.
