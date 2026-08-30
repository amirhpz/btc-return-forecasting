# AGENTS.md

## 1. Project Role

You are implementing a thesis-grade, leakage-safe BTCUSDT return-forecasting benchmark. Work phase by phase. Do not expand scope without a recorded decision.

## 2. Frozen Scope

- Market: Binance Spot
- Symbol: BTCUSDT only
- Raw master timeframe: 5 minutes
- Development timeframe: 1 hour, resampled from the same 5-minute master data
- Comparison timeframe: 5 minutes
- Target: one-hour-ahead future log return
- Lookback: 24 real hours
- Models: naive baselines, Ridge, LSTM, CNN, simple CNN-LSTM
- Primary comparison: all models on the same timestamps, targets, splits, and metrics

## 3. Prohibited Changes During Initial Benchmark

Do not add:

- additional symbols,
- attention,
- transformers,
- VAE or autoencoders,
- sentiment or news,
- order-book data,
- evolutionary optimization,
- Optuna, grid search, or Bayesian search,
- automatic feature-subset search,
- trading-strategy optimization,
- XAI,
- random train/test splitting.

## 4. Non-Negotiable Time-Series Rules

1. Raw data is immutable.
2. All timestamps are UTC.
3. Features at anchor time `t` may use only completed information available at or before `t`.
4. The target may use `close[t+h]`; features may not.
5. Split chronologically by target timestamp.
6. Fit feature and target scalers on training data only.
7. Never backfill from future observations.
8. Never use the final test set before Experiment E10.
9. Any access to the final test set must be recorded in `docs/test_access_log.csv`.
10. A 1h-versus-5m direct comparison must use the common hourly decision grid.

## 5. Change Discipline

- Implement only the requested phase.
- Inspect existing code before modifying it.
- Prefer small, typed, testable functions.
- Keep notebooks out of core logic.
- Do not silently drop rows, timestamps, or incomplete resampling buckets.
- Report every dropped or excluded row with a reason.
- Do not change multiple experimental variables in one comparison.
- Do not modify frozen configs after test access; create a new version instead.

## 6. Required Commands

Use `uv`:

```text
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run btc-forecast validate-configs
```

Deep-learning work requires one explicit backend:

```text
uv sync --extra cpu
```

or:

```text
uv sync --extra cu126
```

## 7. Required Agent Response

Follow `docs/agent/AGENT_RESPONSE_PROTOCOL.md`. Every response must list:

- current phase,
- objective,
- files inspected,
- files changed,
- exact commands run,
- exact results,
- outputs,
- risks,
- phase decision,
- next permitted step.

Never claim a result was reproduced unless the exact dataset and experimental protocol match.
