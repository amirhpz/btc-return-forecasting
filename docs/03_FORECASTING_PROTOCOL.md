# Forecasting Protocol

## Prediction Target

For anchor time `t` and one-hour horizon `h`:

```text
y_t = log(close_t_plus_1h / close_t)
```

The model predicts `y_t`. Reconstructed price is secondary:

```text
predicted_close_t_plus_1h = close_t * exp(predicted_y_t)
```

## Timeframe Mapping

| Timeframe | Lookback | Horizon |
|---|---:|---:|
| 1h | 24 bars | 1 bar |
| 5m | 288 bars | 12 bars |

Both represent 24 hours of history and a one-hour forecast horizon.

## Cross-Timeframe Fairness

Two evaluations are permitted:

1. Within-timeframe evaluation on all valid anchors.
2. Direct 1h-versus-5m comparison on the common hourly decision grid only.

The direct comparison must use the exact intersection of anchor and target timestamps.

## Split Policy

- Chronological 70/15/15 split.
- Boundaries are calculated once after valid anchors are known in E00.
- Boundaries are written to `configs/frozen/split_boundaries_v001.yaml` and committed.
- Split assignment is based on target timestamp so a target cannot cross into a later split.
- Past context from the previous period may be used to form the first validation/test window; this reflects real deployment and does not expose future targets.

## Final Test Policy

The final test set is not used for model choice, feature choice, debugging, threshold choice, or stopping decisions.

E01-E09 use training and validation data only. E10 opens the test set once with frozen code and configuration. Every test access is recorded.
