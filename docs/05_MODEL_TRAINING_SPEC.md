# Model and Training Specification

## Baselines

### B0: Naive Suite

- Zero-return prediction: primary error baseline.
- Previous observed return: persistence/momentum baseline and directional reference.

### B1: Ridge

A linear baseline using the flattened, scaled input window.

### B2: LSTM

Single-layer unidirectional LSTM with 64 hidden units.

### B3: CNN

One Conv1D block with 32 channels followed by global/adaptive average pooling.

### B4: CNN-LSTM

```text
Input [batch, time, features]
  -> Conv1D(32, kernel=3)
  -> ReLU
  -> MaxPool1D(2)
  -> LSTM(hidden=64, layers=1)
  -> Dropout(0.2)
  -> Linear(1)
```

No attention is permitted in the initial benchmark.

## Fixed Training Configuration

```text
optimizer       Adam
learning rate   0.001
loss            Huber
batch size      128
max epochs      30
early stopping  patience 5
feature scaler  RobustScaler, train only
target scaler   StandardScaler, train only
```

The checkpoint is selected by validation RMSE after inverse-transforming the target.

## Reproducibility

- Seed 42 for debugging and development.
- Seeds 42, 123, and 2026 for confirmation.
- Save model configuration, seed, data manifest hashes, split version, scaler files, checkpoint, history, predictions, and metrics.

## Tuning Policy

No automatic search is allowed. A later manual change requires:

1. a documented failure diagnosis;
2. one changed variable;
3. a new experiment ID;
4. validation-only assessment;
5. no test access.
