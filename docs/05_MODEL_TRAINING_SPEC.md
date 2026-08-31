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

## Deep Training Protocol v1

The exact model and training semantics are frozen in
`configs/models/lstm.yaml` and `configs/training.yaml`. E03 uses CUDA and must
fail clearly when CUDA is unavailable.

```text
optimizer       Adam
learning rate   0.001
loss            torch.nn.HuberLoss, delta 0.01, mean reduction
batch size      128
max epochs      30
early stopping  patience 5
feature scaler  RobustScaler, train only
target scaler   none; use the unscaled one-hour log-return
```

The checkpoint is selected by strict improvement in validation Huber loss and
the best-validation-loss checkpoint is restored. The frozen protocol also
specifies the LSTM readout, optimizer defaults, batching, determinism, gradient
clipping, and cosine-annealing scheduler without changing the existing numerical
model or training hyperparameters.

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
