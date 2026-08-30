# Repository Structure

```text
btc-return-forecasting/
  AGENTS.md
  README.md
  pyproject.toml
  uv.lock                    # generated locally, then committed
  configs/
    project.yaml
    data.yaml
    split.yaml
    training.yaml
    evaluation.yaml
    features/
    models/
    experiments/
    frozen/
  docs/
    agent/
    decisions.md
    experiment_log.md
    test_access_log.csv
  data/
    raw/binance/5m/
    interim/
    processed/
    splits/
  artifacts/
    models/
    scalers/
  outputs/
    runs/
    tables/
    figures/
    logs/
  notebooks/
  scripts/
  src/btc_forecasting/
    common/
    data/
    features/
    targets/
    splits/
    preprocessing/
    datasets/
    baselines/
    models/
    training/
    evaluation/
    experiments/
    reporting/
  tests/
```

Core logic belongs in `src/`. Notebooks may inspect generated artifacts but may not contain the only implementation of data, features, training, or evaluation.
