# Risk Register

| Risk | Severity | Control |
|---|---:|---|
| Look-ahead leakage | Critical | Causal feature tests, target-timestamp split, no backfill |
| Repeated test tuning | Critical | Test locked until E10, access log |
| Incomplete 1h resampling buckets | High | Require 12 source bars; report and exclude |
| Target/window misalignment | High | Synthetic alignment tests and exported timestamps |
| Full-data scaling | High | Train-only scaler API and tests |
| Misleading price-level metrics | High | Main target is log return; price metrics secondary |
| Unfair 1h/5m comparison | High | Same horizon, lookback duration, and hourly timestamp intersection |
| Feature redundancy/noise | Medium | Controlled F0/F1/F2 comparison, no subset search |
| Deep-model seed variance | Medium | Three final seeds and per-seed reporting |
| Regime instability | Medium | Chronological holdout; later regime analysis only after benchmark |
| GPU environment conflict | Medium | Explicit mutually exclusive uv extras: cpu or cu126 |
| Existing 52-feature definitions missing | Medium | F2 blocked until exact catalog is frozen |
| Paper-result overclaim | Medium | Literature values contextual unless reimplemented on common benchmark |
