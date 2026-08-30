# Run and Artifact Specification

Each run has an immutable directory:

```text
outputs/runs/<run_id>/
  manifest.json
  resolved_config.yaml
  environment.json
  data_manifest_refs.json
  metrics.json
  predictions.parquet
  training_history.csv
  notes.md
```

Model and scaler binaries are stored under:

```text
artifacts/models/<run_id>/
artifacts/scalers/<run_id>/
```

## Run ID

Recommended format:

```text
E05_1h_F0_B4_seed42_YYYYMMDDTHHMMSSZ
```

## Required Manifest Fields

- experiment ID,
- UTC creation time,
- git commit and dirty status,
- Python and package versions,
- seed,
- data file checksum,
- data manifest version,
- split version and exact boundaries,
- timeframe, lookback, horizon,
- feature set and feature names,
- model and training configuration,
- device and CUDA status,
- input/output row counts,
- exclusions with reasons,
- artifact paths.

A metric without its manifest is not an acceptable research result.
