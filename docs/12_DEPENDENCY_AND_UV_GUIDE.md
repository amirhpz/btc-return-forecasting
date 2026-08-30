# Dependency and uv Guide

## Decision

Use `uv` as the environment and dependency manager because the project needs:

- a pinned Python version,
- a universal lockfile,
- reproducible sync,
- dependency groups,
- explicit PyTorch CPU/CUDA indexes.

`pyproject.toml` is the dependency source of truth. The first successful local `uv lock` creates `uv.lock`; commit that file before running research experiments.

## Python

Python is restricted to `>=3.12,<3.13`. This avoids using a Python version unsupported by the selected Windows PyTorch release.

## Base Environment

```powershell
uv python install 3.12
uv sync
```

The development group is installed by default by uv.

## PyTorch Backend

CPU:

```powershell
uv sync --extra cpu
```

CUDA 12.6:

```powershell
uv sync --extra cu126
```

The extras are declared mutually exclusive. Torch is not included in the base environment so data validation and classical baselines do not require a large deep-learning installation.

## Lock Discipline

Normal reproducible commands use the existing lock:

```powershell
uv lock --check
uv run --locked pytest
```

Do not upgrade dependencies during an experiment. Dependency upgrades require a dedicated maintenance commit and rerun of foundation tests.
