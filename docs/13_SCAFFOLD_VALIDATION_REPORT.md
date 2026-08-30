# Scaffold Validation Report

Generated: 2026-08-30

## Checks Completed

- Project structure created.
- All 24 YAML configuration files parsed successfully.
- Experiment registry contains E00 through E10 with unique IDs.
- Five model IDs and three feature-set IDs resolve correctly.
- Python source compiled successfully.
- Foundation test suite: **9 passed**.
- CLI `validate-configs`, `show-plan`, and `doctor` executed successfully through the source tree.

## uv Lock Status

The project configuration was accepted by uv and dependency resolution started, but this build environment had no outbound DNS/network access and did not contain Python 3.12 locally. Therefore `uv lock` could not download the pinned Python interpreter and no fabricated lockfile is included.

Required first local commands on Windows:

```powershell
uv python install 3.12
uv lock
uv sync --extra cu126
uv run --locked pytest
```

After this succeeds, commit `uv.lock` before E00.

## Validation Environment Limitation

The available build interpreter was Python 3.13.5, while the project intentionally requires Python 3.12 for Windows PyTorch compatibility. The code-only foundation tests passed under Python 3.13.5, but the actual project environment must be recreated with Python 3.12 through uv. Ruff and mypy were not available offline in the build environment; run `scripts/quality.ps1` after local `uv sync`.
