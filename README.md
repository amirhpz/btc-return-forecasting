# BTC Return Forecasting Benchmark

A controlled, leakage-safe benchmark for forecasting the **one-hour-ahead BTCUSDT log return**.

The project intentionally starts small:

- one market: Binance Spot,
- one symbol: BTCUSDT,
- one forecast horizon: one hour,
- one real-time lookback: 24 hours,
- development first on 1-hour bars,
- controlled transfer to 5-minute bars,
- no attention, no multi-asset input, no hyperparameter search, and no XAI in the initial benchmark.

The model output is a future log return. A future price may be reconstructed only as a secondary presentation output:

```text
predicted_price_t_plus_h = close_t * exp(predicted_log_return)
```

## Environment

Python 3.12 is pinned because current PyTorch Windows support includes Python 3.12, while newer Python versions may not be supported by the selected wheel.

Install `uv` on Windows:

```powershell
winget install --id=astral-sh.uv -e
```

Create the base environment and run foundation checks:

```powershell
uv python install 3.12
uv sync
uv run btc-forecast doctor
uv run btc-forecast validate-configs
uv run pytest
```

For the RTX 2060 / CUDA 12.6 environment:

```powershell
uv sync --extra cu126
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For CPU-only PyTorch:

```powershell
uv sync --extra cpu
```

Do not enable `cpu` and `cu126` together.

## Initial Commands

```powershell
uv run btc-forecast show-plan
uv run btc-forecast init-run --experiment E00
```

The scaffold does not train a model yet. It establishes the contracts, tests, configuration, run registry, and phase gates required before implementation.

## Canonical Data Acquisition

E00A acquires the frozen set of 108 official monthly Binance Spot `BTCUSDT` 5-minute archives from
2017-08 through 2026-07. Start with the network-free plan:

```powershell
btc-forecast acquire-data --config configs/data_acquisition.yaml --dry-run
```

The downloader preserves upstream ZIP and checksum files, supports safe partial resume, and verifies
SHA-256 plus ZIP CRC before finalizing an archive. See [docs/14_DATA_ACQUISITION.md](docs/14_DATA_ACQUISITION.md)
for the source contract, owner-run commands, failure handling, and the E00A/E00B boundary.

## Source of Truth

Read these files before making changes:

1. `AGENTS.md`
2. `docs/00_PROJECT_CHARTER.md`
3. `docs/03_FORECASTING_PROTOCOL.md`
4. `docs/08_PHASED_ROADMAP.md`
5. `docs/agent/AGENT_RESPONSE_PROTOCOL.md`

`pyproject.toml` is the dependency source of truth. Run `uv lock` on the target machine, commit the generated `uv.lock`, and then use locked runs. Do not maintain a second manually edited dependency list.
