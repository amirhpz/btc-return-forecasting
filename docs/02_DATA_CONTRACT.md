# Data Contract

## Canonical Source

The canonical dataset is Binance Spot BTCUSDT at 5-minute resolution. The raw file is immutable.

Default path:

```text
data/raw/binance/5m/BTCUSDT_5m.csv
```

CSV and Parquet are allowed. The observed file checksum, row count, minimum timestamp, maximum timestamp, and schema must be written to a manifest during E00.

## Required Columns

```text
open_time, open, high, low, close, volume
```

Optional Binance kline columns are preserved when present:

```text
close_time, quote_asset_volume, number_of_trades,
taker_buy_base_volume, taker_buy_quote_volume
```

## Timestamp Semantics

- `open_time` is UTC and identifies the beginning of a completed bar.
- A row becomes available only after its bar has closed.
- Features for anchor row `t` may use the completed row `t` and prior rows.
- The target uses a later close and is never an input.

## Mandatory E00 Checks

- parsing and UTC normalization,
- chronological ordering,
- duplicate timestamps,
- missing 5-minute intervals,
- non-positive prices,
- negative volume,
- zero-volume count,
- invalid OHLC relationships,
- inconsistent close time when available,
- unexpected columns and dtypes.

No issue is repaired silently.

## 1-Hour Resampling

Each one-hour bar is produced from exactly 12 contiguous 5-minute bars:

```text
open   = first
high   = maximum
low    = minimum
close  = last
volume = sum
```

Additive optional columns are summed. An incomplete one-hour bucket is reported and excluded. No forward fill or backward fill is allowed.

## Data Versioning

Generated datasets use deterministic names and manifests, for example:

```text
data/processed/btcusdt_1h_v001.parquet
data/processed/manifests/resampled_1h_manifest.json
```
