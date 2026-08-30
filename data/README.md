# Data Directory

Raw data is immutable and excluded from Git. E00A preserves official monthly Binance Spot source
archives and their checksum files under:

```text
data/raw/binance/spot/monthly/klines/BTCUSDT/5m/
```

The single canonical row-level master dataset described by `configs/data.yaml` is a later E00B
artifact; E00A does not concatenate, parse, normalize, or convert the archives. Generated data
belongs in `interim`, `processed`, and `splits`, and every generated dataset requires a manifest.
