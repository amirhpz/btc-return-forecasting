# Experiment Log

| Experiment | Run ID | Date UTC | Data Version | Split Version | Feature Set | Model | Seed | Evaluation Split | Decision | Notes |
|---|---|---|---|---|---|---|---:|---|---|---|
| E00 | | 2026-08-30 | Binance monthly archives | | | | | none | E00A COMPLETE; E00B COMPLETE — REVIEW_REQUIRED | All 108 source archives acquired and independently verified. Raw validation completed without modifying or repairing data; measured temporal anomalies require a gap/data-quality policy before E00C. |
| E01 | | | | | F0 | B0 | | validation | BLOCKED | |
| E02 | | | | | F0 | B1 | | validation | BLOCKED | |
| E03 | | | | | F0 | B2 | 42 | validation | BLOCKED | |
| E04 | | | | | F0 | B3 | 42 | validation | BLOCKED | |
| E05 | | | | | F0 | B4 | 42 | validation | BLOCKED | |
| E06 | | | | | F1 | B4 | 42 | validation | BLOCKED | |
| E07 | | | | | F2 | B4 | 42 | validation | BLOCKED | |
| E08 | | | | | frozen | B4 | multi | validation | BLOCKED | |
| E09 | | | | | frozen | B4 | multi | validation | BLOCKED | |
| E10 | | | | | frozen | frozen | multi | test | BLOCKED | |

## E00A — COMPLETE

Owner-provided runtime evidence for the frozen Binance Spot `BTCUSDT` 5-minute monthly archive set:

| Result | Acquisition | Independent verify-only |
|---|---:|---:|
| Requested archives | 108 | 108 |
| Downloaded archives | 108 | 0 |
| Existing verified archives | 0 | 108 |
| Verified archives | 108 | 108 |
| Failed archives | 0 | 0 |
| Checksum failures | 0 | 0 |
| ZIP failures | 0 | 0 |
| Download failures | 0 | 0 |
| Missing remote archives | 0 | 0 |
| Missing local archives | 0 | 0 |
| Total raw archive bytes | 49,531,376 | 49,531,376 |
| First month | 2017-08 | 2017-08 |
| Last month | 2026-07 | 2026-07 |
| Completion status | COMPLETE | COMPLETE |

All 108 expected Binance Spot `BTCUSDT` 5-minute monthly archives were acquired. All 108 passed
their official upstream SHA-256 checks and ZIP integrity checks. The independent verify-only run
confirmed all 108 local archives, with no missing or failed archive.

## E00B — COMPLETE / REVIEW_REQUIRED

The local read-only validator scanned the frozen 108-archive Binance Spot BTCUSDT 5-minute
dataset through the exclusive cutoff 2026-08-01T00:00:00Z. Validation execution completed and
wrote the three ignored E00B artifacts. No raw archive or row was modified, repaired, filled,
removed, extracted, or transformed.

| Result | Measured value |
|---|---:|
| Archives scanned/opened | 108 / 108 |
| Schema-valid / schema-invalid archives | 108 / 0 |
| Parse-valid / parse-invalid archives | 108 / 0 |
| Fully valid / anomaly-bearing archives | 94 / 14 |
| Total rows | 940,297 |
| Actual first candle | 2017-08-17T04:00:00Z |
| Actual last candle | 2026-07-31T23:55:00Z |
| Millisecond / microsecond archives | 89 / 19 |
| Timestamp-unit mismatches | 0 |
| Timestamp parse failures | 0 |
| 5-minute alignment violations | 241 |
| Close-time relationship violations | 17 |
| Archive/global ordering violations | 0 / 0 |
| Duplicate timestamps | 0 |
| Exact / conflicting duplicate rows | 0 / 0 |
| Expected grid candles | 942,000 |
| Observed unique grid candles | 940,056 |
| Observed unique timestamps, including off-grid | 940,297 |
| Missing candles | 1,944 |
| Missing percentage | 0.20636942675159234% |
| Gap episodes | 32 |
| Longest gap | 2018-02-08T00:30:00Z through 2018-02-10T06:10:00Z; 645 candles; PT3225M |
| First gap | 2017-09-06T16:00:00Z through 2017-09-06T22:55:00Z; 84 candles; PT420M |
| Last gap | 2023-03-24T12:40:00Z through 2023-03-24T13:55:00Z; 16 candles; PT80M |
| Malformed rows / column-count problems / unexpected headers | 0 / 0 / 0 |
| Missing payloads / unexpected ZIP members | 0 / 0 |
| Month-containment violations | 0 |
| Cross-month overlaps / gap boundaries / duplicate timestamps | 0 / 0 / 0 |
| Numeric parse failures / NaN / infinity | 0 / 0 / 0 |
| Non-positive price / OHLC violation rows | 0 / 0 |
| Negative volume / negative trade-count rows | 0 / 0 |
| Taker-base / taker-quote volume anomalies | 0 / 0 |
| Zero-volume rows | 932 |
| Unique ignore values | 34,974 |
| ignore == 0 / nonzero ignore rows | 904,852 / 35,445 |
| Dataset quality decision | REVIEW_REQUIRED |

The full ignore value counts and exact timestamp-violation details are retained in the E00B
runtime artifacts. The validator applied no floating-point tolerance: source numeric strings were
compared as exact decimal values. E00B does not choose a repair policy; gap and timestamp-anomaly
handling must be decided before E00C.

## E00C1 — COMPLETE

The canonical derived Binance Spot BTCUSDT 5-minute dataset was built at
`data/processed/btcusdt_5m_v001.parquet`. Raw archives remained unchanged, and the processed
Parquet artifact remains excluded from Git.

| Result | Measured value |
|---|---:|
| Rows | 940,297 |
| First timestamp | 2017-08-17T04:00:00Z |
| Last timestamp | 2026-07-31T23:55:00Z |
| Off-grid timestamps | 0 |
| Strictly ordered | yes |
| Duplicate timestamps | 0 |
| Timestamp corrections applied | 241 |
| Missing grid positions | 1,703 |
| Raw archives modified | 0 |
| Targeted tests | 3 passed |
| SHA-256 | `bce19f37e38431f15dcefd7e09f0ec37a091d41be664b6f509c71e29502dd86f` |
