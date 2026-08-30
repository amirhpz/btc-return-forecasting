# Experiment Log

| Experiment | Run ID | Date UTC | Data Version | Split Version | Feature Set | Model | Seed | Evaluation Split | Decision | Notes |
|---|---|---|---|---|---|---|---:|---|---|---|
| E00 | | 2026-08-31 | btcusdt_5m_v001 / btcusdt_1h_v001 / btcusdt_1h_target_v001 | chronological_70_15_15_v001 | | | | none | COMPLETE | E00 data foundation complete: acquisition, validation, canonical 5m and 1h datasets, one-hour target, and leakage-safe frozen chronological split. |
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

## E00C2 — COMPLETE

The canonical BTCUSDT hourly dataset was built from complete UTC hours in the canonical 5-minute
dataset. Hours missing any expected source timestamp from `HH:00` through `HH:55` were excluded
without filling, interpolation, or candle synthesis. The derived artifacts remain excluded from
Git.

| Result | Measured value |
|---|---:|
| Artifact | `data/processed/btcusdt_1h_v001.parquet` |
| Total possible hourly intervals | 78,500 |
| Complete 1h bars | 78,344 |
| Incomplete hours excluded | 156 |
| First timestamp | 2017-08-17T04:00:00Z |
| Last timestamp | 2026-07-31T23:00:00Z |
| Duplicate timestamps | 0 |
| Off-grid timestamps | 0 |
| Every retained hour contains exactly 12 expected 5m candles | yes |
| Targeted tests | 3 passed |
| SHA-256 | `c46c35459477cb43a6f4ce6e2b2ebbfb64e57e4972a6a472382fad8f32694cb6` |

## E00D — COMPLETE

The one-real-hour future log-return target was constructed only where the next canonical hourly
bar existed at exactly `t + 1 hour`. No target was created across a missing hour, and the final
row received no fabricated target. The derived artifact remains excluded from Git.

| Result | Measured value |
|---|---:|
| Artifact | `data/processed/btcusdt_1h_target_v001.parquet` |
| Input 1h rows | 78,344 |
| Eligible target rows | 78,310 |
| Excluded because next exact hour was missing | 33 |
| Final-row exclusion | 1 |
| First decision_time | 2017-08-17T05:00:00Z |
| Last decision_time | 2026-07-31T23:00:00Z |
| NaN/Inf targets | 0 |
| Target min | -0.2010332141747841 |
| Target mean | 0.00003251044773681708 |
| Target median | 0.00006772905705562043 |
| Target max | 0.16028033154146137 |
| Every target spans exactly one real hour | yes |
| Targeted tests | 3 passed |
| SHA-256 | `bbf4fbde2d779ce022a0dec603ff1f024257833a79e29f47cce7758194a0c63b` |

## E00E — COMPLETE

The fixed chronological 70% / 15% / 15% split was derived from sorted `decision_time` values.
Train and validation samples whose `target_time` reached the next split period were purged rather
than moved or modified. The frozen boundaries are stored in
`configs/frozen/split_boundaries_v001.yaml`.

| Result | Measured value |
|---|---:|
| Total eligible target rows | 78,310 |
| Train rows | 54,816 |
| Validation rows | 11,745 |
| Test rows | 11,747 |
| Purged train-boundary rows | 1 |
| Purged validation-boundary rows | 1 |
| Train boundary | [2017-08-17T05:00:00Z, 2023-11-26T03:00:00Z) |
| Validation boundary | [2023-11-26T03:00:00Z, 2025-03-29T13:00:00Z) |
| Test boundary | [2025-03-29T13:00:00Z, 2026-07-31T23:00:00Z] |
| Chronological ordering | yes |
| Overlap | 0 |
| Targets crossing boundaries | 0 |
| Targeted tests | 3 passed |
| Full pytest | 52 passed |
| Config validation | PASSED: 26 YAML files, 11 experiments, 3 feature sets, 5 models |

## E00 Data Foundation — COMPLETE

E00A through E00E are complete. The immutable raw archive set, validated canonical datasets,
one-real-hour target, and leakage-safe frozen chronological split are ready for later benchmark
phases. E01 has not started.
