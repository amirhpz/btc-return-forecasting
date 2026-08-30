# Experiment Log

| Experiment | Run ID | Date UTC | Data Version | Split Version | Feature Set | Model | Seed | Evaluation Split | Decision | Notes |
|---|---|---|---|---|---|---|---:|---|---|---|
| E00 | | 2026-08-30 | Binance monthly archives | | | | | none | E00A COMPLETE; E00B NOT STARTED | All 108 source archives acquired and independently verified. |
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
confirmed all 108 local archives, with no missing or failed archive. E00B has not started.
