# E00A Canonical Data Acquisition

## Scope Boundary

E00A acquires and verifies immutable upstream archive files. It does not read kline rows,
normalize timestamps, validate candles, concatenate CSV files, create Parquet files, resample to
one hour, construct targets, or create data splits. Those operations begin with E00B and later
controlled phases.

## Frozen Acquisition Contract

| Item | Value |
|---|---|
| Provider | Binance Public Data |
| Host | `https://data.binance.vision` |
| Market | Binance Spot |
| Symbol | `BTCUSDT` |
| Interval | `5m` |
| Timezone | UTC |
| Requested start | `2017-08-17` |
| End, exclusive | `2026-08-01T00:00:00Z` |
| Archive frequency | Monthly |
| First archive month | `2017-08` |
| Last archive month | `2026-07` |
| Expected archive count | 108 |

The source of truth is `configs/data_acquisition.yaml`. The downloader rejects changes that would
violate this frozen E00A contract. It does not use the current date, a rolling cutoff, a REST API,
or another data provider as a fallback.

Monthly official archives are used because their filenames and upstream checksum artifacts are
stable, auditable provenance objects. If an expected official resource returns HTTP 404, the run
records `MISSING_REMOTE` and remains incomplete. HTTP 403 is recorded separately as
`FORBIDDEN_REMOTE`. No substitute source is attempted.

## Official Paths

For each requested month, E00A requests exactly these resources:

```text
/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-YYYY-MM.zip
/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-YYYY-MM.zip.CHECKSUM
```

The local provenance layout is:

```text
data/raw/binance/spot/monthly/klines/BTCUSDT/5m/
  BTCUSDT-5m-2017-08.zip
  BTCUSDT-5m-2017-08.zip.CHECKSUM
  ...
  BTCUSDT-5m-2026-07.zip
  BTCUSDT-5m-2026-07.zip.CHECKSUM
```

These files are excluded from Git. They must remain byte-for-byte copies of the official source
artifacts and must not be edited, extracted over, concatenated, or deleted after later processing.

## Expected Upstream Kline Schema

E00A does not parse these fields, but records the expected 12-field order for E00B:

1. `open_time`
2. `open`
3. `high`
4. `low`
5. `close`
6. `volume`
7. `close_time`
8. `quote_asset_volume`
9. `number_of_trades`
10. `taker_buy_base_volume`
11. `taker_buy_quote_volume`
12. `ignore`

## Binance Timestamp Units

Binance Spot public archive timestamp fields change units at a fixed boundary:

- before `2025-01-01T00:00:00Z`: milliseconds;
- from `2025-01-01T00:00:00Z`: microseconds.

E00A preserves the bytes and records this rule only. E00B must apply the boundary when parsing and
must not assume milliseconds for the complete history.

## Download and Verification Sequence

For a missing local month, the acquisition runner:

1. downloads the official checksum to a temporary `.CHECKSUM.part` file;
2. parses exactly one SHA-256 line bound to the expected ZIP filename;
3. atomically replaces the final `.CHECKSUM` only after successful parsing;
4. requests the identity representation and streams the ZIP into `filename.zip.part`;
5. checks response length whenever `Content-Length` or `Content-Range` provides it;
6. calculates the local ZIP SHA-256 and compares it exactly with the official checksum;
7. opens the ZIP and runs the standard-library member CRC check without extracting it;
8. atomically replaces `.zip.part` with `.zip` only after both checks pass.

A checksum mismatch produces `CHECKSUM_FAILED`. A ZIP open or member CRC failure produces
`ZIP_FAILED`. The `.part` remains diagnostic evidence and the final `.zip` name is never exposed.
An existing final ZIP that fails verification is not overwritten automatically.

## Resume and Retry Policy

An existing `.zip.part` is measured and resumed with `Range: bytes=<size>-`. Data is appended only
when the server returns HTTP 206 with a consistent `Content-Range` starting at exactly that byte.
If the server ignores Range and returns HTTP 200, only that incomplete `.part` is reopened in write
mode and safely restarted; a full response is never appended to it.

Connection failures, HTTP 429, and HTTP 5xx responses use bounded exponential backoff. The frozen
configuration permits five retries after the initial attempt, caps the delay, and acquires one
archive at a time. HTTP 403, HTTP 404, other unexpected HTTP statuses, malformed checksum files,
and unsafe Range metadata fail explicitly rather than retrying without bound.

## Idempotency

When a final ZIP and saved official checksum both exist, E00A recalculates SHA-256 and runs ZIP CRC.
If both succeed, the month is recorded as `VERIFIED_EXISTING` and no network request is made for
that month. Re-running acquisition is therefore safe and avoids replacing verified raw data.

## Modes

Dry-run performs no network access and writes no manifest or fake success artifact:

```powershell
btc-forecast acquire-data --config configs/data_acquisition.yaml --dry-run
```

It reports the provider, market, symbol, interval, destination, first and last month, all expected
filenames, and the expected count of 108.

Normal acquisition downloads or resumes missing archives and verifies every month:

```powershell
btc-forecast acquire-data --config configs/data_acquisition.yaml
```

Verify-only performs no downloads. It checks the expected local checksum and ZIP pair for all 108
months:

```powershell
btc-forecast acquire-data --config configs/data_acquisition.yaml --verify-only
```

Normal and verify-only modes exit non-zero while any expected month is missing or invalid.

## Runtime Manifest and Summary

Normal acquisition atomically updates:

```text
outputs/data/e00a/acquisition_manifest.json
outputs/data/e00a/acquisition_summary.json
```

Verify-only atomically updates separate artifacts so it cannot overwrite the acquisition record:

```text
outputs/data/e00a/verification_manifest.json
outputs/data/e00a/verification_summary.json
```

The manifest contains one stable entry per requested month, official relative source paths,
repository-relative local paths, checksums, byte size, network attempt counts, resume outcome,
checksum result, ZIP result, overall verification result, and explicit status. It does not persist
absolute local paths.

The summary reports requested, verified, existing, downloaded, resumed, failed, missing, checksum
failure, ZIP failure, total verified bytes, month boundaries, and overall completion status. Both
runtime files are excluded from Git through the existing `outputs/**` policy.

## Failure Recovery

- `MISSING_REMOTE`: retain the manifest and investigate availability on the official host. Do not
  substitute another source.
- `FORBIDDEN_REMOTE`: retain the manifest and investigate official-host access. Do not bypass the
  source contract.
- `DOWNLOAD_FAILED`: rerun the same command. A valid partial transfer is resumed when the server
  supports Range.
- `CHECKSUM_FAILED` or `ZIP_FAILED` on a `.part`: preserve the manifest diagnostics. After review,
  remove or quarantine only the named invalid `.part`, then rerun acquisition.
- Invalid existing final `.zip`: preserve it for investigation; do not let acquisition overwrite it
  automatically. After confirming the exact path and retaining evidence, the owner may quarantine
  that one file and rerun.
- Invalid saved `.CHECKSUM`: normal mode retrieves a fresh official checksum into a temporary file
  and replaces the invalid saved checksum only after the new document parses correctly.

Do not proceed to E00B until the acquisition summary reports all 108 archives verified and the
owner has supplied the runtime verification evidence.
