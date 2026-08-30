from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.acquisition_config import AcquisitionConfig
from btc_forecasting.data.binance_archive import (
    ChecksumFormatError,
    MonthlyPeriod,
    ZipIntegrityError,
    archive_filename,
    archive_url,
    checksum_filename,
    generate_monthly_periods,
    parse_sha256_checksum,
    source_relative_url,
    verify_zip_crc,
)
from btc_forecasting.data.http_transfer import (
    ForbiddenRemoteError,
    HttpClient,
    HttpTransfer,
    InvalidChecksumDocumentError,
    MissingRemoteError,
    TransferError,
)


class Status:
    PLANNED = "PLANNED"
    VERIFIED = "VERIFIED"
    VERIFIED_EXISTING = "VERIFIED_EXISTING"
    MISSING_REMOTE = "MISSING_REMOTE"
    FORBIDDEN_REMOTE = "FORBIDDEN_REMOTE"
    MISSING_LOCAL = "MISSING_LOCAL"
    CHECKSUM_FAILED = "CHECKSUM_FAILED"
    ZIP_FAILED = "ZIP_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"


@dataclass
class ArchiveRecord:
    provider: str
    market: str
    symbol: str
    interval: str
    year: int
    month: int
    archive_filename: str
    checksum_filename: str
    source_relative_url: str
    checksum_source_relative_url: str
    local_relative_path: str
    checksum_local_relative_path: str
    partial_local_relative_path: str
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    size_bytes: int | None = None
    status: str = Status.PLANNED
    attempt_count: int = 0
    checksum_attempt_count: int = 0
    archive_attempt_count: int = 0
    resumed: bool = False
    transfer_result: str = "NOT_RUN"
    verification_result: str = "NOT_RUN"
    checksum_result: str = "NOT_RUN"
    zip_crc_result: str = "NOT_RUN"
    error: str | None = None


@dataclass(frozen=True)
class AcquisitionRunResult:
    exit_code: int
    plan: dict[str, object]
    summary: dict[str, object] | None
    manifest_path: Path | None
    summary_path: Path | None


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


class AcquisitionRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        config_path: Path,
        config: AcquisitionConfig,
        session: HttpClient | None,
        sleep: Callable[[float], None],
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = config_path.resolve()
        self.config = config
        self.transfer = (
            HttpTransfer(config=config.download, session=session, sleep=sleep)
            if session is not None
            else None
        )
        self.periods = generate_monthly_periods(config.start_date, config.end_date_exclusive)
        if len(self.periods) != config.expected_archive_count:
            raise ValueError(
                "Generated archive count does not match expected_archive_count: "
                f"{len(self.periods)} != {config.expected_archive_count}"
            )

    def plan(self) -> dict[str, object]:
        filenames = [
            archive_filename(self.config.symbol, self.config.interval, period)
            for period in self.periods
        ]
        return {
            "phase": "E00A",
            "provider": self.config.provider,
            "market": self.config.market,
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "timezone": self.config.timezone,
            "start_archive_month": self.periods[0].key,
            "end_archive_month": self.periods[-1].key,
            "requested_archive_count": len(self.periods),
            "destination": self.config.destination.as_posix(),
            "expected_filenames": filenames,
        }

    def run(self, *, verify_only: bool) -> AcquisitionRunResult:
        records = [self._planned_record(period) for period in self.periods]
        manifest_relative_path = self.config.manifest_path
        summary_relative_path = self.config.summary_path
        if verify_only:
            manifest_relative_path = manifest_relative_path.with_name("verification_manifest.json")
            summary_relative_path = summary_relative_path.with_name("verification_summary.json")
        manifest_path = self._resolve_project_path(manifest_relative_path)
        summary_path = self._resolve_project_path(summary_relative_path)
        self._write_runtime_outputs(records, manifest_path, summary_path, verify_only=verify_only)

        for index, period in enumerate(self.periods):
            records[index] = self._process_period(period, verify_only=verify_only)
            self._write_runtime_outputs(
                records,
                manifest_path,
                summary_path,
                verify_only=verify_only,
            )

        summary = self._summary(records, verify_only=verify_only)
        return AcquisitionRunResult(
            exit_code=0 if summary["completion_status"] == "COMPLETE" else 1,
            plan=self.plan(),
            summary=summary,
            manifest_path=manifest_path,
            summary_path=summary_path,
        )

    def _planned_record(self, period: MonthlyPeriod) -> ArchiveRecord:
        filename = archive_filename(self.config.symbol, self.config.interval, period)
        checksum_name = checksum_filename(self.config.symbol, self.config.interval, period)
        archive_relative_path = self.config.destination / filename
        checksum_relative_path = self.config.destination / checksum_name
        source_path = source_relative_url(self.config.symbol, self.config.interval, period)
        return ArchiveRecord(
            provider=self.config.provider,
            market=self.config.market,
            symbol=self.config.symbol,
            interval=self.config.interval,
            year=period.year,
            month=period.month,
            archive_filename=filename,
            checksum_filename=checksum_name,
            source_relative_url=source_path,
            checksum_source_relative_url=f"{source_path}.CHECKSUM",
            local_relative_path=archive_relative_path.as_posix(),
            checksum_local_relative_path=checksum_relative_path.as_posix(),
            partial_local_relative_path=Path(f"{archive_relative_path}.part").as_posix(),
        )

    def _process_period(self, period: MonthlyPeriod, *, verify_only: bool) -> ArchiveRecord:
        record = self._planned_record(period)
        archive_path = self._resolve_project_path(Path(record.local_relative_path))
        checksum_path = self._resolve_project_path(Path(record.checksum_local_relative_path))
        partial_path = self._resolve_project_path(Path(record.partial_local_relative_path))

        try:
            expected_sha256 = self._obtain_expected_checksum(
                period,
                checksum_path,
                record,
                verify_only=verify_only,
            )
            if expected_sha256 is None:
                return record
            record.expected_sha256 = expected_sha256

            if archive_path.is_file():
                return self._verify_archive(
                    archive_path,
                    record,
                    expected_sha256=expected_sha256,
                    existing=True,
                )
            if verify_only:
                record.status = Status.MISSING_LOCAL
                record.checksum_result = "PARSED"
                record.verification_result = "INVALID"
                record.error = f"Missing local archive {record.archive_filename}"
                return record

            archive_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                transfer = self._require_transfer().download_archive(
                    archive_url(
                        self.config.base_url,
                        self.config.symbol,
                        self.config.interval,
                        period,
                    ),
                    partial_path,
                    resource_name=record.archive_filename,
                )
            except TransferError as exc:
                record.archive_attempt_count += exc.attempt_count
                raise
            record.archive_attempt_count = transfer.attempt_count
            record.attempt_count += transfer.attempt_count
            record.resumed = transfer.resumed
            record.transfer_result = transfer.detail
            verified = self._verify_archive(
                partial_path,
                record,
                expected_sha256=expected_sha256,
                existing=False,
            )
            if verified.status != Status.VERIFIED:
                return verified
            os.replace(partial_path, archive_path)
            verified.size_bytes = archive_path.stat().st_size
            return verified
        except MissingRemoteError as exc:
            record.attempt_count += exc.attempt_count
            record.status = Status.MISSING_REMOTE
            record.verification_result = "INVALID"
            record.error = str(exc)
        except ForbiddenRemoteError as exc:
            record.attempt_count += exc.attempt_count
            record.status = Status.FORBIDDEN_REMOTE
            record.verification_result = "INVALID"
            record.error = str(exc)
        except InvalidChecksumDocumentError as exc:
            record.attempt_count += exc.attempt_count
            record.status = Status.CHECKSUM_FAILED
            record.checksum_result = "INVALID"
            record.verification_result = "INVALID"
            record.error = str(exc)
        except TransferError as exc:
            record.attempt_count += exc.attempt_count
            record.status = Status.DOWNLOAD_FAILED
            record.verification_result = "INVALID"
            record.error = str(exc)
        return record

    def _obtain_expected_checksum(
        self,
        period: MonthlyPeriod,
        checksum_path: Path,
        record: ArchiveRecord,
        *,
        verify_only: bool,
    ) -> str | None:
        if checksum_path.is_file():
            try:
                checksum = parse_sha256_checksum(
                    checksum_path.read_text(encoding="utf-8"),
                    expected_filename=record.archive_filename,
                )
                record.checksum_result = "PARSED"
                return checksum
            except (OSError, UnicodeError, ChecksumFormatError) as exc:
                if verify_only:
                    record.status = Status.CHECKSUM_FAILED
                    record.checksum_result = "INVALID"
                    record.verification_result = "INVALID"
                    record.error = f"Invalid local checksum {record.checksum_filename}: {exc}"
                    return None

        if verify_only:
            record.status = Status.MISSING_LOCAL
            record.checksum_result = "MISSING"
            record.verification_result = "INVALID"
            record.error = f"Missing local checksum {record.checksum_filename}"
            return None

        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_url = (
            f"{archive_url(self.config.base_url, self.config.symbol, self.config.interval, period)}"
            ".CHECKSUM"
        )
        try:
            checksum, attempts = self._require_transfer().download_checksum(
                checksum_url,
                checksum_path,
                archive_name=record.archive_filename,
                resource_name=record.checksum_filename,
            )
        except TransferError as exc:
            record.checksum_attempt_count += exc.attempt_count
            raise
        record.checksum_attempt_count = attempts
        record.attempt_count += attempts
        record.checksum_result = "PARSED"
        return checksum

    def _verify_archive(
        self,
        path: Path,
        record: ArchiveRecord,
        *,
        expected_sha256: str,
        existing: bool,
    ) -> ArchiveRecord:
        record.size_bytes = path.stat().st_size
        record.actual_sha256 = sha256_file(
            path,
            chunk_size=self.config.download.chunk_size_bytes,
        )
        if record.actual_sha256 != expected_sha256:
            record.status = Status.CHECKSUM_FAILED
            record.checksum_result = "MISMATCH"
            record.verification_result = "INVALID"
            record.error = "Local SHA-256 does not match the official checksum"
            return record
        record.checksum_result = "MATCH"

        try:
            verify_zip_crc(path)
        except ZipIntegrityError as exc:
            record.status = Status.ZIP_FAILED
            record.zip_crc_result = "FAIL"
            record.verification_result = "INVALID"
            record.error = str(exc)
            return record

        record.status = Status.VERIFIED_EXISTING if existing else Status.VERIFIED
        record.zip_crc_result = "PASS"
        record.verification_result = "VALID"
        if existing:
            record.transfer_result = "EXISTING"
        return record

    def _require_transfer(self) -> HttpTransfer:
        if self.transfer is None:
            raise RuntimeError("Network transfer is unavailable in this acquisition mode")
        return self.transfer

    def _resolve_project_path(self, relative_path: Path) -> Path:
        path = (self.project_root / relative_path).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"Path escapes project root: {relative_path.as_posix()}") from exc
        return path

    def _write_runtime_outputs(
        self,
        records: list[ArchiveRecord],
        manifest_path: Path,
        summary_path: Path,
        *,
        verify_only: bool,
    ) -> None:
        generated_at = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": 1,
            "phase": "E00A",
            "mode": "verify_only" if verify_only else "acquire",
            "generated_at_utc": generated_at,
            "config_path": self.config_path.relative_to(self.project_root).as_posix(),
            "contract": {
                "provider": self.config.provider,
                "market": self.config.market,
                "symbol": self.config.symbol,
                "interval": self.config.interval,
                "start_date": self.config.start_date.isoformat(),
                "end_date_exclusive": self.config.end_date_exclusive.isoformat(),
                "archive_frequency": self.config.archive_frequency,
                "timezone": self.config.timezone,
                "timestamp_units": {
                    "before_2025_01_01": self.config.timestamp_unit_before_2025,
                    "from_2025_01_01": self.config.timestamp_unit_from_2025,
                },
            },
            "entries": [asdict(record) for record in records],
        }
        summary = self._summary(records, verify_only=verify_only)
        summary["generated_at_utc"] = generated_at
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_json(summary_path, summary)

    def _summary(
        self,
        records: list[ArchiveRecord],
        *,
        verify_only: bool,
    ) -> dict[str, object]:
        valid_statuses = {Status.VERIFIED, Status.VERIFIED_EXISTING}
        verified = [record for record in records if record.status in valid_statuses]
        existing = [record for record in records if record.status == Status.VERIFIED_EXISTING]
        downloaded = [
            record
            for record in records
            if record.status == Status.VERIFIED and not record.resumed
        ]
        resumed = [
            record for record in records if record.status == Status.VERIFIED and record.resumed
        ]
        return {
            "schema_version": 1,
            "phase": "E00A",
            "mode": "verify_only" if verify_only else "acquire",
            "requested_archive_count": len(records),
            "verified_archive_count": len(verified),
            "existing_verified_count": len(existing),
            "downloaded_count": len(downloaded),
            "resumed_count": len(resumed),
            "failed_count": len(records) - len(verified),
            "missing_remote_count": sum(
                record.status == Status.MISSING_REMOTE for record in records
            ),
            "forbidden_remote_count": sum(
                record.status == Status.FORBIDDEN_REMOTE for record in records
            ),
            "missing_local_count": sum(
                record.status == Status.MISSING_LOCAL for record in records
            ),
            "checksum_failure_count": sum(
                record.status == Status.CHECKSUM_FAILED for record in records
            ),
            "zip_failure_count": sum(
                record.status == Status.ZIP_FAILED for record in records
            ),
            "download_failure_count": sum(
                record.status == Status.DOWNLOAD_FAILED for record in records
            ),
            "total_bytes": sum(record.size_bytes or 0 for record in verified),
            "start_month": self.periods[0].key,
            "end_month": self.periods[-1].key,
            "completion_status": "COMPLETE" if len(verified) == len(records) else "INCOMPLETE",
        }


def run_acquisition(
    *,
    project_root: Path,
    config_path: Path,
    config: AcquisitionConfig,
    dry_run: bool = False,
    verify_only: bool = False,
    session: HttpClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AcquisitionRunResult:
    """Run E00A without placing business logic in the CLI."""
    if dry_run and verify_only:
        raise ValueError("dry_run and verify_only are mutually exclusive")

    owned_session: requests.Session | None = None
    client = session
    if not dry_run and not verify_only and client is None:
        owned_session = requests.Session()
        client = owned_session

    runner = AcquisitionRunner(
        project_root=project_root,
        config_path=config_path,
        config=config,
        session=client,
        sleep=sleep,
    )
    if dry_run:
        return AcquisitionRunResult(
            exit_code=0,
            plan=runner.plan(),
            summary=None,
            manifest_path=None,
            summary_path=None,
        )

    try:
        return runner.run(verify_only=verify_only)
    finally:
        if owned_session is not None:
            owned_session.close()
