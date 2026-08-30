from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path


class ChecksumFormatError(ValueError):
    """Raised when an upstream checksum document is ambiguous or malformed."""


class ZipIntegrityError(ValueError):
    """Raised when a ZIP cannot be opened or fails its member CRC check."""


@dataclass(frozen=True, order=True)
class MonthlyPeriod:
    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def generate_monthly_periods(
    start_date: date,
    end_date_exclusive: date,
) -> tuple[MonthlyPeriod, ...]:
    """Return months intersecting a half-open date range in deterministic order."""
    if start_date >= end_date_exclusive:
        raise ValueError("start_date must be before end_date_exclusive")

    current = date(start_date.year, start_date.month, 1)
    periods: list[MonthlyPeriod] = []
    while current < end_date_exclusive:
        periods.append(MonthlyPeriod(current.year, current.month))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return tuple(periods)


def archive_filename(symbol: str, interval: str, period: MonthlyPeriod) -> str:
    return f"{symbol}-{interval}-{period.key}.zip"


def checksum_filename(symbol: str, interval: str, period: MonthlyPeriod) -> str:
    return f"{archive_filename(symbol, interval, period)}.CHECKSUM"


def source_relative_url(symbol: str, interval: str, period: MonthlyPeriod) -> str:
    filename = archive_filename(symbol, interval, period)
    return f"/data/spot/monthly/klines/{symbol}/{interval}/{filename}"


def archive_url(base_url: str, symbol: str, interval: str, period: MonthlyPeriod) -> str:
    return f"{base_url.rstrip('/')}{source_relative_url(symbol, interval, period)}"


def parse_sha256_checksum(text: str, *, expected_filename: str) -> str:
    """Parse one sha256sum-style line and bind it to the expected archive name."""
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        raise ChecksumFormatError("Checksum document must contain exactly one non-empty line")
    match = re.fullmatch(r"([0-9A-Fa-f]{64})\s+\*?(\S+)", nonempty_lines[0])
    if match is None:
        raise ChecksumFormatError("Checksum document is not valid sha256sum format")
    checksum, stated_filename = match.groups()
    if Path(stated_filename).name != expected_filename:
        raise ChecksumFormatError(
            f"Checksum names {stated_filename!r}, expected {expected_filename!r}"
        )
    return checksum.lower()


def verify_zip_crc(path: Path) -> None:
    """Open an archive and verify the CRC of every member without extracting it."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            failed_member = archive.testzip()
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ZipIntegrityError(f"ZIP integrity check could not read {path.name}: {exc}") from exc
    if failed_member is not None:
        raise ZipIntegrityError(f"ZIP CRC failed for member {failed_member!r}")
