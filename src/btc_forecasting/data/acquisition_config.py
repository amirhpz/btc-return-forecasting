from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from btc_forecasting.common.config import load_yaml


@dataclass(frozen=True)
class DownloadConfig:
    max_workers: int
    retries: int
    resume_partial: bool
    verify_upstream_checksum: bool
    verify_zip_crc: bool
    connect_timeout_seconds: float
    read_timeout_seconds: float
    chunk_size_bytes: int
    backoff_initial_seconds: float
    backoff_max_seconds: float


@dataclass(frozen=True)
class AcquisitionConfig:
    provider: str
    base_url: str
    market: str
    symbol: str
    interval: str
    start_date: date
    end_date_exclusive: date
    archive_frequency: str
    timezone: str
    expected_archive_count: int
    destination: Path
    manifest_path: Path
    summary_path: Path
    timestamp_unit_before_2025: str
    timestamp_unit_from_2025: str
    expected_kline_fields: tuple[str, ...]
    download: DownloadConfig


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(mapping: dict[str, Any], key: str, *, minimum: int) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer greater than or equal to {minimum}")
    return value


def _number(mapping: dict[str, Any], key: str, *, minimum_exclusive: float) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if result <= minimum_exclusive:
        raise ValueError(f"{key} must be greater than {minimum_exclusive}")
    return result


def _boolean(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _date(mapping: dict[str, Any], key: str) -> date:
    value = mapping.get(key)
    if isinstance(value, datetime):
        raise ValueError(f"{key} must be a date without a time component")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{key} must use YYYY-MM-DD format") from exc
    raise ValueError(f"{key} must be a date")


def _relative_path(mapping: dict[str, Any], key: str) -> Path:
    value = _string(mapping, key)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key} must be a repository-relative path without '..'")
    return path


def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must contain only non-empty strings")
    return tuple(value)


def load_acquisition_config(path: Path, *, project_root: Path) -> AcquisitionConfig:
    """Load and validate the E00A acquisition contract."""
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Acquisition config must be inside the project repository") from exc

    document = load_yaml(resolved_path)
    acquisition = _mapping(document.get("data_acquisition"), "data_acquisition")
    download_values = _mapping(acquisition.get("download"), "download")
    timestamp_units = _mapping(
        acquisition.get("upstream_timestamp_units"), "upstream_timestamp_units"
    )

    config = AcquisitionConfig(
        provider=_string(acquisition, "provider"),
        base_url=_string(acquisition, "base_url").rstrip("/"),
        market=_string(acquisition, "market"),
        symbol=_string(acquisition, "symbol"),
        interval=_string(acquisition, "interval"),
        start_date=_date(acquisition, "start_date"),
        end_date_exclusive=_date(acquisition, "end_date_exclusive"),
        archive_frequency=_string(acquisition, "archive_frequency"),
        timezone=_string(acquisition, "timezone"),
        expected_archive_count=_integer(acquisition, "expected_archive_count", minimum=1),
        destination=_relative_path(acquisition, "destination"),
        manifest_path=_relative_path(acquisition, "manifest_path"),
        summary_path=_relative_path(acquisition, "summary_path"),
        timestamp_unit_before_2025=_string(timestamp_units, "before_2025_01_01"),
        timestamp_unit_from_2025=_string(timestamp_units, "from_2025_01_01"),
        expected_kline_fields=_string_tuple(acquisition, "expected_kline_fields"),
        download=DownloadConfig(
            max_workers=_integer(download_values, "max_workers", minimum=1),
            retries=_integer(download_values, "retries", minimum=0),
            resume_partial=_boolean(download_values, "resume_partial"),
            verify_upstream_checksum=_boolean(
                download_values, "verify_upstream_checksum"
            ),
            verify_zip_crc=_boolean(download_values, "verify_zip_crc"),
            connect_timeout_seconds=_number(
                download_values, "connect_timeout_seconds", minimum_exclusive=0
            ),
            read_timeout_seconds=_number(
                download_values, "read_timeout_seconds", minimum_exclusive=0
            ),
            chunk_size_bytes=_integer(download_values, "chunk_size_bytes", minimum=1),
            backoff_initial_seconds=_number(
                download_values, "backoff_initial_seconds", minimum_exclusive=0
            ),
            backoff_max_seconds=_number(
                download_values, "backoff_max_seconds", minimum_exclusive=0
            ),
        ),
    )
    _validate_frozen_contract(config)
    return config


def _validate_frozen_contract(config: AcquisitionConfig) -> None:
    expected_fields = (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    )
    required_values = {
        "provider": (config.provider, "binance_public_data"),
        "base_url": (config.base_url, "https://data.binance.vision"),
        "market": (config.market, "spot"),
        "symbol": (config.symbol, "BTCUSDT"),
        "interval": (config.interval, "5m"),
        "archive_frequency": (config.archive_frequency, "monthly"),
        "timezone": (config.timezone, "UTC"),
        "timestamp unit before 2025": (config.timestamp_unit_before_2025, "milliseconds"),
        "timestamp unit from 2025": (config.timestamp_unit_from_2025, "microseconds"),
    }
    for name, (actual, expected) in required_values.items():
        if actual != expected:
            raise ValueError(f"Frozen E00A {name} must be {expected!r}, got {actual!r}")
    if config.start_date != date(2017, 8, 17):
        raise ValueError("Frozen E00A start_date must be 2017-08-17")
    if config.end_date_exclusive != date(2026, 8, 1):
        raise ValueError("Frozen E00A end_date_exclusive must be 2026-08-01")
    if config.expected_archive_count != 108:
        raise ValueError("Frozen E00A expected_archive_count must be 108")
    if config.expected_kline_fields != expected_fields:
        raise ValueError("expected_kline_fields must match the 12-field Binance kline schema")
    expected_destination = Path("data/raw/binance/spot/monthly/klines/BTCUSDT/5m")
    if config.destination != expected_destination:
        raise ValueError(
            f"Frozen E00A destination must be {expected_destination.as_posix()!r}"
        )
    if config.manifest_path == config.summary_path:
        raise ValueError("manifest_path and summary_path must be different files")
    if not config.manifest_path.parts or config.manifest_path.parts[0] != "outputs":
        raise ValueError("manifest_path must be stored below outputs/")
    if not config.summary_path.parts or config.summary_path.parts[0] != "outputs":
        raise ValueError("summary_path must be stored below outputs/")
    if config.download.max_workers != 1:
        raise ValueError("E00A acquisition is intentionally sequential; max_workers must be 1")
    if not config.download.verify_upstream_checksum:
        raise ValueError("verify_upstream_checksum must remain enabled")
    if not config.download.verify_zip_crc:
        raise ValueError("verify_zip_crc must remain enabled")
    if config.download.backoff_max_seconds < config.download.backoff_initial_seconds:
        raise ValueError("backoff_max_seconds cannot be less than backoff_initial_seconds")
