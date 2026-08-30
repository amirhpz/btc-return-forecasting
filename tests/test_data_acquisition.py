from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.acquisition import AcquisitionRunResult, run_acquisition
from btc_forecasting.data.acquisition_config import (
    AcquisitionConfig,
    DownloadConfig,
    load_acquisition_config,
)
from btc_forecasting.data.binance_archive import (
    ChecksumFormatError,
    MonthlyPeriod,
    archive_filename,
    archive_url,
    checksum_filename,
    generate_monthly_periods,
    parse_sha256_checksum,
    source_relative_url,
)
from btc_forecasting.data.http_transfer import HttpResponse


KLINE_FIELDS = (
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


@dataclass
class FakeResponse:
    status_code: int
    body: bytes = b""
    response_headers: dict[str, str] | None = None

    @property
    def headers(self) -> Mapping[str, str]:
        if self.response_headers is not None:
            return self.response_headers
        return {"Content-Length": str(len(self.body))}

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def close(self) -> None:
        return None


class FakeSession:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str], tuple[float, float]]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: tuple[float, float],
    ) -> HttpResponse:
        assert stream is True
        self.calls.append((url, dict(headers), timeout))
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class NoNetworkSession:
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: tuple[float, float],
    ) -> HttpResponse:
        raise AssertionError(f"Network access was forbidden, but requested {url}")


def _config(root: Path, *, retries: int = 2) -> AcquisitionConfig:
    return AcquisitionConfig(
        provider="binance_public_data",
        base_url="https://data.binance.vision",
        market="spot",
        symbol="BTCUSDT",
        interval="5m",
        start_date=date(2020, 1, 1),
        end_date_exclusive=date(2020, 2, 1),
        archive_frequency="monthly",
        timezone="UTC",
        expected_archive_count=1,
        destination=Path("data/raw/binance/spot/monthly/klines/BTCUSDT/5m"),
        manifest_path=Path("outputs/data/e00a/acquisition_manifest.json"),
        summary_path=Path("outputs/data/e00a/acquisition_summary.json"),
        timestamp_unit_before_2025="milliseconds",
        timestamp_unit_from_2025="microseconds",
        expected_kline_fields=KLINE_FIELDS,
        download=DownloadConfig(
            max_workers=1,
            retries=retries,
            resume_partial=True,
            verify_upstream_checksum=True,
            verify_zip_crc=True,
            connect_timeout_seconds=15,
            read_timeout_seconds=120,
            chunk_size_bytes=7,
            backoff_initial_seconds=0.001,
            backoff_max_seconds=0.001,
        ),
    )


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-5m-2020-01.csv", "1,2,3\n")
    return buffer.getvalue()


def _paths(root: Path, config: AcquisitionConfig) -> tuple[Path, Path, Path]:
    filename = "BTCUSDT-5m-2020-01.zip"
    archive = root / config.destination / filename
    checksum = root / config.destination / f"{filename}.CHECKSUM"
    partial = Path(f"{archive}.part")
    return archive, checksum, partial


def _write_checksum(path: Path, archive_name: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{digest}  {archive_name}\n", encoding="utf-8")
    return digest


def _run(
    root: Path,
    config: AcquisitionConfig,
    session: FakeSession | NoNetworkSession,
    *,
    dry_run: bool = False,
    verify_only: bool = False,
) -> AcquisitionRunResult:
    config_path = root / "config.yaml"
    return run_acquisition(
        project_root=root,
        config_path=config_path,
        config=config,
        dry_run=dry_run,
        verify_only=verify_only,
        session=session,
        sleep=lambda _: None,
    )


def test_monthly_period_generation_crosses_year_boundary() -> None:
    periods = generate_monthly_periods(date(2020, 11, 17), date(2021, 2, 1))
    assert [period.key for period in periods] == ["2020-11", "2020-12", "2021-01"]


def test_frozen_range_has_exactly_108_periods() -> None:
    periods = generate_monthly_periods(date(2017, 8, 17), date(2026, 8, 1))
    assert len(periods) == 108
    assert periods[0] == MonthlyPeriod(2017, 8)
    assert periods[-1] == MonthlyPeriod(2026, 7)


def test_official_archive_names_paths_and_url() -> None:
    period = MonthlyPeriod(2026, 7)
    assert archive_filename("BTCUSDT", "5m", period) == "BTCUSDT-5m-2026-07.zip"
    assert checksum_filename("BTCUSDT", "5m", period) == "BTCUSDT-5m-2026-07.zip.CHECKSUM"
    relative = source_relative_url("BTCUSDT", "5m", period)
    assert relative == "/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2026-07.zip"
    assert archive_url("https://data.binance.vision", "BTCUSDT", "5m", period) == (
        "https://data.binance.vision" + relative
    )


def test_checksum_parser_accepts_bound_filename_and_normalizes_case() -> None:
    digest = "AB" * 32
    assert parse_sha256_checksum(
        f"{digest} *BTCUSDT-5m-2020-01.zip\n",
        expected_filename="BTCUSDT-5m-2020-01.zip",
    ) == digest.lower()


def test_checksum_parser_rejects_wrong_filename() -> None:
    with pytest.raises(ChecksumFormatError):
        parse_sha256_checksum(
            f"{'0' * 64}  another.zip\n",
            expected_filename="BTCUSDT-5m-2020-01.zip",
        )


def test_streaming_sha256_calculation(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"abc" * 100)
    assert sha256_file(path, chunk_size=5) == hashlib.sha256(b"abc" * 100).hexdigest()


def test_valid_existing_archive_is_verified_without_network(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, _ = _paths(tmp_path, config)
    payload = _zip_bytes()
    _write_checksum(checksum, archive.name, payload)
    archive.write_bytes(payload)

    result = _run(tmp_path, config, NoNetworkSession())

    assert result.exit_code == 0
    assert result.summary is not None
    assert result.summary["existing_verified_count"] == 1
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    assert manifest["entries"][0]["status"] == "VERIFIED_EXISTING"


def test_existing_archive_with_bad_checksum_is_not_replaced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, _ = _paths(tmp_path, config)
    payload = _zip_bytes()
    checksum.parent.mkdir(parents=True, exist_ok=True)
    checksum.write_text(f"{'0' * 64}  {archive.name}\n", encoding="utf-8")
    archive.write_bytes(payload)

    result = _run(tmp_path, config, NoNetworkSession())

    assert result.exit_code == 1
    assert archive.read_bytes() == payload
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    assert manifest["entries"][0]["status"] == "CHECKSUM_FAILED"


def test_successful_download_uses_part_then_atomic_final_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, partial = _paths(tmp_path, config)
    payload = _zip_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    checksum_body = f"{digest}  {archive.name}\n".encode()
    session = FakeSession([FakeResponse(200, checksum_body), FakeResponse(200, payload)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 0
    assert archive.read_bytes() == payload
    assert checksum.read_bytes() == checksum_body
    assert not partial.exists()
    assert not Path(f"{checksum}.part").exists()


def test_checksum_failure_leaves_part_and_never_exposes_final_zip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, partial = _paths(tmp_path, config)
    payload = _zip_bytes()
    _write_checksum(checksum, archive.name, b"different bytes")
    session = FakeSession([FakeResponse(200, payload)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 1
    assert not archive.exists()
    assert partial.read_bytes() == payload
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    assert manifest["entries"][0]["status"] == "CHECKSUM_FAILED"


def test_range_resume_appends_only_valid_partial_response(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, partial = _paths(tmp_path, config)
    payload = _zip_bytes()
    split = 11
    _write_checksum(checksum, archive.name, payload)
    partial.write_bytes(payload[:split])
    remainder = payload[split:]
    headers = {
        "Content-Range": f"bytes {split}-{len(payload) - 1}/{len(payload)}",
        "Content-Length": str(len(remainder)),
    }
    session = FakeSession([FakeResponse(206, remainder, headers)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 0
    assert archive.read_bytes() == payload
    assert session.calls[0][1] == {
        "Accept-Encoding": "identity",
        "Range": f"bytes={split}-",
    }
    assert result.summary is not None
    assert result.summary["resumed_count"] == 1


def test_server_ignoring_range_restarts_only_partial_archive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, partial = _paths(tmp_path, config)
    payload = _zip_bytes()
    _write_checksum(checksum, archive.name, payload)
    partial.write_bytes(b"incomplete-prefix")
    session = FakeSession([FakeResponse(200, payload)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 0
    assert archive.read_bytes() == payload
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    assert entry["resumed"] is False
    assert entry["transfer_result"] == "RANGE_IGNORED_RESTARTED"


def test_http_404_is_recorded_distinctly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, _ = _paths(tmp_path, config)
    _write_checksum(checksum, archive.name, _zip_bytes())
    session = FakeSession([FakeResponse(404)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 1
    assert result.summary is not None
    assert result.summary["missing_remote_count"] == 1
    assert len(session.calls) == 1


def test_http_403_is_recorded_distinctly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, _ = _paths(tmp_path, config)
    _write_checksum(checksum, archive.name, _zip_bytes())
    session = FakeSession([FakeResponse(403)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 1
    assert result.summary is not None
    assert result.summary["forbidden_remote_count"] == 1
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    assert manifest["entries"][0]["status"] == "FORBIDDEN_REMOTE"


def test_retry_exhaustion_is_bounded(tmp_path: Path) -> None:
    config = _config(tmp_path, retries=2)
    archive, checksum, _ = _paths(tmp_path, config)
    _write_checksum(checksum, archive.name, _zip_bytes())
    session = FakeSession([FakeResponse(503), FakeResponse(503), FakeResponse(503)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 1
    assert len(session.calls) == 3
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    assert entry["status"] == "DOWNLOAD_FAILED"
    assert entry["archive_attempt_count"] == 3
    assert entry["attempt_count"] == 3


def test_zip_crc_failure_never_finalizes_archive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, partial = _paths(tmp_path, config)
    payload = b"not a zip archive"
    _write_checksum(checksum, archive.name, payload)
    session = FakeSession([FakeResponse(200, payload)])

    result = _run(tmp_path, config, session)

    assert result.exit_code == 1
    assert not archive.exists()
    assert partial.exists()
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    assert entry["status"] == "ZIP_FAILED"
    assert entry["zip_crc_result"] == "FAIL"


def test_dry_run_has_no_network_and_writes_no_runtime_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = _run(tmp_path, config, NoNetworkSession(), dry_run=True)

    assert result.exit_code == 0
    assert result.plan["requested_archive_count"] == 1
    assert result.manifest_path is None
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "data").exists()


def test_verify_only_never_downloads_missing_files(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = _run(tmp_path, config, NoNetworkSession(), verify_only=True)

    assert result.exit_code == 1
    assert result.summary is not None
    assert result.summary["missing_local_count"] == 1
    assert not (tmp_path / config.destination).exists()


def test_acquisition_and_verification_artifacts_do_not_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, _ = _paths(tmp_path, config)
    payload = _zip_bytes()
    _write_checksum(checksum, archive.name, payload)
    archive.write_bytes(payload)

    acquisition = _run(tmp_path, config, NoNetworkSession())
    assert acquisition.manifest_path == tmp_path / "outputs/data/e00a/acquisition_manifest.json"
    assert acquisition.summary_path == tmp_path / "outputs/data/e00a/acquisition_summary.json"
    acquisition_manifest = acquisition.manifest_path.read_bytes()
    acquisition_summary = acquisition.summary_path.read_bytes()

    verification = _run(tmp_path, config, NoNetworkSession(), verify_only=True)

    assert verification.manifest_path == tmp_path / "outputs/data/e00a/verification_manifest.json"
    assert verification.summary_path == tmp_path / "outputs/data/e00a/verification_summary.json"
    assert verification.manifest_path.is_file()
    assert verification.summary_path.is_file()
    assert acquisition.manifest_path.read_bytes() == acquisition_manifest
    assert acquisition.summary_path.read_bytes() == acquisition_summary


def test_manifest_contains_only_repository_relative_local_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive, checksum, _ = _paths(tmp_path, config)
    payload = _zip_bytes()
    _write_checksum(checksum, archive.name, payload)
    archive.write_bytes(payload)

    result = _run(tmp_path, config, NoNetworkSession(), verify_only=True)

    assert result.manifest_path is not None
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    entry = manifest["entries"][0]
    assert not Path(entry["local_relative_path"]).is_absolute()
    assert not Path(entry["checksum_local_relative_path"]).is_absolute()
    assert str(tmp_path) not in manifest_text


def test_frozen_acquisition_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_acquisition_config(root / "configs/data_acquisition.yaml", project_root=root)
    assert config.start_date == date(2017, 8, 17)
    assert config.end_date_exclusive == date(2026, 8, 1)
    assert config.expected_archive_count == 108
    assert config.download.max_workers == 1
    assert config.destination.as_posix().endswith("BTCUSDT/5m")


def test_half_open_range_excludes_cutoff_month() -> None:
    periods = generate_monthly_periods(date(2026, 6, 30), date(2026, 8, 1))
    assert [period.key for period in periods] == ["2026-06", "2026-07"]
