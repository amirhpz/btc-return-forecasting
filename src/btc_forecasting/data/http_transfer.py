from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import requests

from btc_forecasting.data.acquisition_config import DownloadConfig
from btc_forecasting.data.binance_archive import ChecksumFormatError, parse_sha256_checksum


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: tuple[float, float],
    ) -> HttpResponse: ...


class TransferError(RuntimeError):
    def __init__(self, message: str, *, attempt_count: int) -> None:
        super().__init__(message)
        self.attempt_count = attempt_count


class MissingRemoteError(TransferError):
    """Raised for an explicit upstream HTTP 404."""


class ForbiddenRemoteError(TransferError):
    """Raised for an explicit upstream HTTP 403."""


class RetryExhaustedError(TransferError):
    """Raised after all bounded transfer attempts fail."""


class InvalidChecksumDocumentError(TransferError):
    """Raised when a downloaded upstream checksum cannot be trusted."""


class ProtocolResponseError(TransferError):
    """Raised when HTTP response metadata makes resumption unsafe."""


class _RetriableTransferError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransferResult:
    attempt_count: int
    resumed: bool
    detail: str


def _header_integer(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name} header: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Invalid {name} header: {value!r}")
    return parsed


def _content_range(headers: Mapping[str, str]) -> tuple[int, int, int]:
    value = headers.get("Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
    if match is None:
        raise ValueError(f"Invalid Content-Range header: {value!r}")
    start, end, total = (int(group) for group in match.groups())
    if end < start or total <= end:
        raise ValueError(f"Inconsistent Content-Range header: {value!r}")
    return start, end, total


def _unsatisfied_range_total(headers: Mapping[str, str]) -> int | None:
    value = headers.get("Content-Range")
    match = re.fullmatch(r"bytes \*/(\d+)", value or "")
    return int(match.group(1)) if match is not None else None


class HttpTransfer:
    """Stream official archive resources with bounded, resume-safe retries."""

    def __init__(
        self,
        *,
        config: DownloadConfig,
        session: HttpClient,
        sleep: Callable[[float], None],
    ) -> None:
        self.config = config
        self.session = session
        self.sleep = sleep

    def download_checksum(
        self,
        url: str,
        destination: Path,
        *,
        archive_name: str,
        resource_name: str,
    ) -> tuple[str, int]:
        temporary_path = Path(f"{destination}.part")
        max_attempts = self.config.retries + 1
        failure: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            response: HttpResponse | None = None
            try:
                response = self.session.get(
                    url,
                    headers={"Accept-Encoding": "identity"},
                    stream=True,
                    timeout=self.timeout,
                )
                self._require_download_status(
                    response.status_code,
                    resource_name=resource_name,
                    attempt=attempt,
                )
                if response.status_code != 200:
                    raise ProtocolResponseError(
                        f"Unexpected partial checksum response for {resource_name}",
                        attempt_count=attempt,
                    )
                expected_length = self._content_length(
                    response.headers,
                    resource_name=resource_name,
                    attempt=attempt,
                )
                bytes_written = self._write_response(response, temporary_path, mode="wb")
                if expected_length is not None and bytes_written != expected_length:
                    raise _RetriableTransferError(
                        f"Short checksum response for {resource_name}: "
                        f"expected {expected_length} bytes, received {bytes_written}"
                    )
                try:
                    text = temporary_path.read_text(encoding="utf-8")
                    checksum = parse_sha256_checksum(text, expected_filename=archive_name)
                except (OSError, UnicodeError, ChecksumFormatError) as exc:
                    raise InvalidChecksumDocumentError(
                        f"Invalid upstream checksum {resource_name}: {exc}",
                        attempt_count=attempt,
                    ) from exc
                os.replace(temporary_path, destination)
                return checksum, attempt
            except (requests.RequestException, _RetriableTransferError) as exc:
                failure = exc
            finally:
                if response is not None:
                    response.close()

            if attempt == max_attempts:
                raise RetryExhaustedError(
                    f"Retry limit exhausted for {resource_name}: {failure}",
                    attempt_count=attempt,
                ) from failure
            self._backoff(attempt)

        raise AssertionError("Unreachable retry loop")

    def download_archive(
        self,
        url: str,
        partial_path: Path,
        *,
        resource_name: str,
    ) -> TransferResult:
        max_attempts = self.config.retries + 1
        failure: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            offset = (
                partial_path.stat().st_size
                if self.config.resume_partial and partial_path.is_file()
                else 0
            )
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            response: HttpResponse | None = None
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                )
                if offset and response.status_code == 416:
                    total = _unsatisfied_range_total(response.headers)
                    if total == offset:
                        return TransferResult(attempt, True, "REUSED_COMPLETE_PARTIAL")
                    if total is not None and offset > total:
                        with partial_path.open("wb"):
                            pass
                        raise _RetriableTransferError(
                            f"Partial file exceeded remote size for {resource_name}; "
                            "restarted safely"
                        )
                    raise ProtocolResponseError(
                        f"Unsafe HTTP 416 response for {resource_name}",
                        attempt_count=attempt,
                    )

                self._require_download_status(
                    response.status_code,
                    resource_name=resource_name,
                    attempt=attempt,
                )
                if offset and response.status_code == 206:
                    try:
                        range_start, range_end, total_size = _content_range(response.headers)
                    except ValueError as exc:
                        raise ProtocolResponseError(
                            f"Unsafe resume metadata for {resource_name}: {exc}",
                            attempt_count=attempt,
                        ) from exc
                    if range_start != offset:
                        raise ProtocolResponseError(
                            f"Resume for {resource_name} started at {range_start}, "
                            f"expected {offset}",
                            attempt_count=attempt,
                        )
                    expected_response_bytes = range_end - range_start + 1
                    content_length = self._content_length(
                        response.headers,
                        resource_name=resource_name,
                        attempt=attempt,
                    )
                    if content_length is not None and content_length != expected_response_bytes:
                        raise ProtocolResponseError(
                            f"Inconsistent resume length for {resource_name}",
                            attempt_count=attempt,
                        )
                    bytes_written = self._write_response(response, partial_path, mode="ab")
                    if bytes_written != expected_response_bytes:
                        raise _RetriableTransferError(
                            f"Short resumed response for {resource_name}: "
                            f"expected {expected_response_bytes} bytes, received {bytes_written}"
                        )
                    if partial_path.stat().st_size != total_size:
                        raise _RetriableTransferError(
                            f"Incomplete resumed archive {resource_name}"
                        )
                    return TransferResult(attempt, True, "RANGE_RESUMED")

                if response.status_code == 206:
                    raise ProtocolResponseError(
                        f"Unexpected partial response without a Range request for {resource_name}",
                        attempt_count=attempt,
                    )

                expected_length = self._content_length(
                    response.headers,
                    resource_name=resource_name,
                    attempt=attempt,
                )
                bytes_written = self._write_response(response, partial_path, mode="wb")
                if expected_length is not None and bytes_written != expected_length:
                    raise _RetriableTransferError(
                        f"Short response for {resource_name}: expected {expected_length} bytes, "
                        f"received {bytes_written}"
                    )
                detail = "RANGE_IGNORED_RESTARTED" if offset else "FULL_DOWNLOAD"
                return TransferResult(attempt, False, detail)
            except (requests.RequestException, _RetriableTransferError) as exc:
                failure = exc
            finally:
                if response is not None:
                    response.close()

            if attempt == max_attempts:
                raise RetryExhaustedError(
                    f"Retry limit exhausted for {resource_name}: {failure}",
                    attempt_count=attempt,
                ) from failure
            self._backoff(attempt)

        raise AssertionError("Unreachable retry loop")

    def _require_download_status(
        self,
        status_code: int,
        *,
        resource_name: str,
        attempt: int,
    ) -> None:
        if status_code in {200, 206}:
            return
        if status_code == 404:
            raise MissingRemoteError(
                f"Official Binance archive resource is missing: {resource_name}",
                attempt_count=attempt,
            )
        if status_code == 403:
            raise ForbiddenRemoteError(
                f"Official Binance archive resource returned HTTP 403: {resource_name}",
                attempt_count=attempt,
            )
        if status_code == 429 or 500 <= status_code <= 599:
            raise _RetriableTransferError(
                f"Retriable HTTP {status_code} for {resource_name}"
            )
        raise TransferError(
            f"Unexpected HTTP {status_code} for {resource_name}",
            attempt_count=attempt,
        )

    def _write_response(
        self,
        response: HttpResponse,
        path: Path,
        *,
        mode: Literal["wb", "ab"],
    ) -> int:
        bytes_written = 0
        with path.open(mode) as handle:
            for chunk in response.iter_content(self.config.chunk_size_bytes):
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        return bytes_written

    @staticmethod
    def _content_length(
        headers: Mapping[str, str],
        *,
        resource_name: str,
        attempt: int,
    ) -> int | None:
        try:
            return _header_integer(headers, "Content-Length")
        except ValueError as exc:
            raise ProtocolResponseError(
                f"Unsafe Content-Length metadata for {resource_name}: {exc}",
                attempt_count=attempt,
            ) from exc

    def _backoff(self, completed_attempt: int) -> None:
        delay = min(
            self.config.backoff_initial_seconds * (2 ** (completed_attempt - 1)),
            self.config.backoff_max_seconds,
        )
        self.sleep(delay)

    @property
    def timeout(self) -> tuple[float, float]:
        return (
            self.config.connect_timeout_seconds,
            self.config.read_timeout_seconds,
        )
