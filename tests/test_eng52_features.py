from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecasting.features.eng52 import (
    ENG52_FEATURE_NAMES,
    _FIRST_VALID_SEGMENT_ROW,
    compute_eng52_features,
)
from btc_forecasting.features.eng52_build import build_eng52_dataset

REFERENCE_SOURCE = Path("references/features/source")
AUDIT_PATH = Path("configs/features/eng52_audit.yaml")


def _hourly_frame(periods: int = 240) -> pd.DataFrame:
    position = np.arange(periods, dtype=float)
    center = 20_000.0 + 7.0 * position + 180.0 * np.sin(position / 7.0)
    open_price = center * (1.0 + 0.001 * np.sin(position / 3.0))
    close = center * (1.0 + 0.0012 * np.cos(position / 5.0))
    high = np.maximum(open_price, close) * (1.003 + 0.0002 * np.sin(position))
    low = np.minimum(open_price, close) * (0.997 - 0.0002 * np.cos(position))
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2020-01-01", periods=periods, freq="h", tz="UTC"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + position % 31 + 10.0 * np.sin(position / 11.0) ** 2,
        }
    )


def _reference_compute(name: str, frame: pd.DataFrame) -> pd.Series:
    source_path = REFERENCE_SOURCE / f"{name}.py"
    sys.path.insert(0, str(REFERENCE_SOURCE.resolve()))
    try:
        spec = importlib.util.spec_from_file_location(f"eng52_reference_{name}", source_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference_input = frame.copy()
        reference_input["datetime"] = reference_input["open_time"]
        return module.compute(reference_input).astype(float)
    finally:
        sys.path.pop(0)


def test_continuous_data_matches_all_reference_implementations() -> None:
    source = _hourly_frame()
    actual = compute_eng52_features(source)

    assert tuple(actual.columns) == ("open_time", *ENG52_FEATURE_NAMES)
    assert len(ENG52_FEATURE_NAMES) == len(set(ENG52_FEATURE_NAMES)) == 52
    for name in ENG52_FEATURE_NAMES:
        expected = _reference_compute(name, source)
        first_valid = _FIRST_VALID_SEGMENT_ROW[name]
        np.testing.assert_allclose(
            actual[name].iloc[first_valid:].to_numpy(),
            expected.iloc[first_valid:].to_numpy(),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
            err_msg=name,
        )


def test_gap_resets_all_feature_state_and_required_history() -> None:
    continuous = _hourly_frame(360)
    gapped = continuous.drop(index=180).reset_index(drop=True)
    actual = compute_eng52_features(gapped)
    second_segment = gapped.iloc[180:].reset_index(drop=True)
    standalone = compute_eng52_features(second_segment)

    np.testing.assert_allclose(
        actual.loc[180:, ENG52_FEATURE_NAMES].to_numpy(),
        standalone.loc[:, ENG52_FEATURE_NAMES].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    assert actual["absret_ema_ratio_20_100"].iloc[180:281].isna().all()
    assert np.isfinite(actual["absret_ema_ratio_20_100"].iloc[281])
    assert actual["open_gap_atr_14"].iloc[180:195].isna().all()
    assert np.isfinite(actual["open_gap_atr_14"].iloc[195])


def test_future_rows_never_change_existing_feature_rows() -> None:
    source = _hourly_frame(260)
    prefix = source.iloc[:210].copy()
    future_changed = source.copy()
    future_changed.loc[210:, ["open", "high", "low", "close", "volume"]] *= 1.7

    expected = compute_eng52_features(prefix)
    actual = compute_eng52_features(future_changed).iloc[:210]
    np.testing.assert_allclose(
        actual.loc[:, ENG52_FEATURE_NAMES].to_numpy(),
        expected.loc[:, ENG52_FEATURE_NAMES].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_synthetic_build_writes_only_expected_schema_and_summary(tmp_path: Path) -> None:
    source_path = tmp_path / "canonical.parquet"
    output_path = tmp_path / "eng52.parquet"
    summary_path = tmp_path / "summary.json"
    _hourly_frame(180).to_parquet(source_path, index=False)

    result = build_eng52_dataset(
        source_path=source_path,
        output_path=output_path,
        summary_path=summary_path,
        audit_path=AUDIT_PATH,
    )
    built = pd.read_parquet(output_path)

    assert tuple(built.columns) == ("open_time", *ENG52_FEATURE_NAMES)
    assert set(result.summary) == {
        "input_row_count",
        "output_row_count",
        "first_open_time",
        "last_open_time",
        "feature_names",
        "per_feature",
    }
    assert result.summary["input_row_count"] == 180
    assert result.summary["output_row_count"] == 180
    assert result.summary["feature_names"] == list(ENG52_FEATURE_NAMES)
    assert set(result.summary["per_feature"]) == set(ENG52_FEATURE_NAMES)  # type: ignore[arg-type]
    assert summary_path.is_file()
