from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.targets.trend_label_audit import (
    ATR_PERIOD,
    CONFIRMATION_LENGTH,
    DOWN,
    FROZEN_HORIZON_HOURS,
    FROZEN_TIMEFRAME,
    LABEL_VALUES,
    NEUTRAL,
    NEUTRAL_QUANTILE,
    UP,
    _atr_on_common_anchors,
    _calibration_positions,
    adaptive_three_class_labels,
    audit_fold,
    calibrate_adaptive_k,
    calibrate_fixed_tau,
    calibrate_volatility_regimes,
    causal_wilder_atr14,
    class_structure,
    feasibility_decision,
    fixed_three_class_labels,
    hysteresis_three,
)
from btc_forecasting.training.lstm_horizon_walkforward import PairedHorizonSamples


def _ohlc(times: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.arange(len(times), dtype=np.float64) + 100.0
    return pd.DataFrame(
        {"open_time": times, "high": close + 2.0, "low": close - 1.0, "close": close}
    )


def test_one_hour_labels_and_calibration_quantiles_are_frozen() -> None:
    calibration_y = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    atr = np.ones(7)
    tau = calibrate_fixed_tau(calibration_y)
    k = calibrate_adaptive_k(calibration_y, atr)
    fixed = fixed_three_class_labels(np.array([-4.0, 0.0, 4.0]), tau=tau)
    adaptive, _ = adaptive_three_class_labels(np.array([-4.0, 0.0, 4.0]), np.ones(3), k=k)

    assert (FROZEN_TIMEFRAME, FROZEN_HORIZON_HOURS) == ("1h", 1)
    assert NEUTRAL_QUANTILE == 1.0 / 3.0
    assert set(fixed) == set(adaptive) == set(LABEL_VALUES)
    assert tuple(fixed) == tuple(adaptive) == (DOWN, NEUTRAL, UP)
    assert tau == pytest.approx(np.quantile(np.abs(calibration_y), 1.0 / 3.0))
    assert k == pytest.approx(np.quantile(np.abs(calibration_y) / atr, 1.0 / 3.0))


def test_atr14_is_causal_and_resets_after_a_gap_without_fill() -> None:
    times = pd.date_range("2020-01-01", periods=30, freq="h", tz="UTC")
    frame = _ohlc(times)
    original = causal_wilder_atr14(frame)
    changed = frame.copy()
    changed.loc[29, "high"] = 1_000.0
    changed_result = causal_wilder_atr14(changed)
    assert ATR_PERIOD == 14
    np.testing.assert_allclose(original.loc[:28, "atr_14"], changed_result.loc[:28, "atr_14"], equal_nan=True)

    first = pd.date_range("2020-01-01", periods=14, freq="h", tz="UTC")
    second = pd.date_range("2020-01-01T15:00:00Z", periods=14, freq="h", tz="UTC")
    gapped = causal_wilder_atr14(_ohlc(first.append(second)))
    assert np.isnan(gapped.loc[14:26, "atr_14"]).all()
    assert np.isfinite(gapped.loc[27, "atr_14"])
    assert gapped.loc[14, "true_range"] == pytest.approx(3.0)


def test_adaptive_labels_and_hysteresis_do_not_use_future_values() -> None:
    returns = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
    atr = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    labels, theta = adaptive_three_class_labels(returns, atr, k=2.0)
    extended, extended_theta = adaptive_three_class_labels(
        np.append(returns, 1_000.0), np.append(atr, 0.001), k=2.0
    )
    np.testing.assert_array_equal(labels, extended[:-1])
    np.testing.assert_allclose(theta, extended_theta[:-1])

    candidates = np.array([UP, UP, UP, DOWN, DOWN, UP, DOWN, DOWN, DOWN])
    times = pd.date_range("2020-01-01", periods=9, freq="h", tz="UTC")
    result = hysteresis_three(candidates, times)
    altered = candidates.copy()
    altered[6:] = UP
    altered_result = hysteresis_three(altered, times)
    assert CONFIRMATION_LENGTH == 3
    assert tuple(result.labels[:6]) == (NEUTRAL, NEUTRAL, UP, UP, UP, UP)
    np.testing.assert_array_equal(result.labels[:6], altered_result.labels[:6])


def test_hysteresis_state_and_pending_confirmation_reset_at_gap() -> None:
    times = pd.to_datetime(
        [
            "2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z", "2020-01-01T02:00:00Z",
            "2020-01-01T05:00:00Z", "2020-01-01T06:00:00Z", "2020-01-01T07:00:00Z",
        ],
        utc=True,
    )
    result = hysteresis_three(np.full(6, UP), pd.DatetimeIndex(times))
    assert tuple(result.labels) == (NEUTRAL, NEUTRAL, UP, NEUTRAL, NEUTRAL, UP)
    assert result.confirmed_switches == 2


def _samples() -> tuple[PairedHorizonSamples, np.ndarray, np.ndarray]:
    count = 42
    times = pd.date_range("2020-01-01", periods=count, freq="h", tz="UTC")
    returns = np.sin(np.arange(count) * 0.9) * 0.02
    features = np.zeros((count, 24, len(F0_FEATURE_NAMES)), dtype=np.float32)
    samples = PairedHorizonSamples(
        features=features,
        decision_times=times,
        sigma=np.full(count, 0.01),
        raw_targets={1: returns},
        normalized_targets={1: returns / 0.01},
    )
    return samples, np.linspace(0.005, 0.025, count), np.tile([0.1, 0.5, 0.9], 14)


def test_outer_evaluation_cannot_change_train_calibration() -> None:
    samples, atr, volatility = _samples()
    calibration, evaluation = np.arange(24), np.arange(24, 42)
    first = audit_fold(
        fold_number=1, samples=samples, calibration_positions=calibration,
        evaluation_positions=evaluation, atr_pct_14=atr, endpoint_volatility=volatility,
    )
    changed = samples.raw_targets[1].copy()
    changed[evaluation] *= 10_000.0
    changed_samples = PairedHorizonSamples(
        samples.features, samples.decision_times, samples.sigma,
        {1: changed}, {1: changed / samples.sigma},
    )
    second = audit_fold(
        fold_number=1, samples=changed_samples, calibration_positions=calibration,
        evaluation_positions=evaluation, atr_pct_14=atr, endpoint_volatility=volatility,
    )
    assert first["calibration_parameters"] == second["calibration_parameters"]
    assert calibrate_volatility_regimes(volatility[calibration]) == (
        first["calibration_parameters"]["volatility_regime_33rd_percentile"],  # type: ignore[index]
        first["calibration_parameters"]["volatility_regime_67th_percentile"],  # type: ignore[index]
    )


def test_one_hour_boundary_purge_prevents_calibration_target_crossing() -> None:
    times = pd.date_range("2020-01-01", periods=12, freq="h", tz="UTC")
    safe, purged = _calibration_positions([np.arange(8)], np.arange(8, 12), times)
    assert np.array_equal(safe, np.arange(7))
    assert purged == 1
    assert times[safe[-1]] + np.timedelta64(1, "h") < times[8]


def test_atr_read_is_bounded_to_common_train_anchors(monkeypatch) -> None:
    times = pd.date_range("2020-01-01", periods=20, freq="h", tz="UTC")
    captured: dict[str, object] = {}

    class FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return _ohlc(times)

    def fake_read_table(path: Path, *, columns: list[str], filters: list[tuple]):
        captured.update(path=path, columns=columns, filters=filters)
        return FakeTable()

    monkeypatch.setattr("btc_forecasting.targets.trend_label_audit.pq.read_table", fake_read_table)
    decisions = times[13:20] + np.timedelta64(1, "h")
    values = _atr_on_common_anchors(project_root=Path("project"), decision_times=decisions)
    assert np.isfinite(values).all()
    assert captured["filters"] == [("open_time", "<=", times[19].to_pydatetime())]
    assert "validation" not in str(captured).lower()
    assert "test" not in str(captured).lower()


def test_feasibility_rule_is_exact_and_deterministic() -> None:
    balanced = class_structure(np.tile([DOWN, NEUTRAL, UP], 10))
    feasible = [balanced] * 4
    assert feasibility_decision(feasible)["result"] == "FEASIBLE"

    low_down = class_structure(np.array([DOWN] * 9 + [NEUTRAL] * 45 + [UP] * 46))
    assert feasibility_decision([low_down, low_down, balanced, balanced])["result"] == "PATHOLOGICAL"
    dominant_up = class_structure(np.array([DOWN] * 9 + [NEUTRAL] * 9 + [UP] * 82))
    inputs = [dominant_up, dominant_up, balanced, balanced]
    assert feasibility_decision(inputs) == feasibility_decision(inputs)
    assert feasibility_decision(inputs)["result"] == "PATHOLOGICAL"
