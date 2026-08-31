from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples, load_sequence_development_samples
from btc_forecasting.training.lstm_diagnostics import (
    DiagnosticRunResult,
    _write_diagnostic,
    load_source_run,
)
from btc_forecasting.training.lstm_generalization import (
    VALIDATION_BLOCK_COUNT,
    _correlation,
)

VOLATILITY_FEATURE_NAME = "rolling_volatility_24h"
VOLATILITY_FEATURE_INDEX = F0_FEATURE_NAMES.index(VOLATILITY_FEATURE_NAME)


@dataclass(frozen=True)
class NormalizationSelection:
    targets: np.ndarray
    sigma: np.ndarray
    valid_mask: np.ndarray
    decision_times: pd.DatetimeIndex

    @property
    def exclusion_count(self) -> int:
        return int(np.count_nonzero(~self.valid_mask))

    @property
    def eligible_targets(self) -> np.ndarray:
        return self.targets[self.valid_mask]

    @property
    def eligible_sigma(self) -> np.ndarray:
        return self.sigma[self.valid_mask]

    @property
    def normalized_targets(self) -> np.ndarray:
        return self.eligible_targets / self.eligible_sigma


def select_normalization_samples(samples: SequenceSamples) -> NormalizationSelection:
    if samples.features.ndim != 3 or samples.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("E03 samples do not match the frozen F0 sequence shape")
    targets = np.asarray(samples.targets, dtype=np.float64)
    sigma = np.asarray(
        samples.features[:, -1, VOLATILITY_FEATURE_INDEX],
        dtype=np.float64,
    )
    if not np.isfinite(targets).all():
        raise ValueError("Official E03 targets must be finite")
    finite_sigma = np.isfinite(sigma)
    if np.any(sigma[finite_sigma] < 0.0):
        raise ValueError("rolling_volatility_24h cannot be negative")
    valid_mask = finite_sigma & (sigma != 0.0)
    return NormalizationSelection(
        targets=targets,
        sigma=sigma,
        valid_mask=valid_mask,
        decision_times=samples.decision_times,
    )


def _distribution(values: np.ndarray, *, absolute_key: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Distribution requires non-empty finite values")
    quantiles = np.quantile(array, [0.05, 0.25, 0.75, 0.95])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        absolute_key: float(np.mean(np.abs(array))),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "q95": float(quantiles[3]),
    }


def _validation_over_train_ratio(
    train: dict[str, float | int],
    validation: dict[str, float | int],
    *,
    absolute_key: str,
) -> dict[str, float | None]:
    train_std = float(train["std"])
    train_absolute = float(train[absolute_key])
    return {
        "std": float(validation["std"]) / train_std if train_std != 0.0 else None,
        absolute_key: (
            float(validation[absolute_key]) / train_absolute
            if train_absolute != 0.0
            else None
        ),
    }


def volatility_target_relationship(
    selection: NormalizationSelection,
) -> dict[str, float | int | None]:
    sigma = selection.eligible_sigma
    magnitude = np.abs(selection.eligible_targets)
    return {
        "n": int(len(sigma)),
        "pearson_sigma_vs_abs_target": _correlation(sigma, magnitude, rank=False),
        "spearman_sigma_vs_abs_target": _correlation(sigma, magnitude, rank=True),
    }


def build_distribution_comparison(
    train: NormalizationSelection,
    validation: NormalizationSelection,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_train = _distribution(
        train.eligible_targets,
        absolute_key="mean_absolute_target",
    )
    raw_validation = _distribution(
        validation.eligible_targets,
        absolute_key="mean_absolute_target",
    )
    normalized_train = _distribution(
        train.normalized_targets,
        absolute_key="mean_absolute_value",
    )
    normalized_validation = _distribution(
        validation.normalized_targets,
        absolute_key="mean_absolute_value",
    )
    raw_report: dict[str, object] = {
        "train": raw_train,
        "validation": raw_validation,
        "validation_over_train": _validation_over_train_ratio(
            raw_train,
            raw_validation,
            absolute_key="mean_absolute_target",
        ),
    }
    normalized_report: dict[str, object] = {
        "train": normalized_train,
        "validation": normalized_validation,
        "validation_over_train": _validation_over_train_ratio(
            normalized_train,
            normalized_validation,
            absolute_key="mean_absolute_value",
        ),
    }
    return raw_report, normalized_report


def validation_normalization_blocks(
    validation: NormalizationSelection,
) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    positions = np.arange(len(validation.targets))
    for number, block_positions in enumerate(
        np.array_split(positions, VALIDATION_BLOCK_COUNT),
        1,
    ):
        if len(block_positions) == 0:
            raise ValueError("Validation is too small for four non-empty blocks")
        block_mask = validation.valid_mask[block_positions]
        eligible_positions = block_positions[block_mask]
        if len(eligible_positions) == 0:
            raise ValueError(f"Validation block {number} has no normalizable samples")
        targets = validation.targets[eligible_positions]
        sigma = validation.sigma[eligible_positions]
        normalized = targets / sigma
        blocks.append(
            {
                "block": number,
                "start_decision_time": validation.decision_times[
                    block_positions[0]
                ].isoformat(),
                "end_decision_time": validation.decision_times[
                    block_positions[-1]
                ].isoformat(),
                "original_e03_n": int(len(block_positions)),
                "excluded_sigma_count": int(len(block_positions) - len(eligible_positions)),
                "n": int(len(eligible_positions)),
                "raw_target_std": float(np.std(targets, ddof=0)),
                "sigma_mean": float(np.mean(sigma)),
                "normalized_target_std": float(np.std(normalized, ddof=0)),
                "normalized_mean_absolute_value": float(np.mean(np.abs(normalized))),
            }
        )
    return blocks


def build_volatility_normalization_report(
    *,
    train: SequenceSamples,
    validation: SequenceSamples,
    source_run_id: str,
) -> dict[str, object]:
    train_selection = select_normalization_samples(train)
    validation_selection = select_normalization_samples(validation)
    raw_distribution, normalized_distribution = build_distribution_comparison(
        train_selection,
        validation_selection,
    )
    return {
        "diagnostic_id": "E03-VN-D",
        "diagnostic_type": "causal_volatility_normalization",
        "source_run_id": source_run_id,
        "definition": {
            "sigma_t": VOLATILITY_FEATURE_NAME,
            "z_t": "target_t / sigma_t",
            "epsilon_added": False,
            "feature_alignment": "sequence_endpoint_available_at_decision_time",
        },
        "evaluated_splits": ["train", "validation"],
        "normalization_exclusions": {
            "train": train_selection.exclusion_count,
            "validation": validation_selection.exclusion_count,
        },
        "volatility_target_relationship": {
            "train": volatility_target_relationship(train_selection),
            "validation": volatility_target_relationship(validation_selection),
        },
        "raw_target_distribution_on_normalizable_samples": raw_distribution,
        "volatility_normalized_target_distribution": normalized_distribution,
        "validation_temporal_blocks": validation_normalization_blocks(
            validation_selection
        ),
        "test_set": "NOT EVALUATED",
    }


def run_volatility_normalization_diagnostic(
    *,
    project_root: Path,
    source_run: Path,
) -> DiagnosticRunResult:
    root = project_root.resolve()
    source = load_source_run(project_root=root, source_run=source_run)
    prepared = load_sequence_development_samples(
        canonical_path=source.canonical_path,
        target_path=source.target_path,
        split_metadata_path=source.split_path,
    )
    result = build_volatility_normalization_report(
        train=prepared.train,
        validation=prepared.validation,
        source_run_id=str(source.manifest["run_id"]),
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _write_diagnostic(
        root=root,
        directory_name=f"E03VN_volatility_normalization_{timestamp}",
        result=result,
    )
