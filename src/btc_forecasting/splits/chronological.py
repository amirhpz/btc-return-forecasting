from __future__ import annotations

import math

import pandas as pd


def assign_splits_by_target_timestamp(
    target_timestamps: pd.Series,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.Series:
    """Assign chronological splits using valid target timestamps.

    Splitting on the target timestamp prevents a training target from crossing into a
    later split. Missing target timestamps remain unassigned.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a positive test fraction")

    parsed = pd.to_datetime(target_timestamps, utc=True, errors="coerce")
    valid = parsed.dropna().sort_values()
    if len(valid) < 3:
        raise ValueError("At least three valid targets are required")

    train_count = max(1, math.floor(len(valid) * train_fraction))
    validation_count = max(1, math.floor(len(valid) * validation_fraction))
    if train_count + validation_count >= len(valid):
        validation_count = len(valid) - train_count - 1

    train_end = valid.iloc[train_count - 1]
    validation_end = valid.iloc[train_count + validation_count - 1]

    labels = pd.Series(pd.NA, index=parsed.index, dtype="string", name="split")
    labels.loc[parsed <= train_end] = "train"
    labels.loc[(parsed > train_end) & (parsed <= validation_end)] = "validation"
    labels.loc[parsed > validation_end] = "test"
    labels.loc[parsed.isna()] = pd.NA
    return labels
