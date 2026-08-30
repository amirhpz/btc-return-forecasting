from __future__ import annotations

import numpy as np


def zero_return_prediction(sample_count: int) -> np.ndarray:
    """Predict zero future return."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    return np.zeros(sample_count, dtype=float)


def previous_return_prediction(observed_returns: np.ndarray) -> np.ndarray:
    """Use the last observed return supplied for each sample as the prediction."""
    return np.asarray(observed_returns, dtype=float).copy()
