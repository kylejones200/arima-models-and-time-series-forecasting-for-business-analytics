"""Lag feature matrix for univariate series (row-major flatten)."""

from __future__ import annotations

import numpy as np


def build_lag_matrix(values: np.ndarray, n_lags: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n_lags <= 0 or n <= n_lags:
        return np.empty(0, dtype=float)
    rows = n - n_lags
    cols = n_lags + 1
    out = np.empty(rows * cols, dtype=float)
    for i in range(rows):
        t = i + n_lags
        out[i * cols] = v[t]
        for lag in range(1, n_lags + 1):
            out[i * cols + lag] = v[t - lag]
    return out
