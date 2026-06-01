#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import build_lag_matrix  # noqa: E402

def main() -> None:
    v = np.ascontiguousarray(np.sin(np.arange(5000) * 0.01)); nl = 12
    t0 = time.perf_counter()
    for _ in range(200):
        build_lag_matrix(v, nl)
    py_s = time.perf_counter() - t0
    try:
        import arima_models_and_time_series_forecasting_for_business_analytics_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(v, nl, 500)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(build_lag_matrix(v, nl), np.asarray(rs.build_lag_matrix_py(v, nl)), rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
