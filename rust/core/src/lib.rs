//! Lag feature matrix (row-major flatten: n_samples * (n_lags + 1)).

pub fn build_lag_matrix(values: &[f64], n_lags: usize) -> Vec<f64> {
    let n = values.len();
    if n_lags == 0 || n <= n_lags {
        return Vec::new();
    }
    let rows = n - n_lags;
    let cols = n_lags + 1;
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        let t = i + n_lags;
        out[i * cols] = values[t];
        for lag in 1..=n_lags {
            out[i * cols + lag] = values[t - lag];
        }
    }
    out
}
