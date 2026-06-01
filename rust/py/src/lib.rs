use arima_models_and_time_series_forecasting_for_business_analytics_core::build_lag_matrix;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn build_lag_matrix_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    n_lags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(build_lag_matrix(values.as_slice()?, n_lags).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (values, n_lags, iterations=500))]
fn bench_kernel_py(values: PyReadonlyArray1<f64>, n_lags: usize, iterations: usize) -> PyResult<f64> {
    let buf = values.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = build_lag_matrix(&buf, n_lags);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn arima_models_and_time_series_forecasting_for_business_analytics_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_lag_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
