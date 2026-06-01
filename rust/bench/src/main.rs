use arima_models_and_time_series_forecasting_for_business_analytics_core::build_lag_matrix;

fn main() {
    let v: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin()).collect();
    for _ in 0..500 {
        let _ = build_lag_matrix(&v, 12);
    }
}
