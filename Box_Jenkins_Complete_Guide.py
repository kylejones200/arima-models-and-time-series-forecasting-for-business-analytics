"""Box-Jenkins methodology: ARIMA model selection, diagnostics, forecasting, and VAR."""

from __future__ import annotations

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.datasets import macrodata
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

plt.style.use("seaborn-v0_8-darkgrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Box-Jenkins ARIMA workflow with optional multivariate VAR analysis.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Use non-interactive backend and close figures instead of plt.show().",
    )
    parser.add_argument(
        "--skip-auto-arima",
        action="store_true",
        help="Skip pmdarima auto_arima (slow step).",
    )
    parser.add_argument(
        "--skip-var",
        action="store_true",
        help="Skip multivariate VAR section.",
    )
    parser.add_argument(
        "--forecast-periods",
        type=int,
        default=20,
        help="Number of periods to forecast (default: 20).",
    )
    return parser.parse_args()


def finish_figure(show: bool) -> None:
    if show:
        plt.show()
    else:
        plt.close()


def load_real_gdp() -> pd.Series:
    """Load quarterly U.S. real GDP from statsmodels macrodata."""
    df = macrodata.load_pandas().data
    df.index = pd.period_range("1959Q1", periods=len(df), freq="Q").to_timestamp()
    data = df["realgdp"]
    print(f"Loaded data: {len(data)} observations")
    print(f"  Period: {data.index[0]} to {data.index[-1]}")
    print("\nFirst few values:")
    print(data.head())
    return data


def plot_series(
    x_index,
    y_values,
    *,
    title: str,
    ylabel: str = "Value",
    show: bool,
) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(x_index, y_values, linewidth=2)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finish_figure(show)


def test_stationarity(timeseries: pd.Series, title: str = "Time Series") -> bool:
    """Augmented Dickey-Fuller test for stationarity."""
    print(f"\n{'=' * 70}")
    print(f"STATIONARITY TEST: {title}")
    print(f"{'=' * 70}")

    result = adfuller(timeseries.dropna(), autolag="AIC")

    print("\nAugmented Dickey-Fuller Test:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print("  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")

    if result[1] <= 0.05:
        print("\nSTATIONARY (p-value <= 0.05)")
        print("  -> Can proceed with ARIMA modeling")
        return True

    print("\nNON-STATIONARY (p-value > 0.05)")
    print("  -> Differencing required")
    return False


def find_differencing_order(
    timeseries: pd.Series, max_d: int = 3
) -> tuple[int, pd.Series]:
    """Find minimum differencing order d for stationarity."""
    print(f"\n{'=' * 70}")
    print("DETERMINING DIFFERENCING ORDER (d)")
    print(f"{'=' * 70}\n")

    current_series = timeseries.copy()

    for d in range(max_d + 1):
        result = adfuller(current_series.dropna(), autolag="AIC")
        p_value = result[1]

        print(f"d={d}: p-value = {p_value:.6f}", end="")

        if p_value <= 0.05:
            print(" -> STATIONARY")
            print(f"\nRecommended d = {d}")
            return d, current_series

        print(" -> Non-stationary, differencing...")
        current_series = current_series.diff()

    print(f"\nWarning: Still non-stationary after {max_d} differences")
    return max_d, current_series


def plot_original_vs_differenced(
    original: pd.Series,
    differenced: pd.Series,
    d_order: int,
    *,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].plot(original.index, original.values, linewidth=2, color="blue")
    axes[0].set_title("Original Series", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(differenced.index, differenced.values, linewidth=2, color="red")
    axes[1].set_title(f"Differenced Series (d={d_order})", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Differenced Value")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    finish_figure(show)


def plot_acf_pacf(timeseries: pd.Series, *, lags: int = 40, show: bool) -> None:
    """Plot ACF and PACF to help identify AR and MA orders."""
    print(f"\n{'=' * 70}")
    print("ACF/PACF ANALYSIS - Identifying p and q")
    print(f"{'=' * 70}\n")
    print("Guidelines:")
    print("  - ACF cuts off at lag q -> MA(q)")
    print("  - PACF cuts off at lag p -> AR(p)")
    print("  - Both tail off gradually -> ARMA(p,q)\n")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    plot_acf(timeseries.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=14, fontweight="bold")

    plot_pacf(timeseries.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title(
        "Partial Autocorrelation Function (PACF)", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    finish_figure(show)


def manual_arima_selection(
    timeseries: pd.Series,
    p_range: range,
    d: int,
    q_range: range,
) -> tuple[object | None, tuple[int, int, int] | None]:
    """Grid search over ARIMA(p,d,q) using statsmodels."""
    print(f"\n{'=' * 70}")
    print("MANUAL ARIMA MODEL SELECTION")
    print(f"{'=' * 70}\n")

    best_aic = np.inf
    best_order = None
    best_model = None
    results: list[dict[str, float | int]] = []

    for p in p_range:
        for q in q_range:
            try:
                fitted = ARIMA(timeseries, order=(p, d, q)).fit()
                results.append(
                    {"p": p, "d": d, "q": q, "AIC": fitted.aic, "BIC": fitted.bic}
                )

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted

                print(f"ARIMA({p},{d},{q}): AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")
            except Exception as exc:
                print(f"ARIMA({p},{d},{q}): Failed - {str(exc)[:50]}")

    if best_order is None:
        print("\nNo ARIMA model converged in the search grid.")
        return None, None

    print(f"\nBest model: ARIMA{best_order} with AIC={best_aic:.2f}")

    results_df = pd.DataFrame(results).sort_values("AIC").head(5)
    print("\nTop 5 models:")
    print(results_df.to_string(index=False))

    return best_model, best_order


def automatic_arima_selection(timeseries: pd.Series):
    """Select ARIMA order with pmdarima auto_arima."""
    print(f"\n{'=' * 70}")
    print("AUTOMATIC ARIMA MODEL SELECTION (auto_arima)")
    print(f"{'=' * 70}\n")
    print("Searching for optimal parameters...")
    print("(This may take a minute)\n")

    model = auto_arima(
        timeseries,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action="ignore",
    )

    print(f"\nBest model: ARIMA{model.order}")
    print(f"  AIC: {model.aic():.2f}")
    print(f"  BIC: {model.bic():.2f}")
    return model


def diagnostic_plots(model, *, title: str = "Model", show: bool) -> None:
    """Residual diagnostics for a fitted ARIMA model (statsmodels or pmdarima)."""
    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC CHECKING: {title}")
    print(f"{'=' * 70}\n")

    residuals = pd.Series(model.resid).dropna()

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(residuals.values)
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_title("Residuals Over Time", fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Residual")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax2.set_title("Histogram of Residuals", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    sm.qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    plot_acf(residuals, lags=40, ax=ax4)
    ax4.set_title("ACF of Residuals", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    finish_figure(show)

    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\nLjung-Box test for residual autocorrelation:")
    print("(p-value > 0.05 indicates no significant autocorrelation)\n")
    print(lb_test)

    print("\nResidual statistics:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Skewness: {residuals.skew():.4f}")
    print(f"  Kurtosis: {residuals.kurtosis():.4f}")


def forecast_arima(
    model,
    original_data: pd.Series,
    *,
    n_periods: int = 20,
    alpha: float = 0.05,
    show: bool,
) -> pd.DataFrame:
    """Forecast with confidence intervals (pmdarima API)."""
    print(f"\n{'=' * 70}")
    print(f"FORECASTING {n_periods} PERIODS AHEAD")
    print(f"{'=' * 70}\n")

    forecast, conf_int = model.predict(
        n_periods=n_periods,
        return_conf_int=True,
        alpha=alpha,
    )

    last_date = original_data.index[-1]
    freq = original_data.index.freq or pd.infer_freq(original_data.index)
    forecast_index = pd.date_range(
        start=last_date, periods=n_periods + 1, freq=freq
    )[1:]

    forecast_df = pd.DataFrame(
        {
            "Forecast": forecast,
            "Lower_CI": conf_int[:, 0],
            "Upper_CI": conf_int[:, 1],
        },
        index=forecast_index,
    )

    print(forecast_df.head(10))

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        original_data.index[-100:],
        original_data.values[-100:],
        label="Historical",
        linewidth=2,
        color="blue",
    )
    ax.plot(
        forecast_index,
        forecast,
        label="Forecast",
        linewidth=2,
        color="red",
        linestyle="--",
    )
    ax.fill_between(
        forecast_index,
        conf_int[:, 0],
        conf_int[:, 1],
        alpha=0.3,
        color="red",
        label=f"{(1 - alpha) * 100:.0f}% CI",
    )
    ax.set_title(f"ARIMA{model.order} Forecast", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    finish_figure(show)

    return forecast_df


def run_arima_pipeline(args: argparse.Namespace) -> None:
    """Univariate Box-Jenkins workflow on U.S. real GDP."""
    print("Box-Jenkins libraries loaded")

    data = load_real_gdp()
    plot_series(
        data.index,
        data.values,
        title="Original Time Series",
        show=not args.no_show,
    )

    test_stationarity(data, "Original Series")

    d_order, differenced_data = find_differencing_order(data)
    plot_original_vs_differenced(data, differenced_data, d_order, show=not args.no_show)
    plot_acf_pacf(differenced_data, show=not args.no_show)

    manual_arima_selection(
        data,
        p_range=range(4),
        d=d_order,
        q_range=range(4),
    )

    if args.skip_auto_arima:
        print("\nSkipping auto_arima (--skip-auto-arima).")
        return

    auto_model = automatic_arima_selection(data)
    diagnostic_plots(
        auto_model,
        title=f"ARIMA{auto_model.order}",
        show=not args.no_show,
    )

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70 + "\n")
    print(auto_model.summary())

    forecast_arima(
        auto_model,
        data,
        n_periods=args.forecast_periods,
        show=not args.no_show,
    )


def load_var_data() -> pd.DataFrame:
    """Load or synthesize bivariate data for VAR (Industrial Production, Retail Sales)."""
    start = datetime(2015, 1, 1)
    end = datetime.today()

    try:
        from pandas_datareader.data import DataReader

        indpro = DataReader("INDPRO", "fred", start, end)
        rsafs = DataReader("RSAFS", "fred", start, end)
        data_var = pd.concat([indpro, rsafs], axis=1)
        data_var.columns = ["Industrial_Production", "Retail_Sales"]
        data_var = data_var.dropna()
        print(f"Loaded multivariate data: {len(data_var)} observations")
        print(f"  Variables: {list(data_var.columns)}")
        return data_var
    except Exception as exc:
        print(f"Could not fetch FRED data: {exc}")
        print("Creating synthetic multivariate data instead...")

    np.random.seed(42)
    time_index = pd.date_range(start="2015-01", periods=100, freq="ME")
    indpro = 50 + np.cumsum(np.random.normal(0, 2, 100))
    rsafs = 30 + 0.5 * indpro + np.random.normal(0, 2, 100)

    data_var = pd.DataFrame(
        {"Industrial_Production": indpro, "Retail_Sales": rsafs},
        index=time_index,
    )
    print(f"Created synthetic data: {len(data_var)} observations")
    return data_var


def run_var_pipeline(args: argparse.Namespace) -> None:
    """Multivariate VAR: stationarity, Granger causality, fit, diagnostics, forecast."""
    print("VAR libraries loaded")

    data_var = load_var_data()

    fig, ax = plt.subplots(figsize=(15, 6))
    data_var.plot(ax=ax, linewidth=2)
    ax.set_title("Multivariate Time Series", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    finish_figure(not args.no_show)

    print("\n" + "=" * 70)
    print("STATIONARITY TEST FOR MULTIVARIATE SERIES")
    print("=" * 70 + "\n")

    for col in data_var.columns:
        result = adfuller(data_var[col])
        status = "Non-stationary" if result[1] > 0.05 else "Stationary"
        print(f"{col}: p-value = {result[1]:.4f} -> {status}")

    data_var_diff = data_var.diff().dropna()

    print("\nAfter differencing:")
    for col in data_var_diff.columns:
        result = adfuller(data_var_diff[col])
        print(f"{col}: p-value = {result[1]:.4f}")

    fig, ax = plt.subplots(figsize=(15, 6))
    data_var_diff.plot(ax=ax, linewidth=2)
    ax.set_title("Differenced Multivariate Series", fontsize=16, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    finish_figure(not args.no_show)

    col1, col2 = data_var_diff.columns[0], data_var_diff.columns[1]
    print("\n" + "=" * 70)
    print("GRANGER CAUSALITY TEST")
    print("=" * 70)
    print(f"\nTesting if {col2} Granger-causes {col1}:\n")

    gc_result = grangercausalitytests(
        data_var_diff[[col1, col2]], maxlag=5, verbose=False
    )

    print("Summary of p-values (F-test):")
    for lag in range(1, 6):
        p_value = gc_result[lag][0]["ssr_ftest"][1]
        significance = (
            "Significant Granger causality"
            if p_value < 0.05
            else "No significant Granger causality"
        )
        print(f"  Lag {lag}: p-value = {p_value:.4f} -> {significance}")

    print("\n" + "=" * 70)
    print("VAR MODEL ESTIMATION")
    print("=" * 70 + "\n")

    model_var = VAR(data_var_diff)
    lag_order = model_var.select_order(maxlags=15)
    print("Lag order selection criteria:")
    print(lag_order.summary())

    fitted_var = model_var.fit(lag_order.aic)
    print(f"\nFitted VAR({lag_order.aic}) model")
    print(fitted_var.summary())

    print("\n" + "=" * 70)
    print("VAR MODEL DIAGNOSTICS")
    print("=" * 70 + "\n")

    residuals_var = fitted_var.resid
    fig, axes = plt.subplots(len(data_var_diff.columns), 1, figsize=(15, 8))

    for i, col in enumerate(residuals_var.columns):
        axes[i].plot(residuals_var.index, residuals_var[col], linewidth=1)
        axes[i].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[i].set_title(f"Residuals: {col}", fontweight="bold")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    finish_figure(not args.no_show)

    print("\nDurbin-Watson test (should be close to 2):")
    for col in residuals_var.columns:
        print(f"  {col}: {durbin_watson(residuals_var[col]):.2f}")

    print("\n" + "=" * 70)
    print("VAR FORECASTING")
    print("=" * 70 + "\n")

    n_forecast = args.forecast_periods
    forecast_var = fitted_var.forecast(
        data_var_diff.values[-lag_order.aic :],
        steps=n_forecast,
    )

    forecast_index = pd.date_range(
        start=data_var.index[-1],
        periods=n_forecast + 1,
        freq=data_var.index.freq or "ME",
    )[1:]

    forecast_var_df = pd.DataFrame(
        forecast_var,
        index=forecast_index,
        columns=data_var.columns,
    )
    forecast_actual = forecast_var_df.cumsum() + data_var.iloc[-1]

    print("Forecast (first 10 periods):")
    print(forecast_actual.head(10))

    fig, axes = plt.subplots(
        len(data_var.columns), 1, figsize=(15, 10), sharex=True
    )
    for i, col in enumerate(data_var.columns):
        axes[i].plot(
            data_var.index[-100:],
            data_var[col][-100:],
            label="Historical",
            linewidth=2,
            color="blue",
        )
        axes[i].plot(
            forecast_actual.index,
            forecast_actual[col],
            label="Forecast",
            linewidth=2,
            color="red",
            linestyle="--",
        )
        axes[i].set_title(f"VAR Forecast: {col}", fontweight="bold")
        axes[i].legend(loc="best")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    finish_figure(not args.no_show)


def main() -> None:
    args = parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    run_arima_pipeline(args)

    if not args.skip_var:
        run_var_pipeline(args)


if __name__ == "__main__":
    main()
