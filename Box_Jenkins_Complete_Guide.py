"""Generated from Jupyter notebook: Box-Jenkins Methodology - Complete Guide

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Statistical tests
import statsmodels.api as sm
from pmdarima import auto_arima

# Plotting
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ARIMA models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Metrics

plt.style.use("seaborn-v0_8-darkgrid")
# %matplotlib inline  # Jupyter-only

print("✓ Box-Jenkins libraries loaded")


# --- code cell ---

# Example 1: Load your own data
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# data = df['value']

# Example 2: Use built-in dataset
from statsmodels.datasets import macrodata

df = macrodata.load_pandas().data
df.index = pd.period_range("1959Q1", periods=len(df), freq="Q").to_timestamp()
data = df["realgdp"]  # Real GDP

print(f"✓ Loaded data: {len(data)} observations")
print(f"  Period: {data.index[0]} to {data.index[-1]}")
print("\nFirst few values:")
print(data.head())

# Plot original data
plt.figure(figsize=(15, 5))
plt.plot(data.index, data.values, linewidth=2)
plt.title("Original Time Series", fontsize=16, fontweight="bold")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- code cell ---


def test_stationarity(timeseries, title="Time Series"):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    print(f"\n{'=' * 70}")
    print(f"STATIONARITY TEST: {title}")
    print(f"{'=' * 70}")

    # Perform ADF test
    result = adfuller(timeseries.dropna(), autolag="AIC")

    print("\nAugmented Dickey-Fuller Test:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print("  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")

    # Interpretation
    if result[1] <= 0.05:
        print("\n✓ STATIONARY (p-value ≤ 0.05)")
        print("  → Can proceed with ARIMA modeling")
        return True
    else:
        print("\n✗ NON-STATIONARY (p-value > 0.05)")
        print("  → Differencing required")
        return False


# Test original series
is_stationary = test_stationarity(data, "Original Series")


# --- code cell ---


def find_differencing_order(timeseries, max_d=3):
    """
    Find the minimum differencing order (d) needed for stationarity.
    """
    print(f"\n{'=' * 70}")
    print("DETERMINING DIFFERENCING ORDER (d)")
    print(f"{'=' * 70}\n")

    current_series = timeseries.copy()

    for d in range(max_d + 1):
        result = adfuller(current_series.dropna(), autolag="AIC")
        p_value = result[1]

        print(f"d={d}: p-value = {p_value:.6f}", end="")

        if p_value <= 0.05:
            print(" → ✓ STATIONARY")
            print(f"\n✓ Recommended d = {d}")
            return d, current_series
        else:
            print(" → Non-stationary, differencing...")
            current_series = current_series.diff()

    print(f"\n⚠️  Warning: Still non-stationary after {max_d} differences")
    return max_d, current_series


# Find optimal d
d_order, differenced_data = find_differencing_order(data)

# Plot original vs differenced
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Original
axes[0].plot(data.index, data.values, linewidth=2, color="blue")
axes[0].set_title("Original Series", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)

# Differenced
axes[1].plot(differenced_data.index, differenced_data.values, linewidth=2, color="red")
axes[1].set_title(f"Differenced Series (d={d_order})", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Differenced Value")
axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --- code cell ---


def plot_acf_pacf(timeseries, lags=40):
    """
    Plot ACF and PACF to identify p and q parameters.
    """
    print(f"\n{'=' * 70}")
    print("ACF/PACF ANALYSIS - Identifying p and q")
    print(f"{'=' * 70}\n")
    print("Guidelines:")
    print("  • ACF cuts off at lag q → Use MA(q) or q parameter")
    print("  • PACF cuts off at lag p → Use AR(p) or p parameter")
    print("  • Both tail off gradually → Use ARMA(p,q)\n")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ACF
    plot_acf(timeseries.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")

    # PACF
    plot_pacf(timeseries.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title(
        "Partial Autocorrelation Function (PACF)", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("PACF")

    plt.tight_layout()
    plt.show()


# Plot ACF/PACF for differenced series
plot_acf_pacf(differenced_data, lags=40)


# --- code cell ---


# Method 1: Manual grid search
def manual_arima_selection(timeseries, p_range, d, q_range):
    """
    Manually test different ARIMA(p,d,q) combinations.
    """
    print(f"\n{'=' * 70}")
    print("MANUAL ARIMA MODEL SELECTION")
    print(f"{'=' * 70}\n")

    best_aic = np.inf
    best_order = None
    best_model = None

    results = []

    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(timeseries, order=(p, d, q))
                fitted = model.fit()

                results.append(
                    {"p": p, "d": d, "q": q, "AIC": fitted.aic, "BIC": fitted.bic}
                )

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted

                print(f"ARIMA({p},{d},{q}): AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")

            except Exception as e:
                print(f"ARIMA({p},{d},{q}): Failed - {str(e)[:50]}")

    print(f"\n✓ Best Model: ARIMA{best_order} with AIC={best_aic:.2f}")

    # Show top 5 models
    results_df = pd.DataFrame(results).sort_values("AIC").head(5)
    print("\nTop 5 Models:")
    print(results_df.to_string(index=False))

    return best_model, best_order


# Try different combinations
manual_model, manual_order = manual_arima_selection(
    data, p_range=range(0, 4), d=d_order, q_range=range(0, 4)
)


# --- code cell ---


# Method 2: Automatic selection with auto_arima
def automatic_arima_selection(timeseries):
    """
    Use auto_arima to automatically find the best ARIMA model.
    """
    print(f"\n{'=' * 70}")
    print("AUTOMATIC ARIMA MODEL SELECTION (auto_arima)")
    print(f"{'=' * 70}\n")
    print("Searching for optimal parameters...")
    print("(This may take a minute)\n")

    auto_model = auto_arima(
        timeseries,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=None,  # Let auto_arima determine d
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action="ignore",
    )

    print(f"\n✓ Best Model: ARIMA{auto_model.order}")
    print(f"  AIC: {auto_model.aic():.2f}")
    print(f"  BIC: {auto_model.bic():.2f}")

    return auto_model


# Run auto_arima
auto_model = automatic_arima_selection(data)


# --- code cell ---


def diagnostic_plots(model, title="Model"):
    """
    Create comprehensive diagnostic plots for ARIMA model.
    """
    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC CHECKING: {title}")
    print(f"{'=' * 70}\n")

    # Get residuals
    residuals = model.resid

    # Create plots
    fig = plt.figure(figsize=(15, 10))

    # 1. Residuals over time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(residuals)
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_title("Residuals Over Time", fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Residual")
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of residuals
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax2.set_title("Histogram of Residuals", fontweight="bold")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q plot
    ax3 = plt.subplot(2, 2, 3)
    sm.qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # 4. ACF of residuals
    ax4 = plt.subplot(2, 2, 4)
    plot_acf(residuals, lags=40, ax=ax4)
    ax4.set_title("ACF of Residuals", fontweight="bold")
    ax4.set_xlabel("Lag")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\nLjung-Box Test for Residual Autocorrelation:")
    print("(p-value > 0.05 indicates no significant autocorrelation)\n")
    print(lb_test)

    # Summary statistics
    print("\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Skewness: {residuals.skew():.4f}")
    print(f"  Kurtosis: {residuals.kurtosis():.4f}")


# Run diagnostics on the auto_arima model
diagnostic_plots(auto_model, f"ARIMA{auto_model.order}")


# --- code cell ---

# Model summary
print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70 + "\n")
print(auto_model.summary())


# --- code cell ---


def forecast_arima(model, original_data, n_periods=20, alpha=0.05):
    """
    Generate forecasts with confidence intervals.
    """
    print(f"\n{'=' * 70}")
    print(f"FORECASTING {n_periods} PERIODS AHEAD")
    print(f"{'=' * 70}\n")

    # Generate forecast
    forecast_result = model.predict(
        n_periods=n_periods, return_conf_int=True, alpha=alpha
    )
    forecast = forecast_result[0]
    conf_int = forecast_result[1]

    # Create forecast index
    last_date = original_data.index[-1]
    freq = original_data.index.freq or pd.infer_freq(original_data.index)
    forecast_index = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq)[
        1:
    ]

    # Create DataFrame
    forecast_df = pd.DataFrame(
        {"Forecast": forecast, "Lower_CI": conf_int[:, 0], "Upper_CI": conf_int[:, 1]},
        index=forecast_index,
    )

    print(forecast_df.head(10))

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))

    # Historical data
    ax.plot(
        original_data.index[-100:],
        original_data.values[-100:],
        label="Historical",
        linewidth=2,
        color="blue",
    )

    # Forecast
    ax.plot(
        forecast_index,
        forecast,
        label="Forecast",
        linewidth=2,
        color="red",
        linestyle="--",
    )

    # Confidence interval
    ax.fill_between(
        forecast_index,
        conf_int[:, 0],
        conf_int[:, 1],
        alpha=0.3,
        color="red",
        label=f"{(1 - alpha) * 100}% CI",
    )

    ax.set_title(f"ARIMA{model.order} Forecast", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return forecast_df


# Generate forecast
forecast_df = forecast_arima(auto_model, data, n_periods=20)


# --- code cell ---

# Additional imports for VAR
from pandas_datareader.data import DataReader
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

print("✓ VAR libraries loaded")


# --- code cell ---

# Example: Load economic data from FRED
# (Requires: pip install pandas-datareader)

start = datetime(2015, 1, 1)
end = datetime.today()

try:
    # Fetch data from FRED
    indpro = DataReader("INDPRO", "fred", start, end)  # Industrial Production
    rsafs = DataReader("RSAFS", "fred", start, end)  # Retail Sales

    # Merge and clean
    data_var = pd.concat([indpro, rsafs], axis=1)
    data_var.columns = ["Industrial_Production", "Retail_Sales"]
    data_var = data_var.dropna()

    print(f"✓ Loaded multivariate data: {len(data_var)} observations")
    print(f"  Variables: {list(data_var.columns)}")
    print("\nFirst few rows:")
    print(data_var.head())

except Exception as e:
    print(f"Could not fetch FRED data: {e}")
    print("Creating synthetic multivariate data instead...")

    # Create synthetic data
    np.random.seed(42)
    time = pd.date_range(start="2015-01", periods=100, freq="ME")
    indpro = 50 + np.cumsum(np.random.normal(0, 2, 100))
    rsafs = 30 + 0.5 * indpro + np.random.normal(0, 2, 100)

    data_var = pd.DataFrame(
        {"Industrial_Production": indpro, "Retail_Sales": rsafs}, index=time
    )

    print(f"✓ Created synthetic data: {len(data_var)} observations")

# Plot the multivariate data
fig, ax = plt.subplots(figsize=(15, 6))
data_var.plot(ax=ax, linewidth=2)
ax.set_title("Multivariate Time Series", fontsize=16, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- code cell ---

print("\n" + "=" * 70)
print("STATIONARITY TEST FOR MULTIVARIATE SERIES")
print("=" * 70 + "\n")

for col in data_var.columns:
    result = adfuller(data_var[col])
    print(f"{col}:")
    print(f"  p-value = {result[1]:.4f}", end="")

    if result[1] > 0.05:
        print(" → Non-stationary. Differencing required.")
    else:
        print(" → Stationary.")
    print()

# Apply differencing if needed
data_var_diff = data_var.diff().dropna()

print("\nAfter differencing:")
for col in data_var_diff.columns:
    result = adfuller(data_var_diff[col])
    print(f"{col}: p-value = {result[1]:.4f}")

# Plot differenced data
fig, ax = plt.subplots(figsize=(15, 6))
data_var_diff.plot(ax=ax, linewidth=2)
ax.set_title("Differenced Multivariate Series", fontsize=16, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Differenced Value")
ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- code cell ---

print("\n" + "=" * 70)
print("GRANGER CAUSALITY TEST")
print("=" * 70)
print("\nTests if one series helps predict another.")
print("Null hypothesis: X does NOT Granger-cause Y\n")

# Test if Retail Sales Granger-causes Industrial Production
col1, col2 = data_var_diff.columns[0], data_var_diff.columns[1]
print(f"\nTesting if {col2} Granger-causes {col1}:")
print("=" * 70)

gc_result = grangercausalitytests(data_var_diff[[col1, col2]], maxlag=5, verbose=True)

# Extract p-values
print("\nSummary of p-values (F-test):")
for lag in range(1, 6):
    p_value = gc_result[lag][0]["ssr_ftest"][1]
    print(f"  Lag {lag}: p-value = {p_value:.4f}", end="")
    if p_value < 0.05:
        print(" → Significant Granger causality")
    else:
        print(" → No significant Granger causality")


# --- code cell ---

print("\n" + "=" * 70)
print("VAR MODEL ESTIMATION")
print("=" * 70 + "\n")

# Create VAR model
model_var = VAR(data_var_diff)

# Select optimal lag order
lag_order = model_var.select_order(maxlags=15)
print("Lag Order Selection Criteria:")
print(lag_order.summary())

# Fit model with optimal lags (using AIC)
fitted_var = model_var.fit(lag_order.aic)
print(f"\n✓ Fitted VAR({lag_order.aic}) model")
print("\nModel Summary:")
print(fitted_var.summary())


# --- code cell ---

print("\n" + "=" * 70)
print("VAR MODEL DIAGNOSTICS")
print("=" * 70 + "\n")

# Get residuals
residuals_var = fitted_var.resid

# Plot residuals
fig, axes = plt.subplots(len(data_var_diff.columns), 1, figsize=(15, 8))

for i, col in enumerate(residuals_var.columns):
    axes[i].plot(residuals_var.index, residuals_var[col], linewidth=1)
    axes[i].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[i].set_title(f"Residuals: {col}", fontweight="bold")
    axes[i].set_ylabel("Residual")
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()

# Durbin-Watson test for serial correlation
print("\nDurbin-Watson Test (should be close to 2):")
for col in residuals_var.columns:
    dw_stat = durbin_watson(residuals_var[col])
    print(f"  {col}: {dw_stat:.2f}")


# --- code cell ---

print("\n" + "=" * 70)
print("VAR FORECASTING")
print("=" * 70 + "\n")

# Forecast next 20 periods
n_forecast = 20
forecast_var = fitted_var.forecast(
    data_var_diff.values[-lag_order.aic :], steps=n_forecast
)

# Create forecast DataFrame
forecast_index = pd.date_range(
    start=data_var.index[-1], periods=n_forecast + 1, freq=data_var.index.freq or "M"
)[1:]

forecast_var_df = pd.DataFrame(
    forecast_var, index=forecast_index, columns=data_var.columns
)

# Convert back from differences
forecast_actual = forecast_var_df.cumsum() + data_var.iloc[-1]

print("Forecast (first 10 periods):")
print(forecast_actual.head(10))

# Plot forecast
fig, axes = plt.subplots(len(data_var.columns), 1, figsize=(15, 10), sharex=True)

for i, col in enumerate(data_var.columns):
    # Historical
    axes[i].plot(
        data_var.index[-100:],
        data_var[col][-100:],
        label="Historical",
        linewidth=2,
        color="blue",
    )

    # Forecast
    axes[i].plot(
        forecast_actual.index,
        forecast_actual[col],
        label="Forecast",
        linewidth=2,
        color="red",
        linestyle="--",
    )

    axes[i].set_title(f"VAR Forecast: {col}", fontweight="bold")
    axes[i].set_ylabel("Value")
    axes[i].legend(loc="best")
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()
