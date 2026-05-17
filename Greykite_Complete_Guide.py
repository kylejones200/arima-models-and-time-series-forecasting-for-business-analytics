"""Generated from Jupyter notebook: Greykite Time Series Forecasting - Complete Guide

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig,
    MetadataParam,
    ModelComponentsParam,
)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from sklearn.metrics import mean_absolute_error, mean_squared_error


def advanced_forecast(df, horizon=12, regressor_cols=None):
    """
    Run advanced Greykite forecast with custom components.
    """
    print("=" * 60)
    print("ADVANCED GREYKITE FORECAST")
    print("=" * 60)
    forecaster = Forecaster()
    metadata_param = MetadataParam(
        time_col="ts", value_col="y", freq=None, regressor_cols=regressor_cols
    )
    model_components_param = ModelComponentsParam(
        custom={
            "growth": {"growth_term": "linear"},
            "seasonality": {
                "yearly_seasonality": "auto",
                "quarterly_seasonality": "auto",
                "monthly_seasonality": False,
                "weekly_seasonality": False,
                "daily_seasonality": False,
            },
            "events": {
                "holidays_to_model_separately": "auto",
                "holiday_lookup_countries": ["US", "UK"],
            },
            "changepoints": {
                "changepoints_dict": {
                    "method": "auto",
                    "resample_freq": "7D",
                    "regularization_strength": 0.5,
                    "potential_changepoint_n": 25,
                    "no_changepoint_distance_from_end": "365D",
                }
            },
            "fit_algorithm_dict": {"fit_algorithm": "ridge"},
            "extra_pred_cols": regressor_cols if regressor_cols else [],
        }
    )
    forecast_config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=horizon,
        coverage=0.95,
        metadata_param=metadata_param,
        model_components_param=model_components_param,
    )
    print(f"\nForecasting {horizon} periods ahead with custom components...")
    if regressor_cols:
        print(f"Using regressors: {regressor_cols}")
    result = forecaster.run_forecast_config(df=df, config=forecast_config)
    print("✓ Advanced forecast completed")
    return result


def evaluate_forecast(forecast_df):
    """
    Calculate evaluation metrics on in-sample fit.
    """
    historical = forecast_df[forecast_df["y"].notna()].copy()
    y_true = historical["y"]
    y_pred = historical["forecast"]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    residuals = y_true - y_pred
    print("=" * 60)
    print("FORECAST EVALUATION METRICS")
    print("=" * 60)
    print("\nIn-Sample Performance:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print("\nResidual Statistics:")
    print(f"  Mean:  {residuals.mean():.2f}")
    print(f"  Std:   {residuals.std():.2f}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(historical["ts"], residuals)
    axes[0, 0].axhline(y=0, color="r", linestyle="--")
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[1, 0].set_title("Actual vs Predicted")
    axes[1, 0].set_xlabel("Actual")
    axes[1, 0].set_ylabel("Predicted")
    axes[1, 0].grid(True, alpha=0.3)
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot")
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return {"mae": mae, "rmse": rmse, "mape": mape}


def load_data(source="uk_marriages"):
    """
    Load time series data from various sources.

    Parameters:
    -----------
    source : str
        'uk_marriages' - UK birth/marriage data
        'sample' - Create synthetic data
        'custom' - Load from your own CSV

    Returns:
    --------
    df : DataFrame with 'ts' (timestamp) and 'y' (target) columns
    """
    if source == "uk_marriages":
        try:
            df = pd.read_excel("Uk marriage data - unique.xlsx")
            df["Year"] = df["Year"].ffill().astype(int)
            month_map = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}
            df["Month"] = df["Quarter"].map(month_map)
            df["ts"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
            df = df[["ts", "Births", "Marriages"]].dropna()
            df = df.rename(columns={"Births": "y"})
            print(f"✓ Loaded UK marriage data: {len(df)} observations")
            print(f"  Date range: {df['ts'].min()} to {df['ts'].max()}")
            return df
        except FileNotFoundError:
            print("UK marriage data file not found. Creating sample data instead.")
            source = "sample"
    if source == "sample":
        print("Creating sample time series data...")
        dates = pd.date_range("2010-01-01", "2023-12-31", freq="MS")
        np.random.seed(42)
        trend = np.linspace(100, 200, len(dates))
        seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 5, len(dates))
        df = pd.DataFrame({"ts": dates, "y": trend + seasonality + noise})
        df["regressor_1"] = (
            50
            + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 6)
            + np.random.normal(0, 2, len(dates))
        )
        print(f"✓ Created sample data: {len(df)} observations")
        print(f"  Date range: {df['ts'].min()} to {df['ts'].max()}")
        return df
    if source == "custom":
        try:
            df = pd.read_csv("your_data.csv", parse_dates=["ts"])
            print(f"✓ Loaded custom data: {len(df)} observations")
            return df
        except FileNotFoundError:
            print("Custom file not found. Using sample data.")
            return load_data("sample")


def plot_forecast(forecast_df, n_historical=100, title="Greykite Forecast"):
    """
    Plot forecast with confidence intervals.
    """
    historical = forecast_df[forecast_df["y"].notna()].tail(n_historical)
    future = forecast_df[forecast_df["y"].isna()]
    plt.figure(figsize=(15, 6))
    plt.plot(
        historical["ts"],
        historical["y"],
        label="Historical",
        color="black",
        linewidth=2,
    )
    plt.plot(
        historical["ts"],
        historical["forecast"],
        label="Fitted",
        color="blue",
        linewidth=1,
        alpha=0.7,
    )
    plt.plot(
        future["ts"],
        future["forecast"],
        label="Forecast",
        color="red",
        linewidth=2,
        linestyle="--",
    )
    plt.fill_between(
        future["ts"],
        future["forecast_lower"],
        future["forecast_upper"],
        alpha=0.3,
        color="red",
        label="95% CI",
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def simple_forecast(df, horizon=12):
    """
    Run a simple Greykite forecast with default settings.

    Parameters:
    -----------
    df : DataFrame with 'ts' and 'y' columns
    horizon : int, forecast horizon

    Returns:
    --------
    result : Forecast result object
    """
    print("=" * 60)
    print("SIMPLE GREYKITE FORECAST")
    print("=" * 60)
    forecaster = Forecaster()
    metadata_param = MetadataParam(time_col="ts", value_col="y", freq=None)
    forecast_config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=horizon,
        coverage=0.95,
        metadata_param=metadata_param,
    )
    print(f"\nForecasting {horizon} periods ahead...")
    result = forecaster.run_forecast_config(df=df, config=forecast_config)
    print("✓ Forecast completed successfully")
    return result


def core_imports() -> None:
    warnings.filterwarnings("ignore")

    plt.style.use("seaborn-v0_8-darkgrid")

    print("✓ Greykite libraries loaded successfully")


def try_to_load_uk_marriage_data() -> None:
    df = load_data("uk_marriages")

    print("\nData structure:")

    print(df.head())

    print(f"\nColumns: {list(df.columns)}")

    print(f"Data types:\n{df.dtypes}")


def visualize_the_data() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].plot(df["ts"], df["y"], linewidth=1)

    axes[0].set_title("Target Variable Over Time", fontsize=14, fontweight="bold")

    axes[0].set_xlabel("Date")

    axes[0].set_ylabel("Value")

    axes[0].grid(False)

    axes[1].hist(df["y"], bins=50, edgecolor="black", alpha=0.7)

    axes[1].set_title("Distribution of Target Variable", fontsize=14, fontweight="bold")

    axes[1].set_xlabel("Value")

    axes[1].set_ylabel("Frequency")

    axes[1].grid(False)

    plt.tight_layout()

    plt.show()

    print("\nSummary Statistics:")

    print(df["y"].describe())


def create_forecaster() -> None:
    result = simple_forecast(df, horizon=12)

    forecast_df = result.forecast.df

    print("\nForecast Results (last 15 rows):")

    print(
        forecast_df[["ts", "y", "forecast", "forecast_lower", "forecast_upper"]].tail(
            15
        )
    )


def visualize_forecast() -> None:
    plot_forecast(forecast_df, title="Greykite Silverkite Forecast")


def metadata() -> None:
    regressor_cols = [col for col in df.columns if col not in ["ts", "y"]]

    if regressor_cols:
        print(f"Found regressors: {regressor_cols}")
        result_advanced = advanced_forecast(
            df, horizon=12, regressor_cols=regressor_cols
        )
    else:
        print("No regressors found, running without external variables")
        result_advanced = advanced_forecast(df, horizon=12, regressor_cols=None)

    forecast_advanced_df = result_advanced.forecast.df

    print("\nAdvanced Forecast Results (future periods):")

    future_only = forecast_advanced_df[forecast_advanced_df["y"].isna()]

    print(future_only[["ts", "forecast", "forecast_lower", "forecast_upper"]].head(12))


def plot_advanced_forecast() -> None:
    plot_forecast(
        forecast_advanced_df, title="Advanced Greykite Forecast with Custom Components"
    )


def get_historical_data_where_actual_values_exist() -> None:
    metrics = evaluate_forecast(forecast_advanced_df)


def get_model_summary() -> None:
    print("=" * 60)

    print("MODEL COMPONENTS SUMMARY")

    print("=" * 60)

    try:
        backtest = result_advanced.backtest
        print("\nBacktest results available")
        print(backtest.test_evaluation)
    except:
        print("\nBacktest not performed (use evaluation_period_param to enable)")

    model = result_advanced.model[-1]

    print("\n✓ Model trained successfully")

    print(f"  Model type: {type(model).__name__}")

    try:
        fig = result_advanced.forecast.plot_components()
        if fig:
            plt.show()
    except Exception as e:
        print(f"\nComponent plots not available: {e}")


def main() -> None:
    core_imports()
    try_to_load_uk_marriage_data()
    visualize_the_data()
    create_forecaster()
    visualize_forecast()
    metadata()
    plot_advanced_forecast()
    get_historical_data_where_actual_values_exist()
    get_model_summary()


if __name__ == "__main__":
    main()
