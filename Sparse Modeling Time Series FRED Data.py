from pandas_datareader.data import DataReader
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


def fetch_fred(series_ids, start="2000-01-01"):
    df = pd.DataFrame()
    for series in series_ids:
        data = DataReader(series, "fred", start)
        data = data.rename(columns={series: series.lower()})
        df = pd.concat([df, data], axis=1)
    return df.dropna()


def create_lag_features(df, lags=(1, 2, 3)):
    lagged = df.copy()
    for col in df.columns:
        for lag in lags:
            lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    return lagged.dropna()


def main() -> None:
    series_list = ["WTISPLC", "CPIAUCNS", "INDPRO", "UNRATE", "T10Y2Y"]
    df = fetch_fred(series_list)
    df = df.resample("ME").mean().dropna()

    df_lagged = create_lag_features(df)
    target = "wtisplc"
    X = df_lagged.drop(columns=[target])
    y = df_lagged[target]

    tscv = TimeSeriesSplit(n_splits=5)
    model = make_pipeline(StandardScaler(), LassoCV(cv=tscv, max_iter=10000))
    model.fit(X, y)

    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    print(f"Mean Squared Error: {mse:.2f}")

    lasso = model.named_steps["lassocv"]
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected = coef[coef != 0]

    plt.figure(figsize=(10, 4))
    selected.sort_values().plot(kind="barh")
    plt.title("Selected Features (Non-Zero Coefficients)")
    plt.tight_layout()
    plt.savefig("lasso_selected_features.png")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(y.index, y, label="Actual")
    plt.plot(y.index, pred, label="LASSO Forecast", alpha=0.7)
    plt.title("Crude Oil Price Forecast with Sparse Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lasso_forecast.png")
    plt.show()


if __name__ == "__main__":
    main()
