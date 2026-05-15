# Description: Short example for ARIMA Models and Time Series Forecasting for Business Analytics.


# Load the time series data

from data_io import read_csv
from statsmodels.tsa.arima_model import ARIMA


def main():
    data = read_csv("sales_data.csv", index_col="date")
    # Fit the ARIMA(1,1,1) model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    # Make forecasts
    forecast = model_fit.forecast(steps=10)[0]


if __name__ == "__main__":
    main()
