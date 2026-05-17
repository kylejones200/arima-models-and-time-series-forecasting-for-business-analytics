"""Generated from Jupyter notebook: military expenditure and autocorrelation

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey


def main():
    # Load and clean data

    df = pd.read_csv(
        "data/NMC_Documentation-6.0/NMC-60-abridged/NMC-60-abridged.csv"
    )
    df = df[["milex", "irst", "pec"]].dropna()
    # Define dependent and independent variables
    Y = df["milex"]
    X = df[["irst", "pec"]]
    X = sm.add_constant(X)
    # Fit OLS model
    ols_model = sm.OLS(Y, X).fit()
    print(ols_model.summary())
    lm_test = acorr_breusch_godfrey(ols_model, nlags=3)
    lm_stat, lm_pvalue, f_stat, f_pvalue = lm_test
    print(f"LM Statistic: {lm_stat:.4f}, p-value: {lm_pvalue:.4f}")
    nw_model = ols_model.get_robustcov_results(cov_type="HAC", maxlags=3)
    print(nw_model.summary())


    # --- code cell ---

    import io

    from matplotlib import pyplot as plt

    # Capture the model summary output

    buffer = io.StringIO()

    print(nw_model.summary(), file=buffer)

    summary_text = buffer.getvalue()


    # Render the text as an image using Matplotlib

    plt.figure(figsize=(10, 7))

    plt.axis("off")  # Hide axes

    plt.text(0, 0, summary_text, fontsize=10, family="monospace")

    plt.savefig("model_summary.png")


    # --- code cell ---

    # Apply differencing
    df["milex_diff"] = df["milex"].diff()
    df = df.dropna()
    # Fit OLS on differenced data
    Y_diff = df["milex_diff"]
    X_diff = df[["irst", "pec"]]
    X_diff = sm.add_constant(X_diff)
    ols_diff_model = sm.OLS(Y_diff, X_diff).fit()

    buffer = io.StringIO()
    print(ols_diff_model.summary(), file=buffer)

    summary_text = buffer.getvalue()


    # Render the text as an image using Matplotlib

    plt.figure(figsize=(10, 7))

    plt.axis("off")  # Hide axes

    plt.text(0, 0, summary_text, fontsize=10, family="monospace")

    plt.savefig("model_summary.png")


    # --- code cell ---

    lm_test_diff = acorr_breusch_godfrey(ols_diff_model, nlags=3)
    print(
        f"LM Statistic (Differenced): {lm_test_diff[0]:.4f}, p-value: {lm_test_diff[1]:.4f}"
    )


    # --- code cell ---

    nw_diff_model = ols_diff_model.get_robustcov_results(cov_type="HAC", maxlags=3)
    print(nw_diff_model.summary())


    # --- code cell ---

    buffer = io.StringIO()
    print(nw_diff_model.summary(), file=buffer)

    summary_text = buffer.getvalue()


    # Render the text as an image using Matplotlib

    plt.figure(figsize=(10, 7))

    plt.axis("off")  # Hide axes

    plt.text(0, 0, summary_text, fontsize=10, family="monospace")

    plt.savefig("model_summary.png")


    # --- code cell ---

    import statsmodels.graphics.tsaplots as tsaplots

    # Filter dataset for only the USA
    df = pd.read_csv(
        "data/NMC_Documentation-6.0/NMC-60-abridged/NMC-60-abridged.csv"
    )
    df_usa = df[df["stateabb"] == "USA"].copy()

    # Apply differencing
    df_usa["milex_diff"] = df_usa["milex"].diff()
    df_usa = df_usa.dropna()

    # Define dependent and independent variables for USA model
    Y_usa = df_usa["milex"]
    X_usa = df_usa[["irst", "pec"]]
    X_usa = sm.add_constant(X_usa)

    # Fit OLS for USA data
    ols_usa_model = sm.OLS(Y_usa, X_usa).fit()

    # Fit OLS for differenced USA data
    Y_usa_diff = df_usa["milex_diff"]
    X_usa_diff = df_usa[["irst", "pec"]]
    X_usa_diff = sm.add_constant(X_usa_diff)

    ols_usa_diff_model = sm.OLS(Y_usa_diff, X_usa_diff).fit()

    # Extract residuals
    residuals_ols_usa = ols_usa_model.resid
    residuals_ols_usa_diff = ols_usa_diff_model.resid

    # Create plots for the USA
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.plot(
        df_usa["year"],
        df_usa["milex"],
        markersize=3,
        color="black",
        label="Military Expenditures (USA)",
    )
    ax.set_title("Military Expenditures Over Time for the USA")
    plt.savefig("milex_usa_over_time.png")
    plt.show()

    # ACF Plot for OLS Residuals (USA)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    tsaplots.plot_acf(residuals_ols_usa, lags=20, alpha=0.05, ax=ax)
    ax.set_title("Autocorrelation of OLS Residuals (USA)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    plt.savefig("ols_residuals_usa_acf.png")
    plt.show()

    # ACF Plot for Differenced Residuals (USA)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    tsaplots.plot_acf(residuals_ols_usa_diff, lags=20, alpha=0.05, ax=ax)
    ax.set_title("Autocorrelation of Differenced Residuals (USA)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    plt.savefig("differenced_residuals_usa_acf.png")
    plt.show()

    # Compare Military Expenditures Before and After Differencing (USA)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.plot(
        df_usa["year"],
        df_usa["milex"],
        label="Original MILEX (USA)",
        color="black",
        linestyle="-",
    )
    ax.plot(
        df_usa["year"],
        df_usa["milex_diff"],
        label="Differenced MILEX (USA)",
        color="gray",
        linestyle="--",
    )
    ax.set_title("Military Expenditures: Original vs Differenced (USA)")
    plt.savefig("milex_usa_original_vs_diff.png")
    plt.show()


    # --- code cell ---

    df.head()


    # --- duplicate code cell omitted (identical to earlier cell) ---


    # --- code cell ---

    df["stateabb"].nunique()


if __name__ == "__main__":
    main()
