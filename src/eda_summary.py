import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import skew, kurtosis, entropy, spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller


def hurst_exponent(ts):
    """Safe Hurst exponent calculation."""
    ts = np.array(ts)
    if len(ts) < 100:
        return np.nan
    lags = range(2, min(100, len(ts)//2))
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def advanced_numeric_summary(
        df,
        dataset_name="dataset",
        time_col=None,
        target_col=None,
        include_hurst=False,
        rolling_window=5,
        save_format="csv",
        output_dir="."
    ):
    """
    Compute advanced EDA metrics for all numeric columns in dataframe.

    Returns:
        summary_df  (also saved to CSV/TXT with dataset name + timestamp)
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # remove time/target from analysis columns if present
    for col in [time_col, target_col]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    results = []

    for col in numeric_cols:

        s_raw = df[col]
        n_total = len(s_raw)

        # --- missing stats FIRST ---
        missing_count = s_raw.isna().sum()
        missing_pct = missing_count / n_total

        # drop NA for calculations
        s = s_raw.dropna()

        if len(s) == 0:
            continue

        # --- BASIC STATS ---
        mean_val = s.mean()
        median_val = s.median()
        std_val = s.std()

        skew_val = skew(s)
        kurt_val = kurtosis(s)

        unique_pct = s.nunique() / len(s)

        # --- DISTRIBUTION SHAPE ---
        q1, q3 = np.percentile(s, [25, 75])
        iqr_val = q3 - q1

        lower = q1 - 1.5 * iqr_val
        upper = q3 + 1.5 * iqr_val
        outlier_count = ((s < lower) | (s > upper)).sum()

        # entropy via histogram
        hist, _ = np.histogram(s, bins="auto", density=True)
        hist = hist[hist > 0]
        entropy_val = entropy(hist)

        # --- TEMPORAL METRICS ---
        lag1_acf = np.nan
        adf_p = np.nan
        roll_var_mean = np.nan
        trend_corr = np.nan
        hurst_val = np.nan

        if time_col is not None:

            ordered = df[[time_col, col]].dropna().sort_values(time_col)[col]

            if len(ordered) > 10:
                lag1_acf = ordered.autocorr(lag=1)

            if len(ordered) > 20:
                try:
                    adf_p = adfuller(ordered)[1]
                except:
                    pass

            if len(ordered) > rolling_window:
                roll_var_mean = ordered.rolling(rolling_window).var().mean()

            # correlation with time index (trend strength)
            time_index = np.arange(len(ordered))
            if len(ordered) > 2:
                trend_corr = np.corrcoef(time_index, ordered)[0,1]

            if include_hurst:
                try:
                    hurst_val = hurst_exponent(ordered.values)
                except:
                    pass

        # --- PREDICTIVE HINTS ---
        pearson_target = np.nan
        spearman_target = np.nan
        mi_target = np.nan

        if target_col is not None and target_col in df.columns:

            paired = df[[col, target_col]].dropna()
            if len(paired) > 5:

                pearson_target = paired[col].corr(paired[target_col])

                try:
                    spearman_target = spearmanr(
                        paired[col], paired[target_col]
                    )[0]
                except:
                    pass

                try:
                    mi_target = mutual_info_regression(
                        paired[[col]],
                        paired[target_col],
                        discrete_features=False
                    )[0]
                except:
                    pass

        results.append({
            "variable": col,
            "n": n_total,
            "missing_count": missing_count,
            "missing_pct": missing_pct,

            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "skew": skew_val,
            "kurtosis": kurt_val,
            "unique_pct": unique_pct,

            "iqr": iqr_val,
            "outlier_count": outlier_count,
            "entropy": entropy_val,

            "lag1_autocorr": lag1_acf,
            "adf_pvalue": adf_p,
            "rolling_var_mean": roll_var_mean,
            "trend_corr": trend_corr,
            "hurst": hurst_val,

            "pearson_target": pearson_target,
            "spearman_target": spearman_target,
            "mutual_info_target": mi_target
        })

    summary_df = pd.DataFrame(results)

    # --- SAVE FILE WITH DATASET NAME + TIMESTAMP ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_numeric_summary_{timestamp}"

    if save_format.lower() == "txt":
        path = f"{output_dir}/{filename}.txt"
        summary_df.to_string(open(path, "w"))
    else:
        path = f"{output_dir}/{filename}.csv"
        summary_df.to_csv(path, index=False)

    print(f"EDA summary saved to: {path}")

    return summary_df