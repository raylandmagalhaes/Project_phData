import json
import pathlib
import pickle
from typing import List, Tuple

import numpy as np
import pandas
from sklearn import model_selection, neighbors, pipeline, preprocessing, metrics
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
OUTPUT_DIR = "models/model_V2"
IGNORE_COLS = ["id"]


def _add_date_features(df: pandas.DataFrame, date_col: str = "date") -> pandas.DataFrame:
    """From 'YYYYMMDDThhmmss' strings, build ML-friendly date features."""
    # Robust parse: e.g., 20141013T000000
    sale_dt = pandas.to_datetime(df[date_col], format="%Y%m%dT%H%M%S", errors="coerce")

    # Basic calendar features
    df["sale_year"] = sale_dt.dt.year.astype("Int64")
    df["sale_month"] = sale_dt.dt.month.astype("Int64")
    df["sale_dayofweek"] = sale_dt.dt.dayofweek.astype("Int64")  # Monday=0
    df["sale_quarter"] = sale_dt.dt.quarter.astype("Int64")
    df["sale_dayofyear"] = sale_dt.dt.dayofyear.astype("Int64")
    df["sale_isoweek"] = sale_dt.dt.isocalendar().week.astype("Int64")

    # Seasonal cyclic encoding (helps linear/KNN models)
    # Scale day of year to [0, 2π)
    two_pi = 2 * np.pi
    frac = (df["sale_dayofyear"].astype(float) - 1) / 365.25
    df["sale_season_sin"] = np.sin(two_pi * frac)
    df["sale_season_cos"] = np.cos(two_pi * frac)

    # Age at sale (accounts for renovation when available)
    # kc_house_data has 'yr_built' and 'yr_renovated'
    if "yr_built" in df.columns:
        eff_year = df["yr_built"].copy()
        if "yr_renovated" in df.columns:
            # if renovated (>0), use the more recent year
            eff_year = np.where(df["yr_renovated"].fillna(0) > 0,
                                np.maximum(df["yr_built"].fillna(0), df["yr_renovated"].fillna(0)),
                                df["yr_built"].fillna(0))
        df["home_age_at_sale"] = (df["sale_year"].astype(float) - eff_year.astype(float)).clip(lower=0)
    else:
        df["home_age_at_sale"] = np.nan

    df.drop(columns=[date_col], inplace=True)
    return df


def load_data(
    sales_path: str, demographics_path: str, ignore_cols: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Returns:
        (X, y) where X are features and y is the *original* price (not logged).
        The model wrappers handle the log transform internally.
    """
    sales = pandas.read_csv(
        sales_path,
        dtype={"zipcode": str},
        usecols=lambda c: c not in ignore_cols  # keep 'date'
    )
    # add engineered date features
    if "date" not in sales.columns:
        raise ValueError("Expected a 'date' column in the sales CSV (e.g., 20141013T000000).")
    sales = _add_date_features(sales, date_col="date")

    demographics = pandas.read_csv(demographics_path, dtype={"zipcode": str})

    merged = sales.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    # target (keep on original $ scale; we'll log-transform via TransformedTargetRegressor)
    y = merged.pop("price").astype(float)
    X = merged

    return X, y


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))


def _wrap_with_log_target(reg):
    """Return a regressor that trains on log(price) but predicts on $ scale."""
    # natural log / exp; price>0 in this dataset
    return TransformedTargetRegressor(
        regressor=reg,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def create_and_evaluate_model():
    """Load data, train multiple models (on log(price)), evaluate, and export artifacts."""
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, IGNORE_COLS)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=42, test_size=0.20
    )

    # Baseline on original scale: predict the mean training price
    mean_price = float(y_train.mean())
    yhat_base = np.full(shape=y_test.shape, fill_value=mean_price, dtype=float)
    baseline = {
        "RMSE_$": rmse(y_test, yhat_base),
        "MAE_$": metrics.mean_absolute_error(y_test, yhat_base),
        "R2_$": metrics.r2_score(y_test, yhat_base),
        "mean_price_train": mean_price,
    }

    # Define models (each wrapped to learn log(price) but predict $)
    models = {
        "KNN": _wrap_with_log_target(
            pipeline.make_pipeline(
                preprocessing.RobustScaler(),
                neighbors.KNeighborsRegressor()
            )
        ),
        "RandomForest": _wrap_with_log_target(
            RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
            )
        ),
        "XGBoost": _wrap_with_log_target(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    print("\n=== Model Evaluation (train on log(price), metrics on $) "
          "(80/20 split, seed=42) ===")
    results_summary = {}
    for name, mdl in models.items():
        # Fit on training data
        mdl.fit(X_train, y_train)

        # Predict on train/test in ORIGINAL $ scale thanks to TransformedTargetRegressor
        yhat_train = mdl.predict(X_train)
        yhat_test = mdl.predict(X_test)

        # Metrics on original $ scale
        train_metrics = {
            "RMSE_$": rmse(y_train, yhat_train),
            "MAE_$": metrics.mean_absolute_error(y_train, yhat_train),
            "R2_$": metrics.r2_score(y_train, yhat_train),
        }
        test_metrics = {
            "RMSE_$": rmse(y_test, yhat_test),
            "MAE_$": metrics.mean_absolute_error(y_test, yhat_test),
            "R2_$": metrics.r2_score(y_test, yhat_test),
        }
        results_summary[name] = {"train": train_metrics, "test": test_metrics}

        # Print summary
        print(f"\nModel: {name}")
        print(f"  Train: RMSE=${train_metrics['RMSE_$']:.0f}  "
              f"MAE=${train_metrics['MAE_$']:.0f}  R2={train_metrics['R2_$']:.3f}")
        print(f"  Test : RMSE=${test_metrics['RMSE_$']:.0f}  "
              f"MAE=${test_metrics['MAE_$']:.0f}  R2={test_metrics['R2_$']:.3f}")
        print(f"  Baseline RMSE=${baseline['RMSE_$']:.0f}  "
              f"MAE=${baseline['MAE_$']:.0f}  R2={baseline['R2_$']:.3f}")

        # Residuals (on $ scale)
        residuals = y_test - yhat_test

        # Figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Residuals vs Predicted
        axes[0].scatter(yhat_test, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel("Predicted Price ($)")
        axes[0].set_ylabel("Residuals ($)")
        axes[0].set_title("Residuals vs Predicted")

        # Histogram of Residuals
        sns.histplot(residuals, kde=True, ax=axes[1], color="blue")
        axes[1].set_xlabel("Residuals ($)")
        axes[1].set_title(f"{name}: Distribution of Residuals")

        # Q-Q Plot (residual normality in $ space will often be skewed; that's okay)
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot of Residuals")

        plt.tight_layout()
        plt.show()

    # Determine best model by test RMSE on $ scale (lower is better)
    best_model_name = min(results_summary, key=lambda m: results_summary[m]["test"]["RMSE_$"])
    print(f"\nBest performing model by test RMSE: {best_model_name}")

    # Train and save only the best model on the full dataset.
    print(f"\nTraining best model ({best_model_name}) on the full dataset and saving to {OUTPUT_DIR} …")
    best_model = models[best_model_name]

    # Refit on a larger split to leverage more data while still fixing randomness
    X_train_full, _X_tmp, y_train_full, _y_tmp = model_selection.train_test_split(
        X, y, random_state=42
    )
    best_model.fit(X_train_full, y_train_full)

    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(exist_ok=True, parents=True)
    # This pickle contains the TransformedTargetRegressor wrapper
    pickle.dump(best_model, open(outdir / "model.pkl", "wb"))
    json.dump(list(X_train_full.columns), open(outdir / "model_features.json", "w"))


if __name__ == "__main__":
    create_and_evaluate_model()
