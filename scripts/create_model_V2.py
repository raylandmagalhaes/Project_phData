import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

SALES_PATH = "../data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "../data/zipcode_demographics.csv"  # path to CSV with demographics
OUTPUT_DIR = "../models/model_V2"  # Directory where output artifacts will be saved
IGNORE_COLS = ["date", "id"]
#todo
#The date could be used to calculate how old the home was when sold.
#Using the log of price could improve the predictions

def load_data(
    sales_path: str, demographics_path: str, ignore_cols: List[str]) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           dtype={'zipcode': str},
                           usecols=lambda c: c not in ignore_cols)
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))



def create_and_evaluate_model():
    """Load data, train multiple models, evaluate them and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, IGNORE_COLS)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42, test_size=0.20)

    # Simple baseline (mean of training prices) for comparison
    mean_price = float(y_train.mean())
    yhat_base = np.full_like(y_test, mean_price, dtype=float)
    baseline = {
        "RMSE": rmse(y_test, yhat_base),
        "MAE": metrics.mean_absolute_error(y_test, yhat_base),
        "R2": metrics.r2_score(y_test, yhat_base),
        "mean_price_train": mean_price,
    }

    # Define models to evaluate
    models = {
        "KNN": pipeline.make_pipeline(preprocessing.RobustScaler(), neighbors.KNeighborsRegressor()),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    print("\n=== Model Evaluation on Extended Feature Set (80/20 split, seed=42) ===")
    results_summary = {}
    for name, mdl in models.items():
        # Fit on training data
        mdl.fit(x_train, y_train)
        # Predict on train/test sets
        yhat_train = mdl.predict(x_train)
        yhat_test = mdl.predict(x_test)
        # Metrics
        train_metrics = {
            "RMSE": rmse(y_train, yhat_train),
            "MAE": metrics.mean_absolute_error(y_train, yhat_train),
            "R2": metrics.r2_score(y_train, yhat_train),
        }
        test_metrics = {
            "RMSE": rmse(y_test, yhat_test),
            "MAE": metrics.mean_absolute_error(y_test, yhat_test),
            "R2": metrics.r2_score(y_test, yhat_test),
        }
        results_summary[name] = {
            "train": train_metrics,
            "test": test_metrics,
        }
        # Print summary
        print(f"\nModel: {name}")
        print(f"  Train: RMSE={train_metrics['RMSE']:.0f}  "
              f"MAE={train_metrics['MAE']:.0f}  R2={train_metrics['R2']:.3f}")
        print(f"  Test : RMSE={test_metrics['RMSE']:.0f}  "
              f"MAE={test_metrics['MAE']:.0f}  R2={test_metrics['R2']:.3f}")
        print(f"  Baseline RMSE={baseline['RMSE']:.0f}  MAE={baseline['MAE']:.0f}  R2={baseline['R2']:.3f}")

    # Determine best model based on test RMSE (lower is better)
    best_model_name = min(results_summary, key=lambda m: results_summary[m]["test"]["RMSE"])
    print(f"\nBest performing model by test RMSE: {best_model_name}")

    # Train and save only the best model on the full dataset.
    print(f"\nTraining best model ({best_model_name}) on the full dataset and saving to {OUTPUT_DIR}…")
    # Build and fit the best model on the entire data
    mdl = models[best_model_name]
    x_train_full, _x_test_full, y_train_full, _y_test_full = model_selection.train_test_split(
        x, y, random_state=42)
    mdl.fit(x_train_full, y_train_full)
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(exist_ok=True, parents=True)
    pickle.dump(mdl, open(outdir / "model.pkl", 'wb'))
    json.dump(list(x_train_full.columns), open(outdir / "model_features.json", 'w'))


if __name__ == "__main__":
    create_and_evaluate_model()

