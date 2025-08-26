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
import numpy as np

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
OUTPUT_DIR = "models/model_V2"  # Directory where output artifacts will be saved
IGNORE_COLS = ["date", "id"]

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
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, IGNORE_COLS)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42, test_size=0.20)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)

    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)

    results = {
        "train": {
            "RMSE": rmse(y_train, yhat_train),
            "MAE": metrics.mean_absolute_error(y_train, yhat_train),
            "R2": metrics.r2_score(y_train, yhat_train),
            "n": int(len(y_train)),
        },
        "test": {
            "RMSE": rmse(y_test, yhat_test),
            "MAE": metrics.mean_absolute_error(y_test, yhat_test),
            "R2": metrics.r2_score(y_test, yhat_test),
            "n": int(len(y_test)),
        },
    }

    # 5) Simple baseline (mean of training prices)
    mean_price = float(y_train.mean())
    yhat_base = np.full_like(y_test, mean_price, dtype=float)
    baseline = {
        "RMSE": rmse(y_test, yhat_base),
        "MAE": metrics.mean_absolute_error(y_test, yhat_base),
        "R2": metrics.r2_score(y_test, yhat_base),
        "mean_price_train": mean_price,
    }

    # 6) Print concise report
    print("\n=== Baseline KNN Evaluation (random 80/20 split, seed=42) ===")
    print(f"Train: RMSE={results['train']['RMSE']:.0f}  "
          f"MAE={results['train']['MAE']:.0f}  R2={results['train']['R2']:.3f}  n={results['train']['n']}")
    print(f"Test : RMSE={results['test']['RMSE']:.0f}  "
          f"MAE={results['test']['MAE']:.0f}  R2={results['test']['R2']:.3f}  n={results['test']['n']}")
    print(f"Base : RMSE={baseline['RMSE']:.0f}  "
          f"MAE={baseline['MAE']:.0f}  R2={baseline['R2']:.3f}  "
          f"(mean train price = {baseline['mean_price_train']:.0f})")

    # 7) Quick verdict
    tr, te = results["train"], results["test"]
    verdict = []
    if tr["RMSE"] < te["RMSE"] * 0.8 and tr["R2"] > te["R2"] + 0.1:
        verdict.append("Model likely **overfits** (train much better than test).")
    elif tr["R2"] < 0.2 and te["R2"] < 0.2:
        verdict.append("Model likely **underfits** (low R² on both).")
    else:
        verdict.append("Fit looks **reasonable** for a simple KNN baseline.")
    if te["RMSE"] >= baseline["RMSE"] * 0.98:
        verdict.append("Warning: test error is close to the **mean-price baseline**; limited predictive value.")

    print("\nVerdict:")
    for line in verdict:
        print("-", line)

    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
        x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))


if __name__ == "__main__":
    create_and_evaluate_model()

