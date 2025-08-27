"""
Views for serving machine‑learning predictions.

This module exposes two endpoints:

* ``/predict/`` – Accepts POSTed JSON payloads containing all of the
  non‑demographic features from ``future_unseen_examples.csv`` and returns
  predictions from the trained model.  Demographic information is merged
  automatically on the backend by joining against the ``zipcode`` column.

* ``/predict_core/`` – A convenience endpoint which requires only the subset
  of features actually used to train the base model.  This is useful when
  callers don't wish to supply the additional home attributes that are not
  consumed by the model.

Both endpoints accept either a single JSON object or a list of objects.  The
response is a JSON object containing a list of predictions and some basic
metadata.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd  # type: ignore
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt

# Preload the model, feature list and demographic data.  Doing this at module
# import time avoids reloading the pickled model on every request.  Should a
# newer model be deployed, you can update the file paths via environment
# variables without modifying code.  To support hot reloading you could
# implement file watchers or route traffic to a new container while the old
# model drains existing connections.

# Determine project base directory.  This file lives in
# <BASE_DIR>/predictor/views.py, so the parent of the parent is BASE_DIR.
BASE_DIR = Path(__file__).resolve().parents[1]


def _get_env_path(env_name: str, default: str) -> Path:
    """Return a Path from an environment variable or a default.

    Using environment variables for model and data paths allows for easy
    upgrades of the model without changing the code.  During a blue‑green
    deployment you can point new containers at a new model path while the
    existing containers continue to serve the old model.
    """
    env_value = os.getenv(env_name)
    if env_value:
        return Path(env_value)
    return BASE_DIR / default


# Paths to model artifacts and demographic data.  These can be overridden by
# setting MODEL_PATH, MODEL_FEATURES_PATH and DEMOGRAPHICS_PATH environment
# variables.  See docker-compose.yml for examples of how to mount a new model
# version and set these variables during deployment.
MODEL_PATH = _get_env_path("MODEL_PATH", "models/model/model.pkl")
FEATURES_PATH = _get_env_path("MODEL_FEATURES_PATH", "models/model/model_features.json")
DEMOGRAPHICS_PATH = _get_env_path("DEMOGRAPHICS_PATH", "data/zipcode_demographics.csv")

try:
    with open(MODEL_PATH, "rb") as f:
        _MODEL = None  # type: ignore
        _MODEL = pd.read_pickle(f)  # type: ignore[attr-defined]
except Exception as exc:
    # Model could not be loaded.  Delay raising until a request comes in so
    # Django still starts up (useful for health endpoints).
    _MODEL = exc  # type: ignore

try:
    with open(FEATURES_PATH, "r") as f:
        MODEL_FEATURES: List[str] = json.load(f)
except Exception as exc:
    MODEL_FEATURES = []
    _feature_error = exc

try:
    # Read demographics as strings for zipcode to avoid type mismatches
    DEMOGRAPHICS_DF = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})
except Exception as exc:
    DEMOGRAPHICS_DF = pd.DataFrame()
    _demographics_error = exc

# Minimal feature set required by /predict_core/ (plus zipcode for the merge)
CORE_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


def _add_date_features(df: pd.DataFrame, date_col: str = "date", default_tz: str = "UTC") -> pd.DataFrame:
    """
    From 'YYYYMMDDThhmmss' strings, build ML-friendly date features.
    If the date is missing, default to *today* (midnight, `default_tz`).
    """
    print("im here")
    if date_col in df.columns:
        sale_dt = pd.to_datetime(df[date_col], format="%Y%m%dT%H%M%S", errors="coerce")
    else:
        sale_dt = pd.Series(pd.NaT, index=df.index)

    # Default any NaT to "today" at midnight in the chosen timezone
    today = pd.Timestamp.now(tz=default_tz).normalize()
    sale_dt = sale_dt.fillna(today)
    print(sale_dt)
    # Basic calendar features
    df["sale_year"] = sale_dt.dt.year.astype("Int64")
    df["sale_month"] = sale_dt.dt.month.astype("Int64")
    df["sale_dayofweek"] = sale_dt.dt.dayofweek.astype("Int64")  # Monday=0
    df["sale_quarter"] = sale_dt.dt.quarter.astype("Int64")
    df["sale_dayofyear"] = sale_dt.dt.dayofyear.astype("Int64")
    df["sale_isoweek"] = sale_dt.dt.isocalendar().week.astype("Int64")

    # Seasonal cyclic encoding (helps linear/KNN models)
    two_pi = 2 * np.pi
    frac = (df["sale_dayofyear"].astype(float) - 1) / 365.25
    df["sale_season_sin"] = np.sin(two_pi * frac)
    df["sale_season_cos"] = np.cos(two_pi * frac)

    # Age at sale (accounts for renovation when available)
    if "yr_built" in df.columns:
        yr_built = pd.to_numeric(df["yr_built"], errors="coerce").fillna(0).astype(float)
        if "yr_renovated" in df.columns:
            yr_reno = pd.to_numeric(df["yr_renovated"], errors="coerce").fillna(0).astype(float)
            eff_year = np.where(yr_reno > 0, np.maximum(yr_built, yr_reno), yr_built)
        else:
            eff_year = yr_built
        df["home_age_at_sale"] = (df["sale_year"].astype(float) - eff_year).clip(lower=0)
    else:
        df["home_age_at_sale"] = np.nan

    # Remove the raw timestamp column if present
    df.drop(columns=[date_col], inplace=True, errors="ignore")
    print("DONE")
    return df


def _validate_required(records: Sequence[Dict[str, Any]], required: Sequence[str]) -> None:
    """Raise ValueError if any record is missing required keys or has nulls."""
    for i, rec in enumerate(records):
        missing = [k for k in required if k not in rec]
        nulls = [k for k in required if k in rec and rec[k] is None]
        if missing or nulls:
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if nulls:
                parts.append(f"nulls={nulls}")
            raise ValueError(f"Record {i} invalid: " + "; ".join(parts))


def _load_model() -> Any:
    """Return the loaded model or raise a descriptive error."""
    if isinstance(_MODEL, Exception):
        raise RuntimeError(f"Could not load model: {_MODEL}")
    return _MODEL


def _prepare_input(records: Sequence[Dict[str, Any]], minimal: bool) -> pd.DataFrame:
    """Merge incoming records with demographic data and reorder columns.

    Args:
        records: An iterable of dictionaries representing input examples.
        minimal: If True, expect only the core model features from the sales
            data. Otherwise, expect the full set of home attributes (minus
            demographics) from ``future_unseen_examples.csv``. In both
            cases the function will merge in demographics using ``zipcode`` and
            discard any extraneous fields before ordering the DataFrame
            according to MODEL_FEATURES.

    Returns:
        A pandas DataFrame ready to be passed directly into the model.
    """
    # Enforce list explicitly
    df = pd.DataFrame(list(records))

    # Ensure zipcode exists and is a string for merging
    if "zipcode" not in df.columns:
        raise ValueError("Input data must include a 'zipcode' field")
    df["zipcode"] = df["zipcode"].astype(str)

    # Expand 'date' into engineered features (defaults to today if missing/unparsable)
    df = _add_date_features(df, date_col="date")

    # Merge with demographics
    merged = df.merge(DEMOGRAPHICS_DF, how="left", on="zipcode")

    # Drop zipcode after merge; the model was trained without it
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])

    # Reindex to the model feature set, fill missing with 0
    model_input = merged.reindex(columns=MODEL_FEATURES, fill_value=0).fillna(0)

    return model_input


def _predict(records: Sequence[Dict[str, Any]], minimal: bool) -> List[float]:
    """Prepare input, invoke the model and return predictions."""
    model = _load_model()
    input_df = _prepare_input(records, minimal=minimal)
    preds = model.predict(input_df)
    return preds.tolist()  # type: ignore


@csrf_exempt
def predict_view(request: HttpRequest) -> JsonResponse:
    """Handle POST requests for full feature prediction.

    The incoming JSON may be a dictionary or a list of dictionaries.  Each
    dictionary should include the columns from ``future_unseen_examples.csv``
    (excluding demographics).  The response contains the predicted price(s)
    under the ``predictions`` key along with metadata such as the model
    version and the time of inference.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    try:
        body = request.body.decode("utf-8")
        data = json.loads(body)
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("Invalid JSON payload: must be an object or list of objects")
        # Generate predictions from the loaded model and build a response
        # that includes the model version.  The model version is read from
        # the ``MODEL_VERSION`` environment variable (defaulting to
        # "unknown" if not set).
        prediction = _predict(records, minimal=False)
        version = os.getenv("MODEL_VERSION", "unknown")
        response = {
            "model_version": version,
            "features": records,
            "prediction": prediction,
        }
        return JsonResponse(response)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@csrf_exempt
def predict_core_view(request: HttpRequest) -> JsonResponse:
    """Handle POST requests for minimal feature prediction.

    Callers only need to supply the core features used by the base model
    (bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above,
    sqft_basement and zipcode).  All other model features are derived from
    demographics.  The same JSON formats as ``predict_view`` are accepted.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    try:
        body = request.body.decode("utf-8")
        data = json.loads(body)
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("Invalid JSON payload: must be an object or list of objects")

        # Strict validation for core endpoint
        _validate_required(records, CORE_FEATURES)

        predictions = _predict(records, minimal=True)
        version = os.getenv("MODEL_VERSION", "unknown")
        response = {
            "model_version": version,
            "features": records,
            "prediction": predictions,
        }
        return JsonResponse(response)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
