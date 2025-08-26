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


def _load_model() -> Any:
    """Return the loaded model or raise a descriptive error."""
    if isinstance(_MODEL, Exception):
        raise RuntimeError(f"Could not load model: {_MODEL}")
    return _MODEL


def _prepare_input(records: Sequence[Dict[str, Any]], minimal: bool) -> pd.DataFrame:
    """Merge incoming records with demographic data and reorder columns.

    Args:
        records: An iterable of  dictionaries representing input examples.
        minimal: If True, expect only the core model features from the sales
            data.  Otherwise, expect the full set of home attributes (minus
            demographics) from ``future_unseen_examples.csv``.  In both
            cases the function will merge in demographics using ``zipcode`` and
            discard any extraneous fields before ordering the DataFrame
            according to MODEL_FEATURES.

    Returns:
        A pandas DataFrame ready to be passed directly into the model.
    """
    # Convert to DataFrame.  If the caller forgets to wrap a single dict in
    # a list, pandas will treat the keys as column names and each character
    # as a row; enforce list explicitly.
    df = pd.DataFrame(list(records))

    # Ensure zipcode exists and is a string for merging
    if "zipcode" not in df.columns:
        raise ValueError("Input data must include a 'zipcode' field")
    df["zipcode"] = df["zipcode"].astype(str)

    # Merge with demographics
    merged = df.merge(DEMOGRAPHICS_DF, how="left", on="zipcode")

    # Drop zipcode after merge; the model was trained without it
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])

    # For minimal endpoint we may get extra columns (shouldn't), drop all
    # columns not in MODEL_FEATURES to avoid passing unknown features.  For
    # full endpoint we also drop unknown columns.
    model_input = merged.reindex(columns=MODEL_FEATURES, fill_value=0)

    # Any missing columns will be filled with 0; ensure numeric types where
    # possible
    model_input = model_input.fillna(0)

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
        # Generate predictions from the loaded model and include the model
        # version in the response, mirroring the full‑feature endpoint.
        prediction = _predict(records, minimal=True)
        version = os.getenv("MODEL_VERSION", "unknown")
        response = {
            "model_version": version,
            "features": records,
            "prediction": prediction,
        }
        return JsonResponse(response)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
