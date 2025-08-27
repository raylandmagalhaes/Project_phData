"""
test_api_compare.py
====================

This script exercises the house price prediction API in order to compare the
outputs of two different versions of the model running behind the same
endpoint.  The API is expected to expose a `/predict/` route which accepts
the full set of home attributes and returns a JSON object containing a
`model_version` field (e.g. ``"1.0"`` or ``"2.0"``) along with the
predicted sale price.  When the NGINX load balancer is configured with
equal weights for the blue and green services (e.g. weight=5 each) the
requests should deterministically alternate between the two models.

The script reads a handful of examples from ``data/future_unseen_examples.csv``,
sends each example repeatedly to the API until it has observed a response
from both model versions, and stores the predictions.  It then prints the
results and the difference between the two model outputs for each example.

Usage:

.. code-block:: bash

    docker-compose up --build -d  # ensure the API is running with both versions
    python test_api_compare.py

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd  # type: ignore
import requests


def fetch_predictions(example: Dict[str, Any], url: str, max_attempts: int = 10) -> Dict[str, float]:
    """Send repeated requests for a single example until predictions from both
    model versions have been observed or max_attempts is reached.
    Make sure web_green is up before using.

    Args:
        example: The record to send to the prediction endpoint.
        url: The full URL of the prediction endpoint (e.g. ``http://localhost:8000/predict/``).
        max_attempts: The maximum number of times to call the endpoint before
            giving up.

    Returns:
        A mapping from model version string to predicted price.  If only one
        version responds within the allotted attempts the mapping will
        contain a single entry.
    """
    preds_by_version: Dict[str, float] = {}
    attempts = 0
    while len(preds_by_version) < 2 and attempts < max_attempts:
        attempts += 1
        try:
            resp = requests.post(url, json=example)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Error calling API: {exc}")
        try:
            data: Dict[str, Any] = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response: {resp.text}") from exc

        version = str(data.get("model_version", "unknown"))
        prediction_field = data.get("prediction")
        # The API returns a list of predictions even when a single record
        # is posted.  Normalise to a single float value.
        if isinstance(prediction_field, list) and prediction_field:
            pred_val = float(prediction_field[0])
        elif isinstance(prediction_field, (int, float)):
            pred_val = float(prediction_field)
        else:
            # Unexpected shape; skip this attempt
            continue
        preds_by_version[version] = pred_val
    return preds_by_version


def main() -> None:

    data_path = "data/future_unseen_examples.csv"
    df = pd.read_csv(data_path)
    # Select the first few examples to test
    examples: List[Dict[str, Any]] = df.head(5).to_dict(orient="records")

    endpoint = "http://localhost:8000/predict/"

    all_results = []
    for idx, example in enumerate(examples, start=1):
        preds_by_version = fetch_predictions(example, endpoint)
        all_results.append((example, preds_by_version))

    # Print comparison of predictions
    for i, (ex, preds) in enumerate(all_results, start=1):
        print(f"\nExample {i}:")
        for version, value in preds.items():
            print(f"  Model v{version}: {value:.2f}")
        if len(preds) == 2:
            versions = list(preds.keys())
            diff = preds[versions[0]] - preds[versions[1]]
            print(f"  Difference ({versions[0]} - {versions[1]}): {diff:.2f}")
            print(f"  Features: {ex}")
        else:
            print("  Only one model version responded within the attempt limit.")


if __name__ == "__main__":
    main()