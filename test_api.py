"""
Simple test script for the house price prediction API.

This script reads a few examples from ``data/future_unseen_examples.csv`` and
sends them to the prediction endpoints.  It prints the responses to the
console so you can verify that the service is working end to end.  Run this
script from within the Docker environment once the containers are up:

.. code-block:: bash

   docker-compose up --build -d
   python test_api.py

You should see a JSON response with the predicted prices.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd  # type: ignore
import requests


def main() -> None:
    # Determine where the project lives so we can find the data file
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "future_unseen_examples.csv"
    df = pd.read_csv(data_path)
    # Select the first few examples
    examples = df.head(3).to_dict(orient="records")

    # Endpoints
    base_url = os.getenv("API_BASE", "http://localhost:8000")
    endpoints = {
        "Full feature endpoint": f"{base_url}/predict/",
        "Core feature endpoint": f"{base_url}/predict_core/",
    }

    for name, url in endpoints.items():
        print(f"\n{name} ({url})")
        for i, example in enumerate(examples, start=1):
            # For the core endpoint, remove all fields except those used by the model
            if "core" in name.lower():
                keys = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement", "zipcode"]
                payload = {k: example[k] for k in keys}
            else:
                payload = example
            resp = requests.post(url, json=payload)
            try:
                resp.raise_for_status()
                print(f"Example {i}:", json.dumps(resp.json(), indent=2))
            except Exception as exc:
                print(f"Example {i}: error calling API: {exc}\nResponse: {resp.text}")


if __name__ == "__main__":
    main()