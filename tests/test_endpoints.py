"""
Automated API tests for the house price prediction service.

These tests use Django's test client to make requests against the API
endpoints defined in the ``predictor`` app.  They verify correct
behaviour for valid requests, input schema validation, and error handling.
Run these tests from the project root using ``pytest``:

.. code-block:: bash

    pytest -q

"""
import json
import os
from pathlib import Path

import pandas as pd  # type: ignore
import pytest

# Set up Django environment before importing Client.  pytest will only
# evaluate this once at module import time.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Project_phData.settings")

import django  # noqa: E402  # isort:skip
# django.setup()  # noqa: E402  # isort:skip

from django.test import Client  # noqa: E402  # isort:skip
from django.conf import settings  # noqa: E402  # isort:skip


@pytest.fixture()
def client() -> Client:
    """Return a Django test client instance."""
    return Client()


def get_example(index: int = 0) -> dict:
    """Return the ``index``‑th record from the future_unseen_examples.csv file as a dict."""
    data_path = Path(settings.BASE_DIR) / "data" / "future_unseen_examples.csv"
    df = pd.read_csv(data_path)
    record = df.iloc[index].to_dict()
    # Ensure zipcode is a string as expected by the service
    if "zipcode" in record:
        record["zipcode"] = str(record["zipcode"])
    return record


def test_predict_valid_single_record(client: Client) -> None:
    """POSTing a single valid record to /predict/ should return a 200 and a prediction list."""
    record = get_example(0)
    resp = client.post(
        "/predict/",
        data=json.dumps(record),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content.decode())
    assert "model_version" in data, "Response should include model_version"
    assert "features" in data, "Response should include features"
    assert "prediction" in data, "Response should include prediction"
    assert isinstance(data["prediction"], list), "prediction should be a list"
    assert len(data["prediction"]) == 1, "prediction list length should match number of records"


def test_predict_valid_multiple_records(client: Client) -> None:
    """POSTing a list of records to /predict/ should return a prediction list of equal length."""
    records = [get_example(0), get_example(1)]
    resp = client.post(
        "/predict/",
        data=json.dumps(records),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content.decode())
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == len(records), "prediction list length should match input list"


def test_predict_get_not_allowed(client: Client) -> None:
    """GET requests to /predict/ should return HTTP 405 (method not allowed)."""
    resp = client.get("/predict/")
    assert resp.status_code == 405


def test_predict_invalid_json_payload(client: Client) -> None:
    """Sending invalid JSON should return HTTP 400 with an error message."""
    resp = client.post("/predict/", data="this is not json", content_type="application/json")
    assert resp.status_code == 400
    data = json.loads(resp.content.decode())
    assert "error" in data


def test_predict_missing_zipcode(client: Client) -> None:
    """A record missing the zipcode field should result in a 400 response."""
    record = get_example(0)
    record.pop("zipcode", None)
    resp = client.post("/predict/", data=json.dumps(record), content_type="application/json")
    assert resp.status_code == 400
    data = json.loads(resp.content.decode())
    assert "error" in data


def test_predict_core_valid(client: Client) -> None:
    """POSTing minimal required fields to /predict_core/ should succeed."""
    record_full = get_example(0)
    # Minimal core fields used by the base model
    core_keys = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "zipcode",
    ]
    minimal = {k: record_full[k] for k in core_keys}
    resp = client.post(
        "/predict_core/",
        data=json.dumps(minimal),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content.decode())
    assert "model_version" in data
    assert "prediction" in data
    assert len(data["prediction"]) == 1


def test_predict_core_missing_required_field(client: Client) -> None:
    """Missing one of the core fields should result in a 400 response."""
    record_full = get_example(0)
    core_keys = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "zipcode",
    ]
    minimal = {k: record_full[k] for k in core_keys}
    # Remove a required field
    minimal.pop("bedrooms")
    resp = client.post(
        "/predict_core/",
        data=json.dumps(minimal),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = json.loads(resp.content.decode())
    assert "error" in data
