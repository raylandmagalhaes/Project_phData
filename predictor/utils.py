"""Utilities for model loading, feature engineering, and prediction."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1]


def _get_env_path(env_name: str, default: str) -> Path:
    val = os.getenv(env_name)
    return Path(val) if val else (BASE_DIR / default)

MODEL_PATH = _get_env_path("MODEL_PATH", "models/model/model.pkl")
FEATURES_PATH = _get_env_path("MODEL_FEATURES_PATH", "models/model/model_features.json")
DEMOGRAPHICS_PATH = _get_env_path("DEMOGRAPHICS_PATH", "data/zipcode_demographics.csv")

try:
    with open(MODEL_PATH, "rb") as f:
        _MODEL = pd.read_pickle(f)  # type: ignore[attr-defined]
except Exception as exc:
    _MODEL = exc  # type: ignore

try:
    with open(FEATURES_PATH, "r") as f:
        MODEL_FEATURES: List[str] = json.load(f)
except Exception as exc:
    MODEL_FEATURES = []
    _feature_error = exc

try:
    DEMOGRAPHICS_DF = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})
except Exception as exc:
    DEMOGRAPHICS_DF = pd.DataFrame()
    _demographics_error = exc

CORE_FEATURES = [
    "bedrooms","bathrooms","sqft_living","sqft_lot",
    "floors","sqft_above","sqft_basement","zipcode",
]


def add_date_features(df: pd.DataFrame, date_col: str = "date", default_tz: str = "UTC") -> pd.DataFrame:
    if date_col in df.columns:
        sale_dt = pd.to_datetime(df[date_col], format="%Y%m%dT%H%M%S", errors="coerce")
    else:
        sale_dt = pd.Series(pd.NaT, index=df.index)
    today = pd.Timestamp.now(tz=default_tz).normalize()
    sale_dt = sale_dt.fillna(today)
    df["sale_year"] = sale_dt.dt.year.astype("Int64")
    df["sale_month"] = sale_dt.dt.month.astype("Int64")
    df["sale_dayofweek"] = sale_dt.dt.dayofweek.astype("Int64")
    df["sale_quarter"] = sale_dt.dt.quarter.astype("Int64")
    df["sale_dayofyear"] = sale_dt.dt.dayofyear.astype("Int64")
    df["sale_isoweek"] = sale_dt.dt.isocalendar().week.astype("Int64")
    two_pi = 2 * np.pi
    frac = (df["sale_dayofyear"].astype(float) - 1) / 365.25
    df["sale_season_sin"] = np.sin(two_pi * frac)
    df["sale_season_cos"] = np.cos(two_pi * frac)
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
    df.drop(columns=[date_col], inplace=True, errors="ignore")
    return df


def validate_required(records: Sequence[Dict[str, Any]], required: Sequence[str]) -> None:
    for i, rec in enumerate(records):
        missing = [k for k in required if k not in rec]
        nulls = [k for k in required if k in rec and rec[k] is None]
        if missing or nulls:
            parts = []
            if missing: parts.append(f"missing={missing}")
            if nulls: parts.append(f"nulls={nulls}")
            raise ValueError(f"Record {i} invalid: " + "; ".join(parts))


def _load_model() -> Any:
    if isinstance(_MODEL, Exception):
        raise RuntimeError(f"Could not load model: {_MODEL}")
    return _MODEL


def prepare_input(records: Sequence[Dict[str, Any]], minimal: bool) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if "zipcode" not in df.columns:
        raise ValueError("Input data must include a 'zipcode' field")
    df["zipcode"] = df["zipcode"].astype(str)
    df = add_date_features(df, date_col="date")
    merged = df.merge(DEMOGRAPHICS_DF, how="left", on="zipcode")
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])
    return merged.reindex(columns=MODEL_FEATURES, fill_value=0).fillna(0)


def predict(records: Sequence[Dict[str, Any]], minimal: bool) -> list[float]:
    model = _load_model()
    input_df = prepare_input(records, minimal=minimal)
    preds = model.predict(input_df)
    return preds.tolist()
