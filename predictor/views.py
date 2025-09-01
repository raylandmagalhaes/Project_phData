"""
Views for serving machine‑learning predictions.
"""
from __future__ import annotations

import json, os
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from . import utils

CORE_FEATURES = utils.CORE_FEATURES

def _get_model_version() -> str:
    return os.getenv("MODEL_VERSION", "unknown")

@csrf_exempt
def predict_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
        records = [data] if isinstance(data, dict) else (data if isinstance(data, list) else None)
        if records is None:
            raise ValueError("Invalid JSON payload: must be an object or list of objects")
        prediction = utils.predict(records, minimal=False)
        return JsonResponse({
            "model_version": _get_model_version(),
            "features": records,
            "prediction": prediction,
        })
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)

@csrf_exempt
def predict_core_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
        records = [data] if isinstance(data, dict) else (data if isinstance(data, list) else None)
        if records is None:
            raise ValueError("Invalid JSON payload: must be an object or list of objects")
        utils.validate_required(records, CORE_FEATURES)
        prediction = utils.predict(records, minimal=True)
        return JsonResponse({
            "model_version": _get_model_version(),
            "features": records,
            "prediction": prediction,
        })
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
