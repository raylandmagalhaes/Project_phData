"""
URL declarations for the predictor app.

The predictor exposes three endpoints:

* ``/predict/`` – full model predictions on arbitrary records.
* ``/predict_core/`` – predictions using only a minimal set of required features.
* ``/health/`` – a lightweight health‑check endpoint used by orchestration
  systems to confirm the service is running.
"""
from django.urls import path

from . import views

urlpatterns = [
    path("predict/", views.predict_view, name="predict"),
    path("predict_core/", views.predict_core_view, name="predict_core"),
    path("health/", views.health_view, name="health"),
]
