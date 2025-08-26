"""
URL declarations for the predictor app.

The predictor exposes two endpoints: `/predict/` and `/predict_core/`.  They
are registered in the project’s top‑level ``urls.py`` via ``include()``.
"""
from django.urls import path

from . import views

urlpatterns = [
    path("predict/", views.predict_view, name="predict"),
    path("predict_core/", views.predict_core_view, name="predict_core"),
]
