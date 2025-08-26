"""
Configure the predictor Django application.

This AppConfig simply registers the app under the name ``predictor``.  It
doesn't need any special initialization at import time because the heavy work
(loading the model and demographic data) is handled lazily when the first
prediction request comes in.  Should you choose to load models eagerly on
startup, you can override the ``ready`` method here.
"""

from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"
