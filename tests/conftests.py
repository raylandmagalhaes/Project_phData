# tests/conftest.py
import os
import sys
from pathlib import Path
import django

# Add project root (folder that contains manage.py) to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Point Django at the settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Project_phData.settings")

# Boot Django so settings are available during collection
django.setup()
