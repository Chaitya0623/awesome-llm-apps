"""Pin model providers to their defaults so a developer's .env cannot redirect the offline fakes.

Import this before any app module: the server loads .env with setdefault, so values set here win.
"""
import os

for key in ("APPRAISAL_PRICING_PROVIDER", "APPRAISAL_VISION_PROVIDER", "APPRAISAL_SKETCH_PROVIDER"):
    os.environ[key] = "gemini"
