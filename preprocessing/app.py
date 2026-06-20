"""Standalone Preprocessing Service."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from services.preprocessing.router import router

app = FastAPI(title="Preprocessing Service", version="1.0.0")
app.include_router(router)
