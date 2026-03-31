"""Pytest hooks: load local `.env` for optional keys (e.g. GOOGLE_API_KEY) without committing them."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root `.env` — ignored by git; used only on developer machines / CI secrets.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
