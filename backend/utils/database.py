"""Compatibility entrypoint for the database layer.

The implementation lives in ``backend/db``. Keep this module so existing
imports like ``from utils.database import DatabaseManager`` continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if (__package__ or "").startswith("backend."):
    from backend.db import DatabaseManager, _make_json_safe, segment_text, segment_welfare
else:
    from db import DatabaseManager, _make_json_safe, segment_text, segment_welfare

__all__ = ["DatabaseManager", "_make_json_safe", "segment_text", "segment_welfare"]
