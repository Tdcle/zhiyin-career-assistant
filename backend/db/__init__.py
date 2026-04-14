from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if __name__ == "backend.db":
    sys.modules.setdefault("db", sys.modules[__name__])
elif __name__ == "db":
    sys.modules.setdefault("backend.db", sys.modules[__name__])

from .base import DatabaseBase
from .schema import SchemaMixin
from .users import UserMixin
from .memory import MemoryMixin
from .conversations import ConversationMixin
from .resumes import ResumeMixin
from .jobs import JobMixin
from .search import SearchMixin
from .legacy_memory import LegacyMemoryCompatMixin
from .common import _make_json_safe, segment_text, segment_welfare


class DatabaseManager(
    SchemaMixin,
    UserMixin,
    MemoryMixin,
    ConversationMixin,
    ResumeMixin,
    JobMixin,
    SearchMixin,
    LegacyMemoryCompatMixin,
    DatabaseBase,
):
    pass


__all__ = ["DatabaseManager", "_make_json_safe", "segment_text", "segment_welfare"]
