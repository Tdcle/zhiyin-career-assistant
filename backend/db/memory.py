from __future__ import annotations

from .memory_schema import MemorySchemaMixin
from .memory_profiles import MemoryProfileMixin
from .memory_extraction import MemoryExtractionMixin


class MemoryMixin(
    MemorySchemaMixin,
    MemoryProfileMixin,
    MemoryExtractionMixin,
):
    pass
