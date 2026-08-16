"""visionrag -- CPU-first structured video memory.

Public surface is deliberately small: build a Config, hand it to an
IngestPipeline with a MemoryStore, and read events back out.
"""

from .config import Config
from .memory.store import MemoryStore
from .pipeline import IngestPipeline
from .types import SCHEMA_VERSION, Event, EventType, Track

__version__ = "0.1.0"

__all__ = [
    "Config",
    "MemoryStore",
    "IngestPipeline",
    "Event",
    "EventType",
    "Track",
    "SCHEMA_VERSION",
    "__version__",
]
