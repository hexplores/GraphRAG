from __future__ import annotations

import os

__all__ = [
    "api",
    "chunking",
    "cli",
    "config",
    "embeddings",
    "graph",
    "ingest",
    "pipeline",
    "retrieval",
]

_package_root = os.path.dirname(__file__)
_legacy_root = os.path.abspath(os.path.join(_package_root, "..", "graphrag_smart_retrival"))
__path__ = [
    os.path.abspath(_package_root),
    os.path.abspath(_legacy_root),
    *list(__path__),
]
