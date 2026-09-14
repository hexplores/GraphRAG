from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class AppConfig:
    chunk_size: int = 200
    chunk_overlap: int = 40
    min_chunk_chars: int = 200
    max_keywords: int = 8
    top_k: int = 5


def load_config(path: str | Path | None) -> AppConfig:
    if not path:
        return AppConfig()

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(**data)
