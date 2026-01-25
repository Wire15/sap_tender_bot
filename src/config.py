from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = repo_root()
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = REPO_ROOT / "logs"
