"""Locate the SAFE-Edit repository root without hard-coded parent depths."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (default: this file) until requirements.txt + scripts/ exist."""
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "requirements.txt").is_file() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate SAFE-Edit repository root from {cur}")
