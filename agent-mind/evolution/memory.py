"""Utilities for persisting AIwake evolution memory data.

This module intentionally depends only on Python's standard library.  All write
helpers are best-effort: failures are swallowed and reported as ``None`` so
callers can use them in non-critical evolution paths without breaking runtime
flows.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EVOLUTION_DIR = Path("/app/data/evolution")
EVOLUTION_DIR = Path(os.environ.get("EVOLUTION_DIR") or DEFAULT_EVOLUTION_DIR)

_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_MULTI_SEP_RE = re.compile(r"[-_]{2,}")


def safe_slug(text: str) -> str:
    """Return a filesystem-safe slug for *text*.

    The slug keeps ASCII letters, digits, dots, underscores, and hyphens;
    other characters are normalized to ``-``. Empty results become ``item``.
    """

    value = str(text or "").strip().lower()
    value = _SAFE_SLUG_RE.sub("-", value)
    value = _MULTI_SEP_RE.sub("-", value).strip("._-")
    return value or "item"


def _ensure_evolution_dir() -> bool:
    try:
        EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def _path_for(name: str, suffix: str) -> Path:
    slug = safe_slug(name)
    if not slug.endswith(suffix):
        slug = f"{slug}{suffix}"
    return EVOLUTION_DIR / slug


def append_jsonl(name: str, item: Any) -> Path | None:
    """Append *item* as one JSON line under the evolution directory.

    Returns the written file path on success, otherwise ``None``.
    """

    if not _ensure_evolution_dir():
        return None

    path = _path_for(name, ".jsonl")
    try:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "item": item,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str))
            handle.write("\n")
        return path
    except (OSError, TypeError, ValueError):
        return None


def write_report(name: str, content: str) -> Path | None:
    """Write a UTF-8 text report under the evolution directory.

    Returns the written file path on success, otherwise ``None``.
    """

    if not _ensure_evolution_dir():
        return None

    path = _path_for(name, ".md")
    try:
        path.write_text(str(content), encoding="utf-8")
        return path
    except OSError:
        return None


def append_growth_log(item: dict[str, Any]) -> Path | None:
    """Append an audit-first growth/self-modification record.

    Growth logs are append-only JSONL records under the evolution directory. They
    are safe to expose because all payloads are secret-redacted before writing.
    Expected fields include purpose, change_summary, files, validation_result,
    and risk_note; callers may include extra context such as external sources.
    """

    return append_jsonl("growth_log", item)


def read_recent_jsonl(name: str, limit: int = 50) -> list[Any]:
    """Read up to *limit* recent JSONL entries from *name*.

    Missing files, invalid limits, unreadable files, and malformed lines are
    handled safely. Malformed lines are skipped.
    """

    try:
        count = max(0, int(limit))
    except (TypeError, ValueError):
        count = 50

    if count == 0:
        return []

    path = _path_for(name, ".jsonl")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    items: list[Any] = []
    for line in lines[-count:]:
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items
