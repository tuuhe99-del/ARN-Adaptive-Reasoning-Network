"""Persistent task queue for the ARN Telegram Bot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from arn_v9.scripts.telegram_bot.config import DATA_DIR

QUEUE_PATH = DATA_DIR / "collab" / "queue.json"


def _load() -> Dict[str, Any]:
    if QUEUE_PATH.exists():
        return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    return {"tasks": []}


def _save(data: Dict[str, Any]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def enqueue(
    task_id: str,
    name: str,
    description: str,
    chain: List[str],
    hint: Optional[str] = None,
) -> int:
    """Add a task to the queue. Returns new queue length."""
    data = _load()
    data["tasks"].append(
        {
            "task_id": task_id,
            "name": name,
            "description": description,
            "chain": chain,
            "hint": hint,
        }
    )
    _save(data)
    return len(data["tasks"])


def dequeue() -> Optional[Dict[str, Any]]:
    """Pop the next task from the queue. Returns None if empty."""
    data = _load()
    if not data["tasks"]:
        return None
    task = data["tasks"].pop(0)
    _save(data)
    return task


def peek() -> Optional[Dict[str, Any]]:
    """Return the next task without removing it."""
    data = _load()
    return data["tasks"][0] if data["tasks"] else None


def list_tasks() -> List[Dict[str, Any]]:
    """Return all queued tasks."""
    return _load()["tasks"]


def remove(index: int) -> bool:
    """Remove task at 0-based index. Returns True if removed."""
    data = _load()
    if 0 <= index < len(data["tasks"]):
        data["tasks"].pop(index)
        _save(data)
        return True
    return False


def clear() -> None:
    """Empty the queue."""
    _save({"tasks": []})


def length() -> int:
    """Return number of queued tasks."""
    return len(_load()["tasks"])
