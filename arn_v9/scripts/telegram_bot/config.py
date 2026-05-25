"""Configuration and security for the ARN Telegram Bot."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path("/Users/hustle/arn-v9-repo")
DATA_DIR = Path.home() / ".arn_data"
COLLAB_DIR = DATA_DIR / "collab"
TASKS_DIR = REPO / "tasks"
CONFIG_PATH = DATA_DIR / "telegram_bot_config.json"
WHITELIST_PATH = DATA_DIR / "telegram_whitelist.json"

# ── Agents ───────────────────────────────────────────────────────────────────
VALID_AGENTS = ["kimi", "claude", "codex"]
AGENT_EMOJI = {"kimi": "🌙", "claude": "🌀", "codex": "🧪"}

# ── Bot Token ────────────────────────────────────────────────────────────────
def _load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def get_bot_token() -> str:
    """Return bot token from env var or secure config file."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        return token
    cfg = _load_config()
    token = cfg.get("bot_token", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set and not found in "
            f"{CONFIG_PATH}. Set the env var or write the config file."
        )
    return token


# ── Whitelist ────────────────────────────────────────────────────────────────
def load_whitelist() -> Set[str]:
    """Load allowed user/chat IDs from whitelist file."""
    if WHITELIST_PATH.exists():
        data = json.loads(WHITELIST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return set(str(x) for x in data.get("ids", []))
        if isinstance(data, list):
            return set(str(x) for x in data)
    return set()


def save_whitelist(ids: List[str]) -> None:
    """Save allowed user/chat IDs to whitelist file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WHITELIST_PATH.write_text(
        json.dumps({"ids": sorted(set(str(x) for x in ids))}, indent=2),
        encoding="utf-8",
    )
    os.chmod(WHITELIST_PATH, 0o600)


# ── Runner tracking ──────────────────────────────────────────────────────────
# In-memory registry of active runner processes (pid -> proc).
# Shared across modules; not persisted.
ACTIVE_RUNNERS: Dict[int, Any] = {}


def get_active_runner() -> Any | None:
    """Return the most recently spawned runner subprocess, if any."""
    if not ACTIVE_RUNNERS:
        return None
    # Return the most recently added (highest pid heuristic)
    return ACTIVE_RUNNERS[max(ACTIVE_RUNNERS.keys())]


# ── Python executable ────────────────────────────────────────────────────────
PY = Path(sys.executable)
