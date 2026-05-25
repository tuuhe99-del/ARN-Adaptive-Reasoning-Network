"""Bridge between the Telegram Bot and the ARN collaboration system."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from arn_v9.scripts.telegram_bot.config import (
    COLLAB_DIR,
    DATA_DIR,
    PY,
    REPO,
    TASKS_DIR,
    VALID_AGENTS,
)

# Import ARN collab primitives directly
from arn_v9.collab import (
    agent_health as _agent_health,
    init_collab,
    list_handoffs as _list_handoffs,
    read_state as _read_state,
    summarize_state,
    write_feed as _write_feed,
)
from arn_v9.scripts.telegram_bot import task_queue


def _safe_code(text: str) -> str:
    """Sanitize text for Telegram inline code spans by replacing backticks."""
    return str(text).replace("`", "'")


# ── State ────────────────────────────────────────────────────────────────────
def read_arn_state() -> Dict[str, Any]:
    """Read current collaboration state."""
    return _read_state(DATA_DIR)


def format_status_card(state: Dict[str, Any]) -> str:
    """Format a Telegram-friendly status card from state dict."""
    summary = summarize_state(state)
    task_id = _safe_code(summary.get("task_id") or "—")
    status = summary.get("status", "?")
    chain = summary.get("review_chain", [])
    step = summary.get("current_step", 0)
    locked = _safe_code(summary.get("locked_by") or "—")
    next_agent = _safe_code(summary.get("next_agent") or "—")
    stale = summary.get("lock_stale", False)
    updated = summary.get("updated_at", "")[:16].replace("T", " ")

    # Status emoji
    if status == "DONE":
        status_emoji = "✅ DONE"
    elif status.startswith("CLAIMED_"):
        status_emoji = f"🔒 CLAIMED ({locked})"
    elif status.startswith("HANDOFF_"):
        status_emoji = f"📤 HANDOFF ({locked})"
    else:
        status_emoji = f"⚪ {status}"

    step_lbl = f"{step}/{len(chain)}" if chain else "—"
    chain_str = " → ".join(
        f"*{a}*" if a == locked else a for a in chain
    )

    lines = [
        "🤖 *ARN Collab Status*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"*Task:*    `{task_id}`",
        f"*Status:*  {status_emoji}",
        f"*Chain:*   {chain_str}",
        f"*Step:*    {step_lbl}",
        f"*Next:*    {next_agent}",
    ]
    if stale and status.startswith("CLAIMED_"):
        lines.append(f"⚠️ Lock is *stale* (> {summary.get('stale_after_minutes', 120)} min)")
    q_len = task_queue.length()
    if q_len > 0:
        lines.append(f"📋 Queue: *{q_len}* task{'s' if q_len != 1 else ''} waiting")
    lines.append(f"_Updated: {updated}_")
    return "\n".join(lines)


# ── Task creation ────────────────────────────────────────────────────────────
def write_task_file(
    task_id: str,
    name: str,
    description: str,
    chain: List[str],
    hint: Optional[str] = None,
) -> Path:
    """Write a task Markdown file to the tasks directory."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    chain_str = " → ".join(chain)
    body = (
        f"# ARN Task: {name}\n\n"
        f"## Task ID\n`{task_id}`\n\n"
        f"## Review Chain\n```\n{chain_str}\n```\n\n"
        f"## Description\n\n{description}\n\n"
        "## Agent Instructions\n\n"
        "- Read `COLLAB.md` and `docs/collab-protocol.md` first.\n"
        "- Claim your step, do minimal correct work, write a handoff.\n"
        "- Run `python3 -m py_compile` on every Python file you change.\n\n"
        "## Verification\n```bash\n"
        "python3 -m pytest arn_v9/tests/ -x -q 2>&1 | tail -10\n"
        "```"
    )
    path = TASKS_DIR / f"{task_id}.md"
    path.write_text(body, encoding="utf-8")
    return path


def slugify_task_id(raw: str) -> str:
    """Convert raw name to ARN task ID slug."""
    slug = re.sub(r"[^a-zA-Z0-9\-]", "-", raw).strip("-")
    return f"ARN-{slug}"


# ── Runner ───────────────────────────────────────────────────────────────────
def is_runner_alive() -> bool:
    """Check if collab_runner is currently running."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "arn_v9.collab_runner"],
            capture_output=True,
            text=True,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


async def spawn_runner(task_id: str, chain: List[str]) -> asyncio.subprocess.Process:
    """Spawn the collab runner asynchronously (fire-and-forget)."""
    # Initialize collab state first
    init_collab(
        data_dir=DATA_DIR,
        task_id=task_id,
        review_chain=chain,
        force=True,
    )
    proc = await asyncio.create_subprocess_exec(
        str(PY),
        "-m",
        "arn_v9.collab_runner",
        "--repo-dir",
        str(REPO),
        "--data-dir",
        str(DATA_DIR),
        "--task-id",
        task_id,
        "--review-chain",
        ",".join(chain),
        "--force",
        "--execute",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    from arn_v9.scripts.telegram_bot.config import ACTIVE_RUNNERS

    ACTIVE_RUNNERS[proc.pid] = proc
    return proc


async def kill_active_runner() -> bool:
    """Send SIGTERM to the most recently spawned runner."""
    from arn_v9.scripts.telegram_bot.config import ACTIVE_RUNNERS, get_active_runner

    proc = get_active_runner()
    if proc is None or proc.returncode is not None:
        # Try to find via pgrep and kill
        try:
            r = subprocess.run(
                ["pgrep", "-f", "arn_v9.collab_runner"],
                capture_output=True,
                text=True,
            )
            if r.stdout.strip():
                for pid_str in r.stdout.strip().splitlines():
                    try:
                        os_pid = int(pid_str)
                        subprocess.run(["kill", "-TERM", str(os_pid)], check=False)
                    except ValueError:
                        continue
                return True
        except Exception:
            pass
        return False
    try:
        proc.send_signal(signal.SIGTERM)
        return True
    except Exception:
        return False


# ── Feeds ────────────────────────────────────────────────────────────────────
def write_feed(message: str, target: str = "all") -> Dict[str, Any]:
    """Append a feed message for an agent."""
    return _write_feed(DATA_DIR, message=message, target=target, source="telegram")


# ── Handoffs / History ───────────────────────────────────────────────────────
def list_handoffs(limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent handoff metadata."""
    return _list_handoffs(DATA_DIR, limit=limit)


def format_handoff_summary(h: Dict[str, Any]) -> str:
    """Format a single handoff entry for Telegram."""
    agent = _safe_code(h.get("agent", "?").upper())
    status = _safe_code(h.get("status", "?"))
    ts = h.get("timestamp", "")[:16].replace("T", " ")
    files = [_safe_code(f) for f in h.get("files_changed", [])]
    task_id = _safe_code(h.get("task_id", "?"))

    status_emoji = {
        "complete": "✅",
        "no_issues": "✅",
        "needs_review": "⚠️",
        "blocked": "🔴",
    }.get(status, "❓")

    lines = [
        f"{status_emoji} *{agent}*  ·  `{task_id}`",
        f"   _{ts}_  ·  status: `{status}`",
    ]
    if files:
        fstr = "  ".join(files[:5])
        if len(files) > 5:
            fstr += " …"
        lines.append(f"   Files: `{fstr}`")
    return "\n".join(lines)


# ── Agents ───────────────────────────────────────────────────────────────────
def get_agent_health() -> Dict[str, Any]:
    """Return agent binary health info."""
    return _agent_health()


def format_agent_health(health: Dict[str, Any]) -> str:
    """Format agent health for Telegram."""
    lines = ["🩺 *Agent Health*", "━━━━━━━━━━━━━━━━━━━━"]
    for name in VALID_AGENTS:
        info = health.get(name, {})
        status = _safe_code(info.get("status", "unknown"))
        exists = info.get("exists", False)
        emoji = "🟢" if info.get("status") in ("ready", "auth_ok") else "🔴"
        if info.get("status") == "auth_expired":
            emoji = "🟡"
        lines.append(f"{emoji} *{name}*  —  `{status}`")
        if "expires_in_seconds" in info:
            mins = info["expires_in_seconds"] // 60
            lines.append(f"   Token expires in {mins}m")
    return "\n".join(lines)


# ── Results ──────────────────────────────────────────────────────────────────
def read_result_file(task_id: str) -> Optional[str]:
    """Read a task result report if it exists."""
    path = TASKS_DIR / f"{task_id}-result.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# ── Logs ─────────────────────────────────────────────────────────────────────
def find_current_log(agent: str) -> Optional[Path]:
    """Return the most recent stdout log for an agent."""
    logs = COLLAB_DIR / "logs"
    if not logs.exists():
        return None
    candidates = []
    for pat in (f"*{agent}*stdout*", f"*{agent}*.log"):
        for f in logs.glob(pat):
            candidates.append(f)
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime)


def read_log_tail(agent: str, max_bytes: int = 8000) -> str:
    """Read the tail of the current agent log."""
    log = find_current_log(agent)
    if not log or not log.exists():
        return "(no log found)"
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_bytes:
            text = "…\n" + text[-max_bytes:]
        return text
    except Exception as exc:
        return f"(error reading log: {exc})"


# ── Time helpers ─────────────────────────────────────────────────────────────
def fmt_elapsed_seconds(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def elapsed_since(ts_str: Optional[str]) -> int:
    if not ts_str:
        return 0
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return 0
