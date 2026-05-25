"""Watcher & Polling Engineer for the ARN Telegram Bot.

Provides two complementary update mechanisms:
1. A JobQueue poller that checks ARN state every 30 s and pushes transitions.
2. A watchdog filesystem observer that reacts to new handoff files immediately.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from telegram import Bot
from telegram.ext import Application, ContextTypes
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from arn_v9.scripts.telegram_bot.bridge import (
    _safe_code,
    format_handoff_summary,
    list_handoffs,
    read_arn_state,
    read_log_tail,
    spawn_runner,
    write_task_file,
)
from arn_v9.scripts.telegram_bot.config import COLLAB_DIR
from arn_v9.scripts.telegram_bot import task_queue
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from arn_v9 import collab

# ── Module-level registry ────────────────────────────────────────────────────
WATCHING_CHATS: Set[int] = set()

# Last known state / handoff path for the poller
_last_state: Optional[Dict[str, Any]] = None
_last_handoff_path: Optional[str] = None

# Track stale lock alerts so we don't spam (cycle_id:agent -> alerted)
_alerted_stale: Dict[str, bool] = {}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ── Chat registry ────────────────────────────────────────────────────────────
def register_watch(chat_id: int) -> None:
    WATCHING_CHATS.add(chat_id)


def unregister_watch(chat_id: int) -> None:
    WATCHING_CHATS.discard(chat_id)


def get_watching_chats() -> Set[int]:
    return set(WATCHING_CHATS)


# ── Frontmatter parser ───────────────────────────────────────────────────────
def _parse_frontmatter(text: str) -> Dict[str, Any]:
    m = _FRONTMATTER_RE.search(text)
    if not m:
        return {}
    result: Dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        result[key] = val
    return result


# ── Async helpers ────────────────────────────────────────────────────────────
async def _send_text(bot: Bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception:
        pass


async def _notify_chats(bot: Bot, chat_ids: Set[int], text: str) -> None:
    for cid in chat_ids:
        await _send_text(bot, cid, text)


def _notify_chats_sync(bot: Bot, text: str) -> None:
    chats = get_watching_chats()
    if not chats:
        return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    asyncio.run_coroutine_threadsafe(_notify_chats(bot, chats, text), loop)


def notify_chats_sync(application: Application, text: str) -> None:
    _notify_chats_sync(application.bot, text)


# ── State transition detection ───────────────────────────────────────────────
def _detect_transition(
    prev: Dict[str, Any], curr: Dict[str, Any]
) -> list[str]:
    messages: list[str] = []
    prev_status = prev.get("status", "")
    curr_status = curr.get("status", "")
    prev_locked = prev.get("locked_by")
    curr_locked = curr.get("locked_by")

    if curr_status == "DONE" and prev_status != "DONE":
        messages.append("✅ Cycle complete! Task is DONE.")
        return messages

    if prev_status == "IDLE" and curr_status.startswith("CLAIMED_"):
        messages.append(f"🔒 {curr_locked} has claimed the task.")
    elif prev_status.startswith("CLAIMED_") and curr_status.startswith("HANDOFF_"):
        agent = prev_locked or curr_locked
        messages.append(f"📤 {agent} completed their step.")
    elif prev_status.startswith("HANDOFF_") and curr_status.startswith("CLAIMED_"):
        messages.append(f"🔒 {curr_locked} has claimed the task.")
    elif (
        prev_status.startswith("CLAIMED_")
        and curr_status.startswith("CLAIMED_")
        and prev_locked != curr_locked
    ):
        messages.append(f"🔒 Agent changed to {curr_locked}.")

    return messages


# ── JobQueue callback ────────────────────────────────────────────────────────
async def state_poll_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    global _last_state, _last_handoff_path, _alerted_stale

    try:
        curr = read_arn_state()
    except Exception:
        return

    if _last_state is None:
        _last_state = curr
        _last_handoff_path = curr.get("last_handoff")
        return

    prev = _last_state
    messages = _detect_transition(prev, curr)

    curr_handoff = curr.get("last_handoff")
    new_handoff = bool(curr_handoff and curr_handoff != _last_handoff_path)
    handoff_summary: Optional[str] = None
    log_text: Optional[str] = None
    finished_agent: Optional[str] = None

    if new_handoff:
        handoffs = list_handoffs(limit=1)
        if handoffs:
            handoff_summary = format_handoff_summary(handoffs[0])
            finished_agent = handoffs[0].get("agent")
        else:
            try:
                text = Path(curr_handoff).read_text(encoding="utf-8")
                fm = _parse_frontmatter(text)
                finished_agent = fm.get("agent")
                task_id = fm.get("task_id", "?")
                status = fm.get("status", "?")
                handoff_summary = (
                    f"📝 *New handoff* by `{_safe_code(finished_agent)}` for `{_safe_code(task_id)}` — "
                    f"status: `{_safe_code(status)}`"
                )
            except Exception:
                pass

        if finished_agent:
            log_text = read_log_tail(finished_agent, 4000)

    if messages or new_handoff:
        bot = context.bot
        chats = get_watching_chats()

        for cid in chats:
            for msg in messages:
                await _send_text(bot, cid, msg)

            if handoff_summary:
                await _send_text(bot, cid, handoff_summary)

            if log_text and log_text != "(no log found)":
                excerpt = log_text[:3900]
                await _send_text(
                    bot,
                    cid,
                    f"🪵 *Log excerpt (`{_safe_code(finished_agent)}`)*\n```\n{excerpt}\n```",
                )

    # ── Stale lock alert ──────────────────────────────────────────────────────
    stale = collab.is_stale(curr)
    locked_by = curr.get("locked_by")
    cycle_id = curr.get("cycle_id", "unknown")
    stale_key = f"{cycle_id}:{locked_by}" if locked_by else ""

    if stale and locked_by and not _alerted_stale.get(stale_key):
        _alerted_stale[stale_key] = True
        stale_mins = curr.get("stale_after_minutes", 120)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🔓 Steal Lock", callback_data=f"stale:steal:{locked_by}"
                    ),
                    InlineKeyboardButton(
                        "Dismiss", callback_data=f"stale:dismiss:{locked_by}"
                    ),
                ]
            ]
        )
        alert_text = (
            f"⚠️ *Stale Lock Alert*\n"
            f"Agent `{_safe_code(locked_by)}` has been locked for > {stale_mins} minutes.\n"
            f"Task: `{_safe_code(curr.get('task_id', '?'))}`"
        )
        bot = context.bot
        for cid in get_watching_chats():
            try:
                await bot.send_message(
                    chat_id=cid,
                    text=alert_text,
                    parse_mode="Markdown",
                    reply_markup=keyboard,
                )
            except Exception:
                pass

    # Clear stale alert tracking when lock is released or changed
    if not locked_by or (stale_key and _alerted_stale.get(stale_key) and not stale):
        # Remove all alerts for this cycle when lock frees up
        keys_to_remove = [k for k in _alerted_stale if k.startswith(f"{cycle_id}:")]
        for k in keys_to_remove:
            del _alerted_stale[k]

    # ── Auto-start next queued task on DONE ───────────────────────────────────
    if curr.get("status") == "DONE" and prev.get("status") != "DONE":
        next_task = task_queue.dequeue()
        if next_task:
            try:
                write_task_file(
                    next_task["task_id"],
                    next_task["name"],
                    next_task["description"],
                    next_task["chain"],
                    next_task.get("hint"),
                )
                proc = await spawn_runner(next_task["task_id"], next_task["chain"])
                bot = context.bot
                for cid in get_watching_chats():
                    try:
                        await bot.send_message(
                            chat_id=cid,
                            text=(
                                f"✅ Cycle complete!\n"
                                f"🚀 Auto-started next queued task:\n"
                                f"`{_safe_code(next_task['task_id'])}`\n"
                                f"Chain: {' → '.join(next_task['chain'])}\n"
                                f"PID: `{proc.pid}`\n\n"
                                f"Remaining in queue: {task_queue.length()}"
                            ),
                            parse_mode="Markdown",
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger = logging.getLogger(__name__)
                logger.error("Failed to auto-start queued task: %s", exc)
                # Re-queue the task so it doesn't get lost
                task_queue.enqueue(
                    next_task["task_id"],
                    next_task["name"],
                    next_task["description"],
                    next_task["chain"],
                    next_task.get("hint"),
                )

    _last_state = curr
    _last_handoff_path = curr_handoff


# ── Job scheduler ────────────────────────────────────────────────────────────
def schedule_watcher_jobs(application: Application) -> None:
    application.job_queue.run_repeating(
        state_poll_callback,
        interval=30,
        first=10,
    )


# ── Watchdog handler ─────────────────────────────────────────────────────────
class ARNEventHandler(FileSystemEventHandler):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    def on_created(self, event: Any) -> None:
        if event.is_directory:
            return
        path = str(event.src_path)
        if not path.endswith(".md"):
            return

        time.sleep(0.5)

        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception:
            return

        fm = _parse_frontmatter(text)
        agent = fm.get("agent", "?")
        status = fm.get("status", "?")
        task_id = fm.get("task_id", "?")

        summary = (
            f"📝 *New handoff* by `{_safe_code(agent)}` for `{_safe_code(task_id)}` — "
            f"status: `{_safe_code(status)}`"
        )
        _notify_chats_sync(self.bot, summary)

        log_text = read_log_tail(agent, 4000)
        if log_text and log_text != "(no log found)":
            excerpt = log_text[:3900]
            _notify_chats_sync(
                self.bot,
                f"🪵 *Log excerpt (`{_safe_code(agent)}`)*\n```\n{excerpt}\n```",
            )


# ── Watchdog starter ─────────────────────────────────────────────────────────
def start_watchdog_observer(application: Application) -> Observer:
    handler = ARNEventHandler(bot=application.bot)
    observer = Observer()
    observer.schedule(handler, str(COLLAB_DIR / "handoffs"), recursive=False)
    observer.start()
    return observer


def stop_watchdog_observer(observer: Observer) -> None:
    observer.stop()
    observer.join(timeout=5)
