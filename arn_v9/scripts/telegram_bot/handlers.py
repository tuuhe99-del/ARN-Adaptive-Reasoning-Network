"""Telegram bot handlers for the ARN Collaboration System."""

from __future__ import annotations

import functools
import logging
from typing import Any, List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from arn_v9.scripts.telegram_bot.config import (
    ACTIVE_RUNNERS,
    AGENT_EMOJI,
    DATA_DIR,
    REPO,
    VALID_AGENTS,
    get_active_runner,
    load_whitelist,
    save_whitelist,
)
from arn_v9.scripts.telegram_bot.watcher import register_watch, unregister_watch
from arn_v9 import collab
from arn_v9.scripts.telegram_bot import task_queue
from arn_v9.scripts.telegram_bot.bridge import (
    _safe_code,
    format_agent_health,
    format_handoff_summary,
    format_status_card,
    get_agent_health,
    is_runner_alive,
    kill_active_runner,
    list_handoffs,
    read_arn_state,
    read_log_tail,
    read_result_file,
    slugify_task_id,
    spawn_runner,
    write_feed,
    write_task_file,
)

logger = logging.getLogger(__name__)

# ── Conversation states ──────────────────────────────────────────────────────

class NewTaskState:
    NAME = 0
    DESC = 1
    CHAIN = 2
    HINT = 3
    CONFIRM = 4


class HandoffState:
    AGENT = 0
    STATUS = 1
    SUMMARY = 2
    CHANGES = 3
    VERIFICATION = 4
    CONCERNS = 5
    NEXT_FOCUS = 6
    CONFIRM = 7


# ── Helpers ──────────────────────────────────────────────────────────────────

def escape_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    chars = "*_[]()~`"
    for ch in chars:
        text = text.replace(ch, "\\" + ch)
    return text


def chunk_message(text: str, limit: int = 4000) -> List[str]:
    """Split long text into chunks that fit Telegram limits."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    lines = text.splitlines(keepends=True)
    current = ""

    for line in lines:
        if len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
        elif len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks


# ── Security decorator ───────────────────────────────────────────────────────

def restricted(func):
    """Decorator that checks user/chat against the whitelist."""

    @functools.wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any
    ):
        user = update.effective_user
        chat = update.effective_chat
        if user is None:
            return

        user_id = str(user.id)
        chat_id = str(chat.id) if chat else None
        whitelist = load_whitelist()

        message = update.message or update.edited_message
        is_start = (
            message is not None
            and message.text is not None
            and message.text.strip().startswith("/start")
        )

        if not whitelist and is_start:
            new_ids = [user_id]
            if chat_id and chat_id != user_id:
                new_ids.append(chat_id)
            save_whitelist(new_ids)
            await message.reply_text(
                "🔐 You've been auto-whitelisted as the first user.",
                parse_mode="Markdown",
            )
            return await func(update, context, *args, **kwargs)

        if user_id in whitelist or chat_id in whitelist:
            return await func(update, context, *args, **kwargs)

        if message:
            await message.reply_text("⛔ Not authorized.", parse_mode="Markdown")
        return

    return wrapper


# ── Command handlers ─────────────────────────────────────────────────────────

@restricted
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user = update.effective_user
        whitelist = load_whitelist()
        auth_status = (
            "✅ Authorized" if str(user.id) in whitelist else "⛔ Not authorized"
        )

        text = (
            "🤖 *Welcome to ARN Collab Bot!*\n\n"
            "Available commands:\n"
            "/start — Show this welcome\n"
            "/status — Check ARN status\n"
            "/claim `<agent>` — Claim the active task\n"
            "/handoff — Submit a handoff for the current task\n"
            "/newtask — Create and launch a new task\n"
            "/feed `<agent> <message>` — Feed a message to an agent\n"
            "/history (limit) — View recent handoffs\n"
            "/agents — Check agent health\n"
            "/watch — Watch current agent\n"
            "/unwatch — Stop watching\n"
            "/results (task_id) — View task results\n"
            "/cancelrun — Kill active runner\n"
            "/queue <name> (chain) — Add or view task queue\n"
            "/queue_next — Start next queued task\n"
            "/queue_remove `<n>` — Remove queued task\n"
            "/queue_clear — Clear queue\n"
            "/whitelist add|remove|list (id) — Manage whitelist\n\n"
            f"_{auth_status}_"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


def _build_status_keyboard(state: dict) -> InlineKeyboardMarkup:
    """Build inline keyboard for status messages."""
    buttons = [
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="status:refresh"),
            InlineKeyboardButton("🛑 Kill", callback_data="runner:kill"),
            InlineKeyboardButton("📄 Results", callback_data="results:show"),
        ],
        [
            InlineKeyboardButton("👁️ Watch", callback_data="watch:start"),
            InlineKeyboardButton("📜 History", callback_data="history:show"),
            InlineKeyboardButton("🤖 Agents", callback_data="agents:show"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


@restricted
async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        state = read_arn_state()
        card = format_status_card(state)
        if is_runner_alive():
            card += "\n🟢 Runner active"
        else:
            card += "\n⚫ Runner idle"
        keyboard = _build_status_keyboard(state)
        await update.message.reply_text(card, parse_mode="Markdown", reply_markup=keyboard)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /claim ───────────────────────────────────────────────────────────────────

@restricted
async def cmd_claim(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        args = context.args
        if not args:
            await update.message.reply_text(
                "Usage: `/claim <agent>`\nAgents: `kimi`, `claude`, `codex`",
                parse_mode="Markdown",
            )
            return

        agent = args[0].lower().strip()
        if agent not in VALID_AGENTS:
            await update.message.reply_text(
                f"⛔ Invalid agent: `{_safe_code(agent)}`. "
                "Use `kimi`, `claude`, or `codex`.",
                parse_mode="Markdown",
            )
            return

        state = read_arn_state()
        status = state.get("status", "")
        task_id = state.get("task_id")

        if status == "DONE" or not task_id:
            await update.message.reply_text(
                "No active task to claim.", parse_mode="Markdown"
            )
            return

        if str(status).startswith("CLAIMED_") and not collab.is_stale(state):
            locked_by = state.get("locked_by", "unknown")
            await update.message.reply_text(
                f"Task is already claimed by {escape_markdown(locked_by)}.",
                parse_mode="Markdown",
            )
            return

        new_state = collab.claim_task(DATA_DIR, agent)
        card = format_status_card(new_state)
        await update.message.reply_text(
            f"🔒 {agent} has claimed the task.\n{card}",
            parse_mode="Markdown",
        )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /handoff conversation ────────────────────────────────────────────────────

@restricted
async def handoff_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        context.user_data.clear()
        await update.message.reply_text(
            "Which agent is handing off? (kimi / claude / codex)",
            parse_mode="Markdown",
        )
        return HandoffState.AGENT
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_agent(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        agent = update.message.text.strip().lower()
        if agent not in VALID_AGENTS:
            await update.message.reply_text(
                f"Invalid agent: `{_safe_code(agent)}`. Try again:",
                parse_mode="Markdown",
            )
            return HandoffState.AGENT
        context.user_data["handoff_agent"] = agent
        await update.message.reply_text(
            "Status? (complete / blocked / needs_review / no_issues)",
            parse_mode="Markdown",
        )
        return HandoffState.STATUS
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_status(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        status = update.message.text.strip().lower()
        valid_statuses = {"complete", "blocked", "needs_review", "no_issues"}
        if status not in valid_statuses:
            await update.message.reply_text(
                f"Invalid status: `{_safe_code(status)}`. Try again:",
                parse_mode="Markdown",
            )
            return HandoffState.STATUS
        context.user_data["handoff_status"] = status
        await update.message.reply_text(
            "Task summary (one line):", parse_mode="Markdown"
        )
        return HandoffState.SUMMARY
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_summary(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        summary = update.message.text.strip()
        context.user_data["handoff_summary"] = summary
        await update.message.reply_text(
            "What changes were made? (describe briefly):", parse_mode="Markdown"
        )
        return HandoffState.CHANGES
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_changes(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        changes = update.message.text.strip()
        context.user_data["handoff_changes"] = changes
        await update.message.reply_text(
            "Verification steps or results:", parse_mode="Markdown"
        )
        return HandoffState.VERIFICATION
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_verification(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        verification = update.message.text.strip()
        context.user_data["handoff_verification"] = verification
        await update.message.reply_text(
            "Any concerns? Send /skip for none.", parse_mode="Markdown"
        )
        return HandoffState.CONCERNS
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_concerns(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        concerns = update.message.text.strip()
        context.user_data["handoff_concerns"] = concerns
        await update.message.reply_text(
            "Next agent focus? Send /skip for none.", parse_mode="Markdown"
        )
        return HandoffState.NEXT_FOCUS
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def skip_concerns(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        context.user_data["handoff_concerns"] = "None"
        await update.message.reply_text(
            "Next agent focus? Send /skip for none.", parse_mode="Markdown"
        )
        return HandoffState.NEXT_FOCUS
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def handoff_next_focus(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        next_focus = update.message.text.strip()
        context.user_data["handoff_next_focus"] = next_focus
        return await _ask_handoff_confirm(update, context)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def skip_next_focus(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        context.user_data["handoff_next_focus"] = "None"
        return await _ask_handoff_confirm(update, context)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def _ask_handoff_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    agent = context.user_data["handoff_agent"]
    status = context.user_data["handoff_status"]
    summary = context.user_data["handoff_summary"]
    changes = context.user_data["handoff_changes"]
    verification = context.user_data["handoff_verification"]
    concerns = context.user_data.get("handoff_concerns", "None")
    next_focus = context.user_data.get("handoff_next_focus", "None")

    text = (
        f"📋 *Handoff Summary*\n"
        f"Agent: `{_safe_code(agent)}`\n"
        f"Status: `{_safe_code(status)}`\n"
        f"Summary: `{_safe_code(summary)}`\n"
        f"Changes: `{_safe_code(changes)}`\n"
        f"Verification: `{_safe_code(verification)}`\n"
        f"Concerns: `{_safe_code(concerns)}`\n"
        f"Next Focus: `{_safe_code(next_focus)}`\n\n"
        f"Submit handoff? (y/n)"
    )
    await update.message.reply_text(text, parse_mode="Markdown")
    return HandoffState.CONFIRM


async def handoff_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        text = update.message.text.strip().lower()
        if text not in ("y", "yes"):
            await update.message.reply_text("❎ Cancelled.", parse_mode="Markdown")
            context.user_data.clear()
            return ConversationHandler.END

        agent = context.user_data["handoff_agent"]
        status = context.user_data["handoff_status"]
        summary = context.user_data["handoff_summary"]
        changes = context.user_data["handoff_changes"]
        verification = context.user_data["handoff_verification"]
        concerns = context.user_data.get("handoff_concerns", "None")
        next_focus = context.user_data.get("handoff_next_focus", "None")

        state = read_arn_state()
        locked_by = state.get("locked_by")
        if locked_by and locked_by != agent:
            await update.message.reply_text(
                f"⚠️ Warning: task is locked by `{_safe_code(locked_by)}`, "
                f"but you are handing off as `{_safe_code(agent)}`. Proceeding anyway.",
                parse_mode="Markdown",
            )

        path, validation = collab.create_handoff(
            data_dir=DATA_DIR,
            agent=agent,
            status=status,
            task_summary=summary,
            changes=changes,
            verification=verification,
            concerns=concerns or "None",
            next_focus=next_focus or "None",
            repo_dir=REPO,
        )

        if not validation.get("valid"):
            err = validation.get("error", "unknown validation error")
            await update.message.reply_text(
                f"❌ Handoff validation failed: {escape_markdown(err)}",
                parse_mode="Markdown",
            )
            context.user_data.clear()
            return ConversationHandler.END

        new_state = read_arn_state()
        card = format_status_card(new_state)
        await update.message.reply_text(
            f"✅ Handoff submitted.\n{card}",
            parse_mode="Markdown",
        )
        context.user_data.clear()
        return ConversationHandler.END
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        context.user_data.clear()
        return ConversationHandler.END


# ── /newtask conversation ────────────────────────────────────────────────────

@restricted
async def newtask_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        context.user_data.clear()
        await update.message.reply_text(
            "What should we call this task? (e.g. fix-recall-bug)",
            parse_mode="Markdown",
        )
        return NewTaskState.NAME
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def newtask_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        name = update.message.text.strip()
        if not name:
            await update.message.reply_text(
                "Name cannot be empty. Try again:", parse_mode="Markdown"
            )
            return NewTaskState.NAME
        context.user_data["newtask_name"] = name
        await update.message.reply_text(
            "Describe the task. Send /skip for empty description.",
            parse_mode="Markdown",
        )
        return NewTaskState.DESC
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def newtask_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        desc = update.message.text.strip()
        context.user_data["newtask_desc"] = desc
        return await _ask_chain(update, context)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def skip_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["newtask_desc"] = ""
        return await _ask_chain(update, context)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def _ask_chain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lines = [f"{i + 1}. {VALID_AGENTS[i]}" for i in range(len(VALID_AGENTS))]
    emojis = [AGENT_EMOJI.get(a, "") for a in VALID_AGENTS]
    for i in range(len(lines)):
        if emojis[i]:
            lines[i] = f"{emojis[i]} {lines[i]}"
    await update.message.reply_text(
        "Review chain agents:\n"
        + "\n".join(lines)
        + "\n\nEnter numbers in order, e.g. `1 3` or `1 2 3`",
        parse_mode="Markdown",
    )
    return NewTaskState.CHAIN


async def newtask_chain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text.strip()
        nums = text.split()
        seen = set()
        chain = []

        for n in nums:
            try:
                idx = int(n) - 1
                if idx < 0 or idx >= len(VALID_AGENTS):
                    raise ValueError
                if idx in seen:
                    await update.message.reply_text(
                        "No duplicates allowed. Try again:", parse_mode="Markdown"
                    )
                    return NewTaskState.CHAIN
                seen.add(idx)
                chain.append(VALID_AGENTS[idx])
            except ValueError:
                await update.message.reply_text(
                    f"Invalid number: `{_safe_code(n)}`. "
                    f"Use 1-{len(VALID_AGENTS)}. Try again:",
                    parse_mode="Markdown",
                )
                return NewTaskState.CHAIN

        if not chain:
            await update.message.reply_text(
                "Please select at least one agent. Try again:", parse_mode="Markdown"
            )
            return NewTaskState.CHAIN

        context.user_data["newtask_chain"] = chain
        await update.message.reply_text(
            "Any hint for the agents? Send /skip to skip.", parse_mode="Markdown"
        )
        return NewTaskState.HINT
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def newtask_hint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        hint = update.message.text.strip()
        context.user_data["newtask_hint"] = hint
        return await _ask_confirm(update, context)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def skip_hint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["newtask_hint"] = None
        return await _ask_confirm(update, context)
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


async def _ask_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    name = context.user_data["newtask_name"]
    chain = context.user_data["newtask_chain"]
    hint = context.user_data.get("newtask_hint")
    hint_str = (
        f"Hint: `{_safe_code(hint)}`" if hint else "Hint: _(none)_"
    )
    chain_emojis = " → ".join(
        f"{AGENT_EMOJI.get(a, '')} {a}" for a in chain
    )
    text = (
        f"📋 *Task Summary*\n"
        f"Name: `{_safe_code(name)}`\n"
        f"Chain: {chain_emojis}\n"
        f"{hint_str}\n\n"
        f"Launch? (y/n)"
    )
    await update.message.reply_text(text, parse_mode="Markdown")
    return NewTaskState.CONFIRM


async def newtask_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        text = update.message.text.strip().lower()
        if text not in ("y", "yes"):
            await update.message.reply_text(
                "❎ Cancelled.", parse_mode="Markdown"
            )
            context.user_data.clear()
            return ConversationHandler.END

        name = context.user_data["newtask_name"]
        desc = context.user_data.get("newtask_desc", "")
        chain = context.user_data["newtask_chain"]
        hint = context.user_data.get("newtask_hint")

        task_id = slugify_task_id(name)
        write_task_file(task_id, name, desc, chain, hint)

        pid = None
        if not is_runner_alive():
            proc = await spawn_runner(task_id, chain)
            pid = proc.pid

        if pid:
            await update.message.reply_text(
                f"✅ Task created: `{task_id}`\n"
                f"🚀 Runner spawned (PID: `{pid}`)",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"✅ Task created: `{task_id}`\n"
                f"⚫ Runner already active — task queued.",
                parse_mode="Markdown",
            )

        context.user_data.clear()
        return ConversationHandler.END
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        context.user_data.clear()
        return ConversationHandler.END


async def cancel_conversation(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        context.user_data.clear()
        await update.message.reply_text(
            "❎ Cancelled.", parse_mode="Markdown"
        )
        return ConversationHandler.END
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )
        return ConversationHandler.END


# ── /feed ────────────────────────────────────────────────────────────────────

@restricted
async def feed_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: `/feed <agent> <message>`\n"
                "Agents: `all`, `kimi`, `claude`, `codex`",
                parse_mode="Markdown",
            )
            return

        agent = args[0].lower()
        if agent not in ("all", "kimi", "claude", "codex"):
            await update.message.reply_text(
                f"⛔ Invalid agent: `{_safe_code(agent)}`. "
                "Use `all`, `kimi`, `claude`, or `codex`.",
                parse_mode="Markdown",
            )
            return

        message = " ".join(args[1:])
        result = write_feed(message, target=agent)
        status = result.get("status", "ok")
        await update.message.reply_text(
            f"✅ Feed sent to `{agent}` (status: `{status}`)",
            parse_mode="Markdown",
        )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /history ─────────────────────────────────────────────────────────────────

@restricted
async def history_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        limit = 5
        if context.args:
            try:
                limit = int(context.args[0])
            except ValueError:
                await update.message.reply_text(
                    "Invalid limit. Using default 5.", parse_mode="Markdown"
                )
                limit = 5
        limit = max(1, min(limit, 15))

        handoffs = list_handoffs(limit)
        if not handoffs:
            await update.message.reply_text(
                "No handoffs yet.", parse_mode="Markdown"
            )
            return

        summaries = [format_handoff_summary(h) for h in handoffs]
        full_text = "\n\n".join(summaries)
        chunks = chunk_message(full_text)
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode="Markdown")
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /agents ──────────────────────────────────────────────────────────────────

@restricted
async def agents_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        health = get_agent_health()
        text = format_agent_health(health)
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /watch ───────────────────────────────────────────────────────────────────

@restricted
async def watch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        state = read_arn_state()
        agent = state.get("locked_by") or state.get("next_agent") or "system"
        context.chat_data["watching"] = True
        await update.message.reply_text(
            f"👁️ Watching {agent}… I'll update on state changes.",
            parse_mode="Markdown",
        )
        log_tail = read_log_tail(agent, 3000)
        if log_tail and log_tail != "(no log found)":
            escaped = escape_markdown(log_tail)
            chunks = chunk_message(f"📄 *Last log tail:*\n{escaped}", limit=4000)
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode="Markdown")
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /unwatch ─────────────────────────────────────────────────────────────────

@restricted
async def unwatch_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        unregister_watch(update.effective_chat.id)
        await update.message.reply_text(
            "🛑 Stopped watching.", parse_mode="Markdown"
        )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /results ─────────────────────────────────────────────────────────────────

@restricted
async def results_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        task_id = None
        if context.args:
            task_id = context.args[0].strip()
        else:
            state = read_arn_state()
            task_id = state.get("task_id")

        if not task_id:
            await update.message.reply_text(
                "No task ID found.", parse_mode="Markdown"
            )
            return

        result = read_result_file(task_id)
        if result is None:
            await update.message.reply_text(
                "No result report found.", parse_mode="Markdown"
            )
            return

        chunks = chunk_message(result)
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode="Markdown")
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /cancelrun ───────────────────────────────────────────────────────────────

@restricted
async def cancelrun_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        success = await kill_active_runner()
        ACTIVE_RUNNERS.clear()
        if success:
            await update.message.reply_text(
                "🛑 Active runner killed.", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                "⚫ No active runner to kill.", parse_mode="Markdown"
            )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── /whitelist ───────────────────────────────────────────────────────────────

@restricted
async def whitelist_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        args = context.args
        if not args or args[0].lower() == "list":
            ids = load_whitelist()
            text = "📋 Whitelist:\n" + (
                "\n".join(f"• `{i}`" for i in sorted(ids)) or "_(empty)_"
            )
            await update.message.reply_text(text, parse_mode="Markdown")
            return

        action = args[0].lower()
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: `/whitelist add|remove <id>`", parse_mode="Markdown"
            )
            return

        id_str = args[1].strip()
        ids = set(load_whitelist())

        if action == "add":
            ids.add(id_str)
            save_whitelist(list(ids))
            await update.message.reply_text(
                f"✅ Added `{id_str}` to whitelist.", parse_mode="Markdown"
            )
        elif action == "remove":
            ids.discard(id_str)
            save_whitelist(list(ids))
            await update.message.reply_text(
                f"❎ Removed `{id_str}` from whitelist.", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                "Usage: `/whitelist add|remove|list (id)`",
                parse_mode="Markdown",
            )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── Global error handler ─────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Update %s caused error %s", update, context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(f"❌ Unexpected error: {context.error}")


# ── Queue commands ───────────────────────────────────────────────────────────

@restricted
async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args:
        tasks = task_queue.list_tasks()
        if not tasks:
            await update.message.reply_text(
                "📭 Queue is empty.\nUsage: `/queue <task-name> (chain)`\nExample: `/queue fix-bug codex,claude`",
                parse_mode="Markdown",
            )
            return
        lines = ["📋 *Task Queue*", ""]
        for i, t in enumerate(tasks, 1):
            chain_str = " → ".join(t.get("chain", []))
            lines.append(f"{i}. `{t['task_id']}` — {chain_str}")
        lines.append("")
        lines.append(f"_Total: {len(tasks)} tasks queued_")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    # Quick-add mode
    name = args[0]
    task_id = slugify_task_id(name)
    chain = VALID_AGENTS.copy()
    if len(args) > 1:
        try:
            chain = collab.sanitize_review_chain(args[1])
        except ValueError as exc:
            await update.message.reply_text(f"❌ Invalid chain: {exc}")
            return

    try:
        write_task_file(task_id, name, "(queued via Telegram)", chain, None)
        task_queue.enqueue(task_id, name, "(queued via Telegram)", chain)
        await update.message.reply_text(
            f"✅ Queued `{task_id}`\nChain: {' → '.join(chain)}\n\nQueue size: {task_queue.length()}",
            parse_mode="Markdown",
        )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Error: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


@restricted
async def cmd_queue_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: `/queue_remove <number>`")
        return
    idx = int(args[0]) - 1
    if task_queue.remove(idx):
        await update.message.reply_text(f"🗑 Removed task #{idx + 1}. Queue: {task_queue.length()}")
    else:
        await update.message.reply_text("❌ Invalid queue number.")


@restricted
async def cmd_queue_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    task_queue.clear()
    await update.message.reply_text("🧹 Queue cleared.")


@restricted
async def cmd_queue_next(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    next_task = task_queue.dequeue()
    if not next_task:
        await update.message.reply_text("📭 Queue is empty.")
        return
    if is_runner_alive():
        await update.message.reply_text(
            f"⚠️ Runner already active. Re-queued `{next_task['task_id']}`.",
            parse_mode="Markdown",
        )
        task_queue.enqueue(
            next_task["task_id"],
            next_task["name"],
            next_task["description"],
            next_task["chain"],
            next_task.get("hint"),
        )
        return
    try:
        proc = await spawn_runner(next_task["task_id"], next_task["chain"])
        await update.message.reply_text(
            f"🚀 Started `{next_task['task_id']}` from queue!\n"
            f"Chain: {' → '.join(next_task['chain'])}\n"
            f"PID: `{proc.pid}`\n\nRemaining: {task_queue.length()}",
            parse_mode="Markdown",
        )
    except Exception as exc:
        await update.message.reply_text(
            f"❌ Failed to start: {escape_markdown(str(exc))}", parse_mode="Markdown"
        )


# ── Inline callback router ───────────────────────────────────────────────────

async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat.id

    try:
        if data == "status:refresh":
            state = read_arn_state()
            card = format_status_card(state)
            if is_runner_alive():
                card += "\n🟢 Runner active"
            else:
                card += "\n⚫ Runner idle"
            keyboard = _build_status_keyboard(state)
            await query.edit_message_text(card, parse_mode="Markdown", reply_markup=keyboard)

        elif data == "runner:kill":
            ok = await kill_active_runner()
            text = "🔴 Runner cancelled." if ok else "⚫ No active runner."
            await context.bot.send_message(chat_id=chat_id, text=text)

        elif data == "results:show":
            state = read_arn_state()
            task_id = state.get("task_id")
            if not task_id:
                await context.bot.send_message(chat_id=chat_id, text="No active task.")
                return
            result = read_result_file(task_id)
            if result is None:
                await context.bot.send_message(chat_id=chat_id, text=f"📭 No result for `{task_id}`.", parse_mode="Markdown")
                return
            header = f"📄 *Result for `{task_id}`*\n"
            for chunk in _chunk_text(header + result, 4000):
                await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")

        elif data == "watch:start":
            register_watch(chat_id)
            state = read_arn_state()
            agent = state.get("locked_by") or (state.get("review_chain") or ["?"])[0]
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"👁️ Watching *{agent}*…",
                parse_mode="Markdown",
            )

        elif data == "history:show":
            handoffs = list_handoffs(limit=5)
            if not handoffs:
                await context.bot.send_message(chat_id=chat_id, text="📭 No handoffs yet.")
                return
            lines = [f"📜 *Last {len(handoffs)} handoffs*", ""]
            for h in handoffs:
                lines.append(format_handoff_summary(h))
                lines.append("")
            for chunk in _chunk_text("\n".join(lines), 4000):
                await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")

        elif data == "agents:show":
            health = get_agent_health()
            text = format_agent_health(health)
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")

        elif data.startswith("stale:steal:"):
            agent = data.split(":", 2)[2]
            try:
                collab.claim_task(DATA_DIR, agent, steal_stale=True)
                state = read_arn_state()
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🔓 Stole lock for *{agent}*.\n" + format_status_card(state),
                    parse_mode="Markdown",
                )
            except Exception as exc:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Could not steal lock: {escape_markdown(str(exc))}",
                    parse_mode="Markdown",
                )

        elif data.startswith("stale:dismiss:"):
            agent = data.split(":", 2)[2]
            await query.edit_message_text(
                f"Dismissed stale alert for `{agent}`.", parse_mode="Markdown"
            )

        else:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Unknown action: {data}")
    except Exception as exc:
        logger.error("Callback error: %s", exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Error: {escape_markdown(str(exc))}",
            parse_mode="Markdown",
        )


# ── Registration ─────────────────────────────────────────────────────────────

def register_handlers(application: Application) -> None:
    """Register all handlers on the PTB Application."""
    newtask_conv = ConversationHandler(
        entry_points=[CommandHandler("newtask", newtask_entry)],
        states={
            NewTaskState.NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, newtask_name)
            ],
            NewTaskState.DESC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, newtask_desc),
                CommandHandler("skip", skip_desc),
            ],
            NewTaskState.CHAIN: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, newtask_chain)
            ],
            NewTaskState.HINT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, newtask_hint),
                CommandHandler("skip", skip_hint),
            ],
            NewTaskState.CONFIRM: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, newtask_confirm)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
    )

    handoff_conv = ConversationHandler(
        entry_points=[CommandHandler("handoff", handoff_entry)],
        states={
            HandoffState.AGENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_agent)
            ],
            HandoffState.STATUS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_status)
            ],
            HandoffState.SUMMARY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_summary)
            ],
            HandoffState.CHANGES: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_changes)
            ],
            HandoffState.VERIFICATION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_verification)
            ],
            HandoffState.CONCERNS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_concerns),
                CommandHandler("skip", skip_concerns),
            ],
            HandoffState.NEXT_FOCUS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_next_focus),
                CommandHandler("skip", skip_next_focus),
            ],
            HandoffState.CONFIRM: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handoff_confirm)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
    )

    handlers = [
        CommandHandler("start", start_handler),
        CommandHandler("status", status_handler),
        CommandHandler("claim", cmd_claim),
        handoff_conv,
        newtask_conv,
        CommandHandler("feed", feed_handler),
        CommandHandler("history", history_handler),
        CommandHandler("agents", agents_handler),
        CommandHandler("watch", watch_handler),
        CommandHandler("unwatch", unwatch_handler),
        CommandHandler("results", results_handler),
        CommandHandler("cancelrun", cancelrun_handler),
        CommandHandler("whitelist", whitelist_handler),
        CommandHandler("queue", cmd_queue),
        CommandHandler("queue_remove", cmd_queue_remove),
        CommandHandler("queue_clear", cmd_queue_clear),
        CommandHandler("queue_next", cmd_queue_next),
        CallbackQueryHandler(callback_router),
    ]

    for handler in handlers:
        application.add_handler(handler)

    application.add_error_handler(error_handler)
