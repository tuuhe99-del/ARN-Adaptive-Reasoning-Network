#!/usr/bin/env python3
"""ARN Collaboration System — Telegram Bot main entry point."""

import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path

from telegram import BotCommand
from telegram.ext import Application, ApplicationBuilder

from .config import get_bot_token
from .handlers import register_handlers
from .watcher import schedule_watcher_jobs, start_watchdog_observer, stop_watchdog_observer


def _setup_logging() -> None:
    log_dir = Path.home() / ".arn_data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "telegram_bot.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


_BOT_COMMANDS = [
    BotCommand("start", "Show welcome and available commands"),
    BotCommand("status", "Check current ARN cycle status"),
    BotCommand("claim", "Manually claim a task step for an agent"),
    BotCommand("handoff", "Submit a handoff to advance the cycle"),
    BotCommand("newtask", "Create and launch a new collaboration task"),
    BotCommand("feed", "Inject a message into an agent's next prompt"),
    BotCommand("history", "View recent agent handoffs"),
    BotCommand("agents", "Check agent binary health"),
    BotCommand("watch", "Auto-push updates on state changes"),
    BotCommand("unwatch", "Stop auto-push updates"),
    BotCommand("results", "View task result report"),
    BotCommand("cancelrun", "Kill the active runner process"),
    BotCommand("queue", "Add or view task queue"),
    BotCommand("queue_next", "Start next queued task"),
    BotCommand("queue_remove", "Remove queued task"),
    BotCommand("queue_clear", "Clear task queue"),
    BotCommand("whitelist", "Manage access control"),
]


async def _post_init(application: Application) -> None:
    """Register the command menu with Telegram after initialization."""
    try:
        await application.bot.set_my_commands(_BOT_COMMANDS)
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not set bot commands menu: %s", exc)


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    try:
        token = get_bot_token()
        if not token:
            logger.error("Bot token is missing or empty. Check your configuration.")
            sys.exit(1)
    except Exception as exc:
        logger.error("Failed to retrieve bot token: %s", exc)
        sys.exit(1)

    try:
        application = ApplicationBuilder().token(token).post_init(_post_init).build()
    except Exception as exc:
        logger.error("Failed to build Telegram application: %s", exc)
        sys.exit(1)

    register_handlers(application)

    schedule_watcher_jobs(application)
    observer = start_watchdog_observer(application)
    application.bot_data["observer"] = observer

    def _signal_handler(signum: int, _frame) -> None:
        logger.info("Received signal %d, shutting down gracefully...", signum)
        stop_watchdog_observer(observer)
        application.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("ARN Telegram Bot starting polling...")
    application.run_polling()
    logger.info("ARN Telegram Bot stopped.")


if __name__ == "__main__":
    main()
