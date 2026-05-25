"""Integration tests for the ARN Telegram Bot."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from arn_v9.scripts.telegram_bot import task_queue
from arn_v9.scripts.telegram_bot.bridge import (
    format_status_card,
    read_arn_state,
    slugify_task_id,
    write_task_file,
)
from arn_v9.scripts.telegram_bot.config import DATA_DIR, VALID_AGENTS
from arn_v9.scripts.telegram_bot.handlers import callback_router
from arn_v9.scripts.telegram_bot.watcher import (
    _detect_transition,
    get_watching_chats,
    register_watch,
    unregister_watch,
)


def test_queue() -> None:
    print("\n=== Queue Tests ===")
    task_queue.clear()
    assert task_queue.length() == 0

    task_queue.enqueue("ARN-test-1", "Test 1", "Desc 1", ["codex"])
    task_queue.enqueue("ARN-test-2", "Test 2", "Desc 2", ["claude", "kimi"])
    assert task_queue.length() == 2

    tasks = task_queue.list_tasks()
    assert tasks[0]["task_id"] == "ARN-test-1"
    assert tasks[1]["chain"] == ["claude", "kimi"]

    next_task = task_queue.dequeue()
    assert next_task is not None
    assert next_task["task_id"] == "ARN-test-1"
    assert task_queue.length() == 1

    assert task_queue.remove(0) is True
    assert task_queue.length() == 0
    assert task_queue.remove(0) is False

    task_queue.clear()
    print("✅ Queue tests passed")


def test_bridge() -> None:
    print("\n=== Bridge Tests ===")
    state = read_arn_state()
    assert "status" in state
    card = format_status_card(state)
    assert "ARN Collab Status" in card
    assert "Queue:" not in card  # queue is empty

    task_queue.enqueue("ARN-queued", "Queued", "Desc", VALID_AGENTS)
    card = format_status_card(state)
    assert "Queue:" in card
    task_queue.clear()
    print("✅ Bridge tests passed")


def test_watcher_transitions() -> None:
    print("\n=== Watcher Transition Tests ===")

    # IDLE -> CLAIMED
    prev = {"status": "IDLE", "locked_by": None}
    curr = {"status": "CLAIMED_CODEX", "locked_by": "codex"}
    msgs = _detect_transition(prev, curr)
    assert any("codex has claimed" in m for m in msgs)

    # CLAIMED -> HANDOFF
    prev = {"status": "CLAIMED_CODEX", "locked_by": "codex"}
    curr = {"status": "HANDOFF_CODEX", "locked_by": None}
    msgs = _detect_transition(prev, curr)
    assert any("completed" in m for m in msgs)

    # HANDOFF -> CLAIMED (next agent)
    prev = {"status": "HANDOFF_CODEX", "locked_by": None}
    curr = {"status": "CLAIMED_CLAUDE", "locked_by": "claude"}
    msgs = _detect_transition(prev, curr)
    assert any("claude has claimed" in m for m in msgs)

    # -> DONE
    prev = {"status": "CLAIMED_KIMI", "locked_by": "kimi"}
    curr = {"status": "DONE", "locked_by": None}
    msgs = _detect_transition(prev, curr)
    assert any("DONE" in m for m in msgs)

    print("✅ Watcher transition tests passed")


def test_watcher_chat_registry() -> None:
    print("\n=== Watcher Chat Registry Tests ===")
    register_watch(12345)
    register_watch(67890)
    assert 12345 in get_watching_chats()
    assert 67890 in get_watching_chats()
    unregister_watch(12345)
    assert 12345 not in get_watching_chats()
    assert 67890 in get_watching_chats()
    unregister_watch(67890)
    print("✅ Chat registry tests passed")


async def test_callback_router() -> None:
    print("\n=== Callback Router Tests ===")

    # Build a mock Update + CallbackQuery
    mock_query = MagicMock()
    mock_query.data = "status:refresh"
    mock_query.answer = AsyncMock()
    mock_query.message.chat.id = 99999

    mock_update = MagicMock()
    mock_update.callback_query = mock_query

    mock_context = MagicMock()
    mock_context.bot = MagicMock()
    mock_context.bot.send_message = AsyncMock()

    # Test unknown callback
    mock_query.data = "stale:dismiss:codex"
    await callback_router(mock_update, mock_context)
    mock_query.answer.assert_called()

    print("✅ Callback router tests passed")


def test_slugify() -> None:
    print("\n=== Slugify Tests ===")
    assert slugify_task_id("fix recall bug") == "ARN-fix-recall-bug"
    assert slugify_task_id("My Task!!!") == "ARN-My-Task"
    print("✅ Slugify tests passed")


def test_task_file_write() -> None:
    print("\n=== Task File Write Tests ===")
    path = write_task_file("ARN-test-write", "Test Write", "Test desc", ["kimi"])
    assert path.exists()
    text = path.read_text()
    assert "ARN-test-write" in text
    assert "kimi" in text
    path.unlink()
    print("✅ Task file write tests passed")


async def main() -> None:
    test_queue()
    test_bridge()
    test_watcher_transitions()
    test_watcher_chat_registry()
    await test_callback_router()
    test_slugify()
    test_task_file_write()
    print("\n🎉 All integration tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
