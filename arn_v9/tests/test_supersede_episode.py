"""
Direct unit tests for arn_v9.extensions.supersede_episode().

The function is called by the contradiction detector to mark an old
episode as superseded by a new one.  These tests verify the exact
SQLite state produced by the call — something no indirect test checks.
"""

import time
import sqlite3
import numpy as np
import pytest

from arn_v9.storage.persistence import StorageEngine
from arn_v9.extensions import supersede_episode


DIM = 384


def _make_storage(tmp_path) -> StorageEngine:
    return StorageEngine(str(tmp_path), max_episodes=64, max_semantics=32, embedding_dim=DIM)


def _store(storage: StorageEngine, text: str) -> int:
    vec = np.zeros(DIM, dtype=np.float32)
    return storage.store_episode(content=text, vector=vec, source="user")


def _row(storage: StorageEngine, episode_id: int) -> sqlite3.Row:
    conn = storage._get_conn()
    return conn.execute(
        "SELECT id, superseded_by, invalidated_at FROM episodes WHERE id = ?",
        (episode_id,),
    ).fetchone()


# ──────────────────────────────────────────────────────────────────────────────
# Core behaviour
# ──────────────────────────────────────────────────────────────────────────────

def test_superseded_by_is_set_to_new_id(tmp_path):
    storage = _make_storage(tmp_path)
    old_id = _store(storage, "User's name is Alice.")
    new_id = _store(storage, "User's name is Bob.")

    supersede_episode(storage, old_id, new_id)

    row = _row(storage, old_id)
    assert row["superseded_by"] == new_id


def test_invalidated_at_is_set_after_supersession(tmp_path):
    storage = _make_storage(tmp_path)
    before = time.time()
    old_id = _store(storage, "User prefers Python.")
    new_id = _store(storage, "User prefers Rust.")

    supersede_episode(storage, old_id, new_id)

    row = _row(storage, old_id)
    assert row["invalidated_at"] is not None
    assert row["invalidated_at"] >= before


def test_new_episode_is_not_invalidated(tmp_path):
    """The replacement episode itself must remain active."""
    storage = _make_storage(tmp_path)
    old_id = _store(storage, "User prefers Python.")
    new_id = _store(storage, "User prefers Rust.")

    supersede_episode(storage, old_id, new_id)

    row = _row(storage, new_id)
    assert row["invalidated_at"] is None
    assert row["superseded_by"] is None


def test_unrelated_episode_is_untouched(tmp_path):
    """Episodes not involved in the call must not be modified."""
    storage = _make_storage(tmp_path)
    old_id = _store(storage, "User prefers Python.")
    new_id = _store(storage, "User prefers Rust.")
    other_id = _store(storage, "Unrelated fact about something else.")

    supersede_episode(storage, old_id, new_id)

    row = _row(storage, other_id)
    assert row["invalidated_at"] is None
    assert row["superseded_by"] is None


def test_supersession_is_persisted_across_reconnect(tmp_path):
    """The change must survive closing and re-opening storage."""
    storage = _make_storage(tmp_path)
    old_id = _store(storage, "User is based in London.")
    new_id = _store(storage, "User is based in Tokyo.")

    supersede_episode(storage, old_id, new_id)
    storage._conn.close()

    # Re-open with a fresh StorageEngine
    storage2 = _make_storage(tmp_path)
    row = _row(storage2, old_id)
    assert row["superseded_by"] == new_id
    assert row["invalidated_at"] is not None


def test_supersede_nonexistent_episode_does_not_raise(tmp_path):
    """Calling supersede_episode with a non-existent old_id must not crash."""
    storage = _make_storage(tmp_path)
    new_id = _store(storage, "Some new fact.")

    # 99999 does not exist — should be a silent no-op, not an exception
    supersede_episode(storage, 99999, new_id)


def test_invalidated_at_is_close_to_current_time(tmp_path):
    storage = _make_storage(tmp_path)
    old_id = _store(storage, "User prefers dark mode.")
    new_id = _store(storage, "User prefers light mode.")

    t_before = time.time()
    supersede_episode(storage, old_id, new_id)
    t_after = time.time()

    row = _row(storage, old_id)
    assert t_before <= row["invalidated_at"] <= t_after + 1.0
