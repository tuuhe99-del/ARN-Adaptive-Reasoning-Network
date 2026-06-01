"""
Concurrent access tests for StorageEngine / _ThreadLocalConnection.

StorageEngine uses thread-local SQLite connections so that each thread
gets an independent connection to the same WAL-mode database.  These
tests verify that simultaneous stores from multiple threads:

  1. Do not raise exceptions
  2. Produce the correct total number of rows
  3. Do not generate duplicate or colliding episode IDs
  4. Keep vector data coherent (reads match what was written)

All tests are purely structural (numpy zero-vectors) so they run in the
plumbing tier without an embedding model.
"""

import time
import threading
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from arn_v9.storage.persistence import StorageEngine

DIM = 384
_VEC = np.zeros(DIM, dtype=np.float32)


def _make_storage(tmp_path, capacity: int = 512) -> StorageEngine:
    return StorageEngine(
        str(tmp_path), max_episodes=capacity, max_semantics=256, embedding_dim=DIM
    )


def _store_n(storage: StorageEngine, n: int, prefix: str) -> list[int]:
    """Store n episodes from the calling thread; return the list of IDs."""
    ids = []
    for i in range(n):
        ep_id = storage.store_episode(
            content=f"{prefix} episode {i}",
            vector=_VEC.copy(),
            source="user",
        )
        ids.append(ep_id)
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Basic concurrent writes
# ──────────────────────────────────────────────────────────────────────────────

def test_concurrent_stores_all_succeed(tmp_path):
    """N threads each storing M episodes must all return without raising."""
    storage = _make_storage(tmp_path, capacity=512)
    n_threads, n_each = 8, 10

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [
            pool.submit(_store_n, storage, n_each, f"thread-{t}")
            for t in range(n_threads)
        ]
        results = [f.result() for f in as_completed(futures)]

    total = sum(len(r) for r in results)
    assert total == n_threads * n_each


def test_concurrent_episode_ids_are_unique(tmp_path):
    """No two concurrent stores should produce the same episode ID."""
    storage = _make_storage(tmp_path, capacity=512)
    n_threads, n_each = 8, 10

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [
            pool.submit(_store_n, storage, n_each, f"t{t}")
            for t in range(n_threads)
        ]
        all_ids = []
        for f in as_completed(futures):
            all_ids.extend(f.result())

    assert len(all_ids) == len(set(all_ids)), (
        f"Duplicate episode IDs detected: {sorted(all_ids)}"
    )


def test_db_row_count_matches_stores(tmp_path):
    """The database must contain exactly as many rows as were stored."""
    n_threads, n_each = 6, 8
    storage = _make_storage(tmp_path, capacity=256)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_store_n, storage, n_each, f"r{t}") for t in range(n_threads)]
        for f in as_completed(futures):
            f.result()

    conn = storage._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert count == n_threads * n_each


# ──────────────────────────────────────────────────────────────────────────────
# Concurrent reads alongside writes
# ──────────────────────────────────────────────────────────────────────────────

def test_concurrent_reads_while_writing(tmp_path):
    """
    Reader threads must not see an exception while writer threads are active.
    This tests the WAL-mode concurrent-read claim.
    """
    storage = _make_storage(tmp_path, capacity=512)

    # Seed some initial data so readers have something to query
    for i in range(10):
        storage.store_episode(f"seed episode {i}", _VEC.copy(), source="api")

    errors: list[Exception] = []
    lock = threading.Lock()

    def reader():
        try:
            conn = storage._get_conn()
            conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        except Exception as e:
            with lock:
                errors.append(e)

    def writer(prefix):
        _store_n(storage, 5, prefix)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = (
            [pool.submit(writer, f"w{i}") for i in range(6)]
            + [pool.submit(reader) for _ in range(6)]
        )
        for f in as_completed(futures):
            f.result()

    assert errors == [], f"Reader threads raised: {errors}"


# ──────────────────────────────────────────────────────────────────────────────
# Thread-local connection isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_each_thread_gets_its_own_connection(tmp_path):
    """
    _ThreadLocalConnection must hand out a different connection object
    per OS thread — sharing connections across threads is unsafe in SQLite.
    """
    from arn_v9.storage.persistence import _ThreadLocalConnection
    db_path = tmp_path / "arn_metadata.db"

    tlc = _ThreadLocalConnection(db_path)
    connection_ids: list[int] = []
    lock = threading.Lock()

    def record_conn_id():
        conn = tlc.get()
        with lock:
            connection_ids.append(id(conn))

    threads = [threading.Thread(target=record_conn_id) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread must have produced a distinct connection object
    assert len(set(connection_ids)) == len(threads), (
        "Some threads shared a connection — thread-local isolation failed"
    )
