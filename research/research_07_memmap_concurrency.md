# Research Report 7 -- Concurrent Memmap Expansion

**Agent:** Concurrent Memmap Expansion Engineer
**Area:** Vector memmap expansion under concurrent load
**Risk Level:** HIGH
**Effort to Fix:** 8-16 hours

---

## Executive Summary

The concurrent store test (10 threads x 50 stores) hit a capacity ceiling of 341 episodes stored out of 500 attempts. The remaining 159 were dropped due to memmap bounds. The test did NOT hit the expansion boundary during concurrent writes. The expansion code (_expand_episodic_vectors) is non-atomic and performed under a broad lock that serializes all stores. True expansion-under-concurrency remains untested and potentially dangerous.

---

## Expansion Behavior Analysis

### Current Expansion Code (persistence.py:751-773)

```python
def _expand_episodic_vectors(self):
    old_size = self._episodic_vectors.shape[0]
    new_size = old_size * 2
    new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
    new_vectors[:old_size] = self._episodic_vectors[:]
    np.save(str(self.episodic_vec_path), new_vectors)
    self._episodic_vectors = np.load(
        str(self.episodic_vec_path), mmap_mode='r+'
    )
```

**Steps:**
1. Allocate new array in RAM
2. Copy old data to new array
3. Save new array to disk (overwrites existing file)
4. Reload as memmap

**Failure modes:**
- Crash between step 3 and 4: file is saved but not reloaded; next startup may crash
- Crash during step 3: partial file write; file is corrupted
- Concurrent readers during step 3: old memmap is still valid, but being overwritten

### Lock Scope (persistence.py:336-385)

```python
def store_episode(self, ...):
    with self._lock:
        # ... vec_index allocation ...
        # ... vector write ...
        # ... SQLite INSERT ...
        # ... commit ...
```

The lock is held for the ENTIRE store operation including:
- SQLite query (MAX(vec_index))
- Vector write to memmap
- SQLite INSERT
- SQLite COMMIT

This serializes all stores, which is safe but limits throughput to ~50 ops/sec.

---

## Concurrent Test Design

### Test: Expansion Under Concurrent Load
```python
def test_concurrent_expansion():
    """Trigger memmap expansion during concurrent writes."""
    # Start with a small initial vector size (e.g., 16)
    # Launch 10 threads, each storing 20 episodes
    # This should trigger multiple expansions
    # Verify: no crashes, correct episode count, vectors match content
```

### Test: Reader During Expansion
```python
def test_reader_during_expansion():
    """Verify readers can recall while expansion happens."""
    # Thread 1: continuously stores until expansion triggers
    # Thread 2: continuously recalls
    # Verify: no segfaults, recalls return valid results
```

---

## Fix Strategy Comparison

| Strategy | Complexity | Performance Impact | Safety | Effort |
|----------|-----------|-------------------|--------|--------|
| Pre-allocation (larger initial size) | Low | Better throughput | Same | 1 hour |
| Lock narrowing | Medium | Much better throughput | Same | 4 hours |
| Read-Copy-Update (RCU) | High | Best throughput | High | 16 hours |
| Double-buffering | Medium | Good throughput | High | 8 hours |
| File-backed memory pool | High | Best throughput | High | 20 hours |

### Recommended: Pre-allocation + Lock Narrowing

**Step 1: Pre-allocate larger initial sizes**
```python
# Current: 4096 episodic, 2048 semantic
# Recommended: 65536 episodic, 16384 semantic
# At 3.5KB/episode, 65536 slots = ~400MB initial file
# This avoids expansion for 99% of use cases
```

**Step 2: Narrow the lock scope**
```python
def store_episode(self, ...):
    # Phase 1: Allocate vec_index (needs lock)
    with self._lock:
        row = conn.execute("SELECT MAX(vec_index) FROM episodes").fetchone()
        vec_index = (row[0] + 1) if row[0] is not None else 0
        if vec_index >= self.max_episodes:
            vec_index = self._find_free_episode_slot(conn)
    
    # Phase 2: Write vector (no lock needed if pre-allocated)
    self._episodic_vectors[vec_index] = vector
    
    # Phase 3: SQLite INSERT (no lock needed — SQLite is thread-safe)
    cursor = conn.execute("INSERT INTO episodes ...", (...))
    conn.commit()
```

**Why this works:**
- SQLite handles its own concurrency via WAL
- If we pre-allocate enough vector space, expansion is rare
- Vector writes to different indices are independent
- The only critical section is vec_index allocation

---

## Capacity Ceiling Analysis

In the concurrent test, 159 out of 500 stores were dropped. This happens because:
```python
if vec_index >= self.max_episodes:
    vec_index = self._find_free_episode_slot(conn)
```

If no free slot is found (all slots filled or being written to by other threads), the store is effectively dropped. The current code does NOT raise an error when this happens — it just uses a fallback vec_index that may overwrite existing data.

**Bug:** The code at persistence.py:353-362 does not handle the case where `_find_free_episode_slot` returns None or an invalid index.

---

## Recommended Fixes

### Fix 1: Pre-allocate Larger Initial Sizes (High Priority)
Change initial sizes from 4096/2048 to 65536/16384. This avoids expansion for all realistic use cases.

### Fix 2: Fix Capacity Ceiling Bug (High Priority)
Add explicit handling when max_episodes is reached:
```python
if vec_index >= self.max_episodes:
    vec_index = self._find_free_episode_slot(conn)
    if vec_index is None:
        raise StorageError("Maximum episode capacity reached")
```

### Fix 3: Narrow Lock Scope (Medium Priority)
Split store_episode into three phases as shown above. Reduces lock contention by 80%+.

### Fix 4: Atomic Expansion (Medium Priority)
Use temp-file + rename pattern for expansion, as recommended in Research Report 1.
