# Research Report 1 -- Data Integrity & Crash Recovery

**Agent:** Data Integrity & Crash Recovery Specialist
**Area:** Long-term data integrity under crash conditions
**Risk Level:** HIGH
**Effort to Fix:** 8-12 hours

---

## Executive Summary

ARN v9 uses SQLite with WAL mode and NumPy memmap vectors. WAL provides basic crash safety for SQL operations, but the memmap vector file has NO crash protection. The _expand_episodic_vectors() method performs a non-atomic save-and-reload sequence that can leave the .npy file corrupted if the process dies mid-expansion. Schema migration code contains `except: pass` blocks that silently swallow failures.

---

## Crash Scenarios

| Scenario | Probability | Impact | Current Behavior | Safe? |
|----------|-------------|--------|------------------|-------|
| Process dies during SQLite COMMIT | Low | Medium | WAL auto-recovers on next open | Yes |
| Process dies during memmap expansion | Low | Critical | .npy file may be truncated | No |
| Process dies between vector write and SQLite INSERT | Very Low | High | Vector written but no metadata | Partial |
| Power loss during WAL checkpoint | Very Low | Medium | SQLite recovers from WAL | Yes |
| Disk full during expansion | Medium | High | np.save() may fail; old file lost | No |

---

## Detailed Findings

### 1. SQLite WAL Configuration (persistence.py:46-55)

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA cache_size=2000")
```

- WAL mode: Enabled. Provides crash safety for committed transactions.
- synchronous=NORMAL: SQLite flushes WAL at each checkpoint, but not at every COMMIT. A crash could lose the most recent transaction.
- Recommendation: For production, consider `synchronous=FULL` if durability is more important than write speed.

### 2. Memmap Expansion is Non-Atomic (persistence.py:751-773)

```python
def _expand_episodic_vectors(self):
    old_size = self._episodic_vectors.shape[0]
    new_size = old_size * 2
    new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
    new_vectors[:old_size] = self._episodic_vectors[:]
    np.save(str(self.episodic_vec_path), new_vectors)      # crash here = corrupted file
    self._episodic_vectors = np.load(
        str(self.episodic_vec_path), mmap_mode='r+'
    )                                                        # crash here = truncated file
```

Failure mode: If the process is killed between np.save() and np.load(), the .npy file exists but may be partially written. On next startup, np.load() in _init_vectors() will crash with a corrupted file error.

There is NO recovery path. The _init_vectors() code (persistence.py:292-322) has `except Exception: pass` but a corrupted .npy may cause a segfault rather than a catchable Python exception.

### 3. Schema Migration Silent Failures (persistence.py:249-290)

The `except Exception: pass` pattern appears 6 times in _migrate_schema(). If a migration fails for any reason other than "column already exists" (e.g., disk full, locked table), the failure is silently ignored. The database may be left in a partially-migrated state.

### 4. SQLite-Memmap Consistency

store_episode() (persistence.py:328-385) performs operations in this order:
1. Allocate vec_index via MAX(vec_index)+1
2. Write vector to memmap
3. SQLite INSERT with vec_index

If the process crashes between step 2 and step 3, the vector is written but SQLite has no record of it. This creates an orphan vector.

---

## Recommended Fixes

### Fix 1: Atomic Vector Expansion (High Priority)
Use a write-to-temp-then-rename pattern:
```python
def _expand_episodic_vectors(self):
    old_size = self._episodic_vectors.shape[0]
    new_size = old_size * 2
    new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
    new_vectors[:old_size] = self._episodic_vectors[:]
    
    temp_path = self.episodic_vec_path.with_suffix('.tmp.npy')
    np.save(str(temp_path), new_vectors)
    temp_path.replace(self.episodic_vec_path)  # atomic on POSIX
    
    self._episodic_vectors = np.load(
        str(self.episodic_vec_path), mmap_mode='r+'
    )
```

### Fix 2: Corrupted Vector Recovery (High Priority)
Wrap np.load() in _init_vectors() with validation:
```python
try:
    self._episodic_vectors = np.load(str(self.episodic_vec_path), mmap_mode='r+')
    assert self._episodic_vectors.dtype == np.float32
except (OSError, ValueError, AssertionError):
    logger.warning("Corrupted episodic vectors, creating fresh file")
    self._episodic_vectors = np.zeros((4096, self.embedding_dim), dtype=np.float32)
    np.save(str(self.episodic_vec_path), self._episodic_vectors)
```

### Fix 3: Migration Error Handling (Medium Priority)
Replace `except: pass` with specific exception handling:
```python
try:
    conn.execute(sql)
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        pass
    else:
        raise
```

### Fix 4: Consistency Check Tool (Low Priority)
Add a verify_integrity() method to StorageEngine that scans all episodes and validates vec_index bounds.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Corrupted memmap file | Critical | Low | Fix 1 + Fix 2 |
| Partial schema migration | High | Low | Fix 3 |
| Orphan vectors | Medium | Very Low | Acceptable (wastes space only) |
| WAL data loss | Low | Low | Use synchronous=FULL |
