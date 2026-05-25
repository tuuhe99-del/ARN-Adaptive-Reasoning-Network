# Research Report 3 -- Resource Exhaustion

**Agent:** Resource Exhaustion Engineer
**Area:** Disk-full and OOM handling
**Risk Level:** MEDIUM
**Effort to Fix:** 6-10 hours

---

## Executive Summary

ARN v9 stores approximately 3.5KB per episode (SQLite metadata + 384-dim float32 vector = 1.5KB raw vector + ~2KB SQLite). At this rate, 100K episodes need ~350MB disk. The embedding model holds ~90MB RAM. No disk-space checks exist before writes. The FastAPI server has no explicit request body size limit.

---

## Failure Mode Analysis

| Resource | Failure Mode | Current Behavior | Desired Behavior |
|----------|-------------|------------------|------------------|
| Disk full during SQLite INSERT | sqlite3.OperationalError | Crashes the store request | Graceful rejection |
| Disk full during memmap expansion | np.save() may fail | Truncated .npy file | Atomic expansion |
| OOM during model load | Python MemoryError | Crash on startup | Clear error, suggest smaller tier |
| OOM during large request | FastAPI may crash | Server restart | Request size limit |
| Too many agents | AgentPool evicts LRU | Data loss risk | Pre-flush on eviction |

---

## Detailed Findings

### 1. No Disk Space Checks (persistence.py:328-385)

store_episode() performs SQLite INSERT and np.save without checking available disk space:
```python
np.save(str(self.episodic_vec_path), new_vectors)
conn.execute("INSERT INTO episodes ...")
```

SQLite raises OperationalError but NumPy may produce a truncated file.

### 2. AgentPool Eviction Without Flush (server.py:236-245)

```python
def _evict_oldest(self):
    oldest = min(self._access_times, key=self._access_times.get)
    plugin = self._plugins.pop(oldest, None)
    if plugin:
        plugin.shutdown()
```

The `shutdown()` call does flush SQLite and memmap, but if shutdown itself fails (e.g., disk full), the eviction proceeds anyway.

### 3. Resource Budgets

| Metric | Value | Notes |
|--------|-------|-------|
| Per-episode disk | ~3.5KB | 2KB SQLite + 1.5KB vector |
| Per-episode RAM | ~0KB | Vectors are memmapped |
| Model RAM | ~90MB | MiniLM-L6-v2 |
| Max agents (default) | 100 | Configurable via ARN_MAX_AGENTS |
| Max episodes per agent | 1M | Hard cap in code |
| 100K episodes disk | ~350MB | Excluding growth overhead |
| 100 agents total disk | ~35GB | Worst case |

---

## Recommended Fixes

### Fix 1: Pre-flight Disk Check (Medium Priority)
Add a disk-space check before expansion:
```python
import shutil
free = shutil.disk_usage(self.data_dir).free
needed = new_vectors.nbytes * 2  # double for temp file
if free < needed:
    raise StorageError(f"Insufficient disk space: {free} bytes available, {needed} needed")
```

### Fix 2: Request Size Limit (Medium Priority)
Add max_content_length to StoreRequest:
```python
class StoreRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000)  # ~100KB limit
```

### Fix 3: Graceful Degradation on Disk Full (High Priority)
Catch disk-full errors and return HTTP 507 Insufficient Storage:
```python
except sqlite3.OperationalError as e:
    if "disk full" in str(e).lower():
        raise HTTPException(status_code=507, detail="Disk full")
    raise
```

### Fix 4: Resource Monitoring Endpoint (Low Priority)
Add `/v1/health/storage` that reports disk usage per agent.
