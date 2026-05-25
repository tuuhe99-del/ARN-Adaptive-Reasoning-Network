# ARN Task: Overhaul — Simplify, Fix Bugs, Harden

## Task ID
`ARN-overhaul-simplify`

## Review Chain
```
claude → kimi
```

## Context
A full audit of the ARN v9 codebase identified ~4,000 lines of dead code, several functional bugs, and missing production hardening. This task implements all fixes in one pass. The goal is a leaner, correct, production-ready ARN core with zero functional regression on store/recall/persist/inject.

Previous completed work is in `tasks/ARN-live-integration-test-result.md`. Do not repeat what's already verified there.

---

## SECTION 1: Remove Dead Code (no regressions possible)

### 1a. `arn_v9/core/extensions.py` — purge unused classes and functions

Remove these entirely (confirmed zero callers in the production path):

- **`HybridRetriever` class** — BM25 index is built on startup but `hybrid_search()` is never called anywhere. Remove the class, the `rank_bm25` import, and any instantiation in `plugin.py` (`self._hybrid`).
- **`StoreCallbackManager` class** — callbacks can be registered but `.fire()` is never called. Remove the class and any instantiation.
- **`EntityExtractor` class** + `store_entity`, `search_entities`, `get_entity_episodes`, `get_episode_entities` — never called from `perceive()` or any API path. Remove all four functions and the class.
- **`forget_by_query`, `forget_by_tag`, `apply_importance_decay`, `purge_expired`, `export_memory`, `import_memory`, `get_version_history`** — defined in extensions.py with no REST endpoints and no callers in the main path. Remove all seven functions.

After removing, clean up any unused imports (`rank_bm25`, anything only used by removed code).

### 1b. `arn_v9/core/persistence.py` — remove entity tables

Remove the `CREATE TABLE IF NOT EXISTS entities` and `CREATE TABLE IF NOT EXISTS entity_episodes` SQL from `_create_tables()`. Remove any indexes for those tables. These tables exist in the schema but are never written to.

### 1c. Remove `arn_v9/memory_llm.py` entirely

This file (~433 lines) duplicates what the OpenClaw plugin does better. It has no API surface and no callers outside its own tests. Delete the file. If a test imports it, delete that test or replace with a comment explaining why it was removed.

### 1d. `arn_v9/api/server.py` — extract dashboard HTML + remove dead dependency

**Dashboard extraction:** The `/dashboard` route returns a 1,100+ line HTML/CSS/JS string embedded in server.py. Extract it:
1. Create `arn_v9/api/dashboard.html` with the full HTML content
2. In server.py, replace the inline string with: read the file at import time (or on first request) and return it
3. server.py should shrink to ~700 lines from ~1,841

**Dead dependency:** Remove the `check_rate_limit` function — it's defined as a FastAPI dependency but never used in any `Depends()` call. The actual rate limiting happens via inline `rate_limiter.check()` calls. Just delete the function.

---

## SECTION 2: Fix Functional Bugs

### 2a. Vector slot overflow bug — `persistence.py`

`_find_free_episode_slot()` falls back to returning slot `0` when all slots are full. This silently overwrites the oldest episode's embedding vector with new content — a data corruption bug.

**Fix:** When no free slot exists, find the episode with the oldest `created_at` timestamp, remove it from the memmap slot, and return that slot number. Log a warning: "Memory capacity reached, evicting oldest episode (id=X) to make room."

### 2b. Domain column prototypes lost on restart — `cognitive.py`

`_save_state()` saves `error_mean`, `error_var`, `expertise`, `sample_count` per column but NOT the prototype vector. On every restart, `_init_columns()` re-encodes 8 hardcoded seed phrases, discarding what the column learned.

**Fix in `_save_state()`:** Add prototype to the saved state for each column:
```python
"prototype": col.prototype.tolist() if col.prototype is not None else None
```

**Fix in `_load_state()` / `_init_columns()`:** After loading saved state, if a column's `prototype` key is present and non-null, restore it with `np.array(data["prototype"], dtype=np.float32)` instead of re-encoding the seed phrase.

### 2c. Score weights don't sum to 1.0 — `cognitive.py`

In `recall()`, the base weights are 0.55 (similarity) + 0.20 (importance) + 0.15 (recency) = 0.90. This means the maximum base score is 0.90, with a 0.05 bonus for non-superseded memories capping it at 0.95. The math is inconsistent and confusing.

**Fix:** Adjust weights to: similarity=0.60, importance=0.20, recency=0.15, non-superseded bonus=0.05. These sum to 1.0 at maximum. Alternatively restructure as: `score = 0.60*sim + 0.20*imp + 0.15*rec + 0.05*(not superseded)`. Update the comment explaining the formula.

### 2d. `prediction_error` — use it or lose it — `cognitive.py`

Each episode stores a `prediction_error` field (0.0–1.0, where 1.0 = maximally surprising). It is stored in the DB but never used in `recall()` scoring. It's used in consolidation replay priority (weight 0.3) — that's legitimate.

**Decision: use it in recall.** High-prediction-error memories are more informative (the agent was most surprised). Add a small bonus in `recall()`:
```python
# Surprise bonus: high-error episodes are more informative
surprise_bonus = 0.05 * episode.prediction_error
score += surprise_bonus
```
Adjust the other weights down slightly (similarity=0.58, importance=0.19, recency=0.13, non-superseded=0.05, surprise=0.05 = 1.0 at maximum).

### 2e. Async blocking — `server.py`

Every FastAPI endpoint is `async def` but `plugin.store()` and `plugin.recall()` call embedding synchronously on the CPU, blocking the event loop for 30–80ms per call.

**Fix:** In `server.py`, wrap the blocking plugin calls in `asyncio.to_thread()`:
```python
result = await asyncio.run_in_executor(None, plugin.store, ...)
# or
result = await asyncio.to_thread(plugin.store, ...)
```

Do this for at least the `store` and `recall` endpoints. If `recall` calls `plugin.recall()` synchronously, wrap the whole call. Add `import asyncio` if not already present.

### 2f. Default API key on startup — `server.py`

If `ARN_API_KEY` environment variable is not set at startup, ARN runs with zero authentication — any process can read or write any agent's memories.

**Fix:** In the FastAPI startup event (or at module level):
```python
import secrets
from pathlib import Path

_key_file = Path(settings.data_root) / ".api_key"
if not os.getenv("ARN_API_KEY"):
    if _key_file.exists():
        _auto_key = _key_file.read_text().strip()
    else:
        _auto_key = secrets.token_urlsafe(32)
        _key_file.write_text(_auto_key)
    os.environ["ARN_API_KEY"] = _auto_key
    logger.warning(f"ARN_API_KEY not set. Auto-generated key active: {_auto_key[:8]}... (full key in {_key_file})")
```

This ensures there's always an auth key, persisted across restarts, without breaking existing setups that set `ARN_API_KEY` explicitly. Existing code that checks `ARN_API_KEY` will continue to work.

**IMPORTANT:** After adding this, update the tests that call the API without a key — add `headers={"X-API-Key": settings.api_key}` or set `ARN_API_KEY=""` in test env to skip auth for tests (check how the existing auth middleware handles empty string).

---

## SECTION 3: Working Memory Persistence

Working memory currently starts empty every session. The agent has no short-term context from where it left off.

### 3a. Serialize working memory to disk — `cognitive.py`

In `ARNPlugin` (or wherever `WorkingMemory` lives), after every `perceive()` call, serialize the working memory to `{data_dir}/{agent_id}/working_memory.json`:

```python
# Serializable fields: recent_inputs (list of str), attention_weights (dict), active_goals (list)
# Do NOT serialize the full episode objects — just their IDs and summaries
wm_data = {
    "recent_input_ids": [ep.id for ep in self._working.recent_episodes[-10:]],
    "attention_weights": dict(self._working.attention_weights),
    "updated_at": time.time(),
}
wm_path = self._data_dir / self._agent_id / "working_memory.json"
wm_path.write_text(json.dumps(wm_data))
```

On `ARNPlugin` init, load the file if it exists and restore `recent_episodes` by fetching those episode IDs from persistence. Use the last 10 episodes maximum.

Adapt the exact field names to what `WorkingMemory` actually has — the goal is: after a restart, the agent's working context window is NOT empty, it picks up the last 10 memories it was actively processing.

---

## SECTION 4: Honest Naming (minor)

### 4a. `SimilarityCalibrator.observe()` — rename to `record()`

The method records stats but explicitly does NOT adapt the sigmoid (the adaptive approach was removed because it drifted). Rename `observe()` → `record()` everywhere it's called. Update the docstring to say: "Records the similarity score for statistical tracking. Does not adapt calibration parameters."

### 4b. Remove `domain_signals` from recall/perceive response where not needed

The `perceive()` return includes a verbose `domain_signals` list (one entry per column). No API caller uses this — the server returns it but the plugin ignores it. Remove it from the `PerceiveResult` return in the API layer (server.py response model). Keep it in the internal `cognitive.py` return for future use. Just don't expose it over the HTTP API where it's pure noise.

---

## SECTION 5: Tests

After all changes:
1. Run `python3 -m pytest arn_v9/tests/ -q` — must pass 30+ tests (some tests for removed code will be deleted)
2. Run `python3 -m py_compile arn_v9/api/server.py arn_v9/core/cognitive.py arn_v9/core/embeddings.py arn_v9/core/persistence.py arn_v9/plugin.py`
3. Add a **precision/recall quality test**: store 10 distinct facts for a test agent, recall each by a natural-language query, assert the correct fact is in the top-2 results. This goes in `arn_v9/tests/test_all.py`.
4. Verify the dashboard is still served correctly: `GET /dashboard` returns 200 with HTML content.
5. Verify the API key mechanism: start a test client, confirm an unauthenticated request gets 401/403 (or that the auto-key is set).

---

## Handoff Command

```bash
python3 arn_v9/scripts/arn_cli.py collab handoff \
  --agent claude \
  --status complete \
  --task "ARN overhaul: simplify, fix bugs, harden" \
  --changes "..." \
  --verification "pytest: X passed. py_compile: OK. Lines removed: ~N. Dead code cut: [list]. Bugs fixed: [list]. Working memory persistence: [yes/no]. Auth key: [auto-generated/not done]." \
  --concerns "..." \
  --next-focus "Kimi: review the changes, run pytest, test the store/recall API directly with curl or requests, confirm dashboard still loads, confirm auth key is working, verify working memory loads on restart."
```

## Success Criteria
- [ ] `extensions.py` has no BM25, no StoreCallbackManager, no EntityExtractor, no unused bulk functions
- [ ] `persistence.py` has no entity tables
- [ ] `memory_llm.py` deleted
- [ ] `server.py` < 800 lines (dashboard extracted to separate file)
- [ ] `dashboard.html` exists and is served correctly
- [ ] Vector slot overflow evicts oldest instead of corrupting slot 0
- [ ] Domain column prototypes saved and restored across restarts
- [ ] Score weights sum to 1.0
- [ ] `prediction_error` used in recall scoring
- [ ] Embedding calls wrapped in asyncio thread executor
- [ ] Auto-generated API key on startup if none set
- [ ] Working memory loaded from disk on agent init
- [ ] `observe()` renamed to `record()`
- [ ] `domain_signals` removed from HTTP API response
- [ ] 30+ tests passing
- [ ] Precision/recall quality test added and passing
