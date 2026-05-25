# ARN v9 — Foundation Code Forensic Report

**Generated:** 2026-05-16  
**Repo:** `/Users/hustle/arn-v9-repo/`  
**Scope:** All Python source files (≈ 8,003 LOC across 20 non-empty modules).  
**Methodology:** Static analysis via `grep`, `read`, and `wc`. Every claim is verifiable in the source.

---

## 1. Module Call Graph

| Module | Approx. LOC | What It Imports | What Imports It | Key Classes / Functions |
|--------|------------|----------------|-----------------|------------------------|
| `arn_v9/__init__.py` | 22 | `core.cognitive.ARNv9`, `plugin.ARNPlugin` | `tests/check_env.py` (imports `ARNv9`) | `ARNv9`, `ARNPlugin` |
| `arn_v9/core/__init__.py` | 2 | `embeddings.EmbeddingEngine`, `embeddings.EMBEDDING_DIM`, `cognitive.ARNv9`, `cognitive.DomainType`, `cognitive.DomainColumn`, `cognitive.WorkingMemory`, `cognitive.ConsolidationEngine` | — (rarely imported directly) | — |
| `arn_v9/core/cognitive.py` | 1,040 | `numpy`, `time`, `json`, `logging`, `threading`, `typing`, `dataclasses`, `enum`, `collections.deque`;<br>`.embeddings.EmbeddingEngine`, `.embeddings.EMBEDDING_DIM`;<br>`..storage.persistence.StorageEngine`;<br>`.contradictions.ContradictionDetector` | `arn_v9/__init__.py`, `plugin.py`, `tests/test_all.py`, `tests/test_stress_strain.py`, `benchmarks/simulate_agent.py`, `benchmarks/stress_test.py` | `ARNv9`, `ConsolidationEngine`, `WorkingMemory`, `DomainColumn`, `DomainType` |
| `arn_v9/core/embeddings.py` | 564 | `os`, `numpy`, `typing`, `hashlib`, `logging`, `re`, `pathlib.Path`, `warnings` | `core/cognitive.py`, `storage/persistence.py`, `core/__init__.py`, `tests/test_all.py`, `tests/test_stress_strain.py`, `tests/check_env.py`, `api/server.py` | `EmbeddingEngine`, `SimilarityCalibrator`, `EMBEDDING_DIM` |
| `arn_v9/core/contradictions.py` | 368 | `re`, `time`, `logging`, `typing`, `dataclasses`, `numpy` | `core/cognitive.py` | `ContradictionDetector`, `ClaimExtractor`, `Claim` |
| `arn_v9/storage/__init__.py` | 1 | `.persistence.StorageEngine` | — (rarely used) | — |
| `arn_v9/storage/persistence.py` | 811 | `sqlite3`, `numpy`, `os`, `shutil`, `json`, `time`, `hashlib`, `logging`, `threading`, `typing`, `pathlib.Path`;<br>`..core.embeddings.EMBEDDING_DIM` | `core/cognitive.py`, `tests/test_all.py` | `StorageEngine`, `_ThreadLocalConnection` |
| `arn_v9/extensions.py` | 523 | `hashlib`, `json`, `logging`, `re`, `time`, `typing`, `numpy` | `core/contradictions.py` (`supersede_episode`),<br>`plugin.py` (`HybridRetriever`, `StoreCallbackManager`) | `HybridRetriever`, `EntityExtractor`, `StoreCallbackManager`, `export_memory`, `import_memory`, `forget_by_query`, `forget_by_tag`, `apply_importance_decay`, `purge_expired` |
| `arn_v9/plugin.py` | 430 | `os`, `time`, `json`, `logging`, `typing`;<br>`.core.cognitive.ARNv9` | `memory_llm.py`, `api/server.py`, `scripts/arn_cli.py`, `scripts/bootstrap_agent.py`, `scripts/memory_editor.py`, `scripts/test_openclaw_simulation.py`, `benchmarks/simulate_agent.py`, `benchmarks/stress_test.py`, `tests/test_stress_strain.py`, `__init__.py` | `ARNPlugin` |
| `arn_v9/memory_llm.py` | 433 | `os`, `re`, `time`, `json`, `logging`, `typing`;<br>`.plugin.ARNPlugin` | — (user-facing top-level) | `MemoryAugmentedLLM`, `OllamaBackend`, `OpenAICompatibleBackend`, `CallbackBackend` |
| `arn_v9/api/server.py` | 646 | `os`, `sys`, `time`, `json`, `shutil`, `logging`, `asyncio`, `typing`, `contextlib`, `collections.defaultdict`;<br>`fastapi.*`, `pydantic.BaseModel`;<br>`arn_v9.plugin.ARNPlugin` | — (service entry-point) | `app` (FastAPI), `AgentPool`, `RateLimiter` |
| `arn_v9/scripts/arn_cli.py` | 701 | `sys`, `os`, `json`, `argparse`, `hashlib`, `time`, `logging`, `warnings`, `pathlib.Path`;<br>`arn_v9.plugin.ARNPlugin` | — (CLI entry-point) | `get_plugin()`, `cmd_store()`, `cmd_recall()`, `cmd_setup()`, etc. |
| `arn_v9/scripts/bootstrap_agent.py` | 168 | `argparse`, `os`, `sys`, `pathlib.Path`;<br>`arn_v9.plugin.ARNPlugin` | — (migration script) | `migrate_agent()`, `read_markdown_files()` |
| `arn_v9/scripts/memory_editor.py` | 178 | `argparse`, `os`, `sys`, `json`, `subprocess`, `tempfile`;<br>`arn_v9.plugin.ARNPlugin` | — (CLI editor) | `cmd_list()`, `cmd_edit()`, `cmd_add()`, `cmd_delete()`, `cmd_show()` |
| `arn_v9/scripts/test_openclaw_simulation.py` | 231 | `os`, `sys`, `tempfile`, `shutil`, `time`;<br>`arn_v9.plugin.ARNPlugin` | — (simulation harness) | `simulate_turn()`, `run_agent_simulation()` |
| `arn_v9/benchmarks/simulate_agent.py` | 250 | `sys`, `os`, `time`, `json`, `shutil`, `tempfile`, `numpy`;<br>`arn_v9.plugin.ARNPlugin` | — (benchmark) | `run_simulation()` |
| `arn_v9/benchmarks/stress_test.py` | 284 | `sys`, `os`, `time`, `shutil`, `tempfile`, `numpy`, `collections.defaultdict`;<br>`arn_v9.plugin.ARNPlugin` | — (benchmark) | `scenario_cross_session()`, `scenario_distractor()`, etc. |
| `arn_v9/tests/test_all.py` | 750 | `sys`, `os`, `time`, `json`, `shutil`, `tempfile`, `traceback`, `numpy`;<br>`core.embeddings.EmbeddingEngine`, `core.embeddings.EMBEDDING_DIM`;<br>`storage.persistence.StorageEngine`;<br>`core.cognitive.ARNv9`, `core.cognitive.DomainColumn`, etc. | — (test runner) | `TestResults`, `requires_embeddings`, 11 test functions |
| `arn_v9/tests/test_stress_strain.py` | 465 | `os`, `sys`, `time`, `random`, `string`, `tempfile`, `shutil`, `threading`, `concurrent.futures`, `statistics`;<br>`core.cognitive.ARNv9`, `core.embeddings.EmbeddingEngine`, `plugin.ARNPlugin` | — (stress test runner) | `Reporter`, `run_stress_tests()` |
| `arn_v9/tests/check_env.py` | 136 | `sys`, `os` | — (env check) | `check()` |

**Notes**
- `openclaw-arn-plugin/` exists at repo root but contains only JavaScript (`index.js`, `test-plugin.js`, etc.) — outside the Python call graph.
- `memory_llm.py` is *not* imported by any internal module; it is a user-facing wrapper.
- `api/server.py`, `scripts/*`, `benchmarks/*`, and `tests/*` are all entry-points; nothing else imports them.

---

## 2. Data Flow Map — Lifecycle of a Memory

### Store Path

| Step | File | Function / Line | Details |
|------|------|----------------|---------|
| 1. `store()` call | `arn_v9/plugin.py` | `ARNPlugin.store()` — **L125** | Validates `time_context`, builds `ctx` dict, forwards to `ARNv9.perceive()`. |
| 2. `perceive()` | `arn_v9/core/cognitive.py` | `ARNv9.perceive()` — **L687** | Decays working memory → encodes text → computes prediction error → domain-column routing → contradiction check → stores episode → updates working memory. |
| 3. `EmbeddingEngine.encode()` | `arn_v9/core/embeddings.py` | `EmbeddingEngine.encode()` — **L291** | Applies query/passage prefix, checks LRU cache, calls `SentenceTransformer.encode()` (or hash fallback), normalizes vector. |
| 4. `StorageEngine.store_episode()` | `arn_v9/storage/persistence.py` | `StorageEngine.store_episode()` — **L328** | Acquires `self._lock` (**L336**), computes `content_hash`, allocates `vec_index` via `MAX(vec_index)+1`, writes vector to memmap, inserts SQLite row. |
| 5. SQLite INSERT + memmap write | `arn_v9/storage/persistence.py` | `INSERT INTO episodes …` — **L365**;<br>`self._episodic_vectors[vec_index] = vector` — **L358**;<br>`conn.commit()` — **L384** | Metadata goes to SQLite; vector goes to the numpy memmap array backed by `episodic_vectors.npy`. |

### Recall Path

| Step | File | Function / Line | Details |
|------|------|----------------|---------|
| 6. `recall()` | `arn_v9/core/cognitive.py` | `ARNv9.recall()` — **L805** | Encodes query, fetches episodic and/or semantic vectors, scores by cosine similarity. |
| 7. Vector search | `arn_v9/core/cognitive.py` | `query_vector = self.embedder.encode(query, mode='query')` — **L819**;<br>`ep_vectors, ep_ids = self.storage.get_episode_vectors(ep_ids_list)` — **L838**;<br>`similarities = ep_vectors @ query_vector` — **L841**;<br>`sem_vectors, sem_ids = self.storage.get_semantic_vectors()` — **L926**;<br>`similarities = sem_vectors @ query_vector` — **L928** | Brute-force dot-product against all active (non-superseded) episode vectors and all semantic vectors. |
| 8. Result formatting | `arn_v9/core/cognitive.py` | Score blending (recency, importance, supersession penalty) — **L875–L919**;<br>Confidence tier tagging — **L962–L966** | Raw similarities are blended with recency/access frequency, then tagged with `confidence_tier` and `calibrated_confidence`. |
| 9. Context injection | `arn_v9/plugin.py` | `ARNPlugin.recall()` simplification — **L204–L292**;<br>`ARNPlugin.get_context_window()` — **L322–L397** | `recall()` simplifies dicts for agents; `get_context_window()` formats working-memory + long-term results into a markdown string suitable for LLM prompt injection. Also used by `memory_llm.py` `build_memory_system_prompt()` — **L152** for the `MemoryAugmentedLLM` wrapper. |

**Important Sub-Paths**
- **Contradiction detection** (best-effort, swallowed on failure): `ARNv9.perceive()` **L752** → `ContradictionDetector.check()` **L207** (`core/contradictions.py`) → `ClaimExtractor.extract()` **L120**.
- **Consolidation** (episodic → semantic): `ARNv9.consolidate()` **L970** → `ConsolidationEngine.consolidate()` **L268** → clustering (**L429**), contradiction detection (**L516**), `storage.store_semantic()` (**L495**), `storage.mark_episodes_consolidated()` (**L462**).
- **Working memory update**: `ARNv9.perceive()` **L777** → `WorkingMemory.add()` **L163**.

---

## 3. Exception Handling Audit

**Scope:** Every `try/except` block in the Python codebase. `try/finally` (no `except`) blocks are noted as cleanup-only and excluded from risk scoring.

| File | Line(s) | Exception Type(s) Caught | Handling | Risk |
|------|---------|--------------------------|----------|------|
| `storage/persistence.py` | 108–119 | `Exception` | `pass` (silent) | **HIGH** — Corrupted existing vector file on startup is ignored; engine may overwrite with zeros. |
| `storage/persistence.py` | 261–263 | `Exception` | `pass` (silent) | **MEDIUM** — Migration `ALTER TABLE` failures swallowed. Assumes column already exists, but could mask real schema errors. |
| `storage/persistence.py` | 285–289 | `Exception` | `pass` (silent) | **MEDIUM** — v2→v3 `ALTER TABLE` failure swallowed. Same rationale as above. |
| `core/embeddings.py` | 152–203 | `ImportError` | Sets `_model = None`, logs `CRITICAL` | LOW — Graceful degradation to hash fallback. |
| `core/embeddings.py` | 152–212 | `Exception` | Sets `_model = None`, logs `CRITICAL` | LOW — Catches any other model-load failure. |
| `core/embeddings.py` | 241–254 | `OSError` | Returns original `model_name` string | LOW — Falls back to letting `SentenceTransformer` resolve via network. |
| `core/cognitive.py` | 752–755 | `Exception` | `pass` (silent) | **HIGH** — Contradiction detection is completely silenced. If it breaks, the user gets no signal. |
| `core/cognitive.py` | 770–773 | `Exception` | `pass` (silent) | **HIGH** — Supersession of old contradictory episodes is silently abandoned on any error. |
| `core/contradictions.py` | 362–367 | `Exception` | Logs `warning` | LOW — Logged, not re-raised; caller (`cognitive.py`) also swallows it. |
| `extensions.py` | 58–61 | `ImportError` | Logs `info`, returns | LOW — Optional `rank_bm25` dependency. |
| `extensions.py` | 216–220 | `json.JSONDecodeError`, `TypeError` | `pass` (silent) | **MEDIUM** — Bad JSON in `context_json` skipped during tag-based forgetting. Function is dead code anyway. |
| `extensions.py` | 289–293 | `ImportError`, `OSError` | Logs `info` | LOW — Optional `spacy` dependency. |
| `extensions.py` | 474–478 | `Exception` | `pass` (silent) | **MEDIUM** — Duplicate-check skipped during JSON import. Duplicate episodes may be re-imported. |
| `extensions.py` | 520–522 | `Exception` | Logs `error` | LOW — Callback fire failure is logged. |
| `plugin.py` | 95–103 | `Exception` | Sets extensions to `None`, disables dedup | LOW — Graceful fallback if `extensions.py` fails to import. |
| `plugin.py` | 114–118 | `Exception` | `pass` (silent) | **MEDIUM** — BM25 index rebuild fails silently on startup. |
| `memory_llm.py` | 364–368 | `Exception` | Logs `error`, returns `"[Error: ...]"` | LOW — LLM generation failure surfaces an error string to the caller. |
| `memory_llm.py` | 390–393 | `Exception` | `pass` (silent) | **MEDIUM** — Periodic maintenance (`maintain()`) fails silently in chat loop. |
| `api/server.py` | 624–630 | `Exception` (global handler) | Logs `error`, returns JSON 500 | LOW — Centralized FastAPI fallback. |
| `scripts/arn_cli.py` | 253–260 | `Exception` | Increments `skipped` counter | LOW — Individual import entry skipped. |
| `scripts/arn_cli.py` | 407–410 | `ImportError` | Prints missing, calls `sys.exit(1)` | LOW — Fatal on missing critical dependency. |
| `scripts/arn_cli.py` | 414–417 | `ImportError` | Prints missing, calls `sys.exit(1)` | LOW — Fatal on missing `sentence-transformers`. |
| `scripts/arn_cli.py` | 429–432 | `ImportError` | Prints optional not installed | LOW — Optional dependency check. |
| `scripts/arn_cli.py` | 475–488 | `Exception` | Prints warning | LOW — Model pre-download issue warned. |
| `scripts/arn_cli.py` | 494–520 | `Exception` | Prints test-failure warning | LOW — Setup round-trip test warned. |
| `tests/test_all.py` | 52–62 | `Exception` | Returns `False` | LOW — Embedding availability probe. |
| `tests/test_all.py` | 739–742 | `Exception` | Logs failure, prints traceback | LOW — Test runner catches unexpected test crashes. |
| `tests/test_stress_strain.py` | 155–171 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 178–204 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 211–222 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 229–242 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 249–264 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 271–278 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 285–309 | `Exception` (outer) | `reporter.fail()` | LOW — Concurrent-store outer harness. |
| `tests/test_stress_strain.py` | 290–293 | `Exception` (inner worker) | Appends to `errors` list | LOW — Per-thread error captured and reported by outer block. |
| `tests/test_stress_strain.py` | 316–335 | `Exception` (outer) | `reporter.fail()` | LOW — Edge-case outer harness. |
| `tests/test_stress_strain.py` | 330–332 | `Exception` (inner) | `reporter.warn()` | LOW — Individual edge case warned. |
| `tests/test_stress_strain.py` | 342–364 | `KeyError`, `Exception` | `reporter.fail()` | LOW — Missing stat key or general failure captured. |
| `tests/test_stress_strain.py` | 371–382 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 389–413 | `Exception` | `reporter.fail()` | LOW — Test harness records failure. |
| `tests/test_stress_strain.py` | 420–448 | `Exception` | `reporter.ok("skipped...")` | LOW — API server not running; skips gracefully. |
| `tests/check_env.py` | 38–41 | `ImportError` | Prints missing, adds to `errors` | LOW — Environment probe. |
| `tests/check_env.py` | 46–49 | `ImportError` | Prints missing, adds to `errors` | LOW — Environment probe. |
| `tests/check_env.py` | 54–57 | `ImportError` | Prints warning | LOW — Optional package probe. |
| `tests/check_env.py` | 64–67 | `ImportError` | Prints warning | LOW — Optional package probe. |
| `tests/check_env.py` | 74–87 | `Exception` | Prints warning | LOW — Model-load probe. |
| `tests/check_env.py` | 92–95 | `ImportError` | Prints missing, adds to `errors` | LOW — Package import probe. |
| `tests/check_env.py` | 100–109 | `Exception` | Ignored (`pass`) | LOW — Disk-space probe failure is harmless. |

**Swallowed Exceptions (High / Medium Risk Summary)**
1. `storage/persistence.py:108` — Existing vector load failure → silent ignore.
2. `storage/persistence.py:261` & `285` — Schema migration `ALTER TABLE` failures → silent ignore.
3. `core/cognitive.py:752` — Contradiction detector crash → silent ignore.
4. `core/cognitive.py:770` — Supersession operation crash → silent ignore.
5. `extensions.py:216` — Bad JSON during tag-based forget → silent ignore.
6. `extensions.py:474` — Duplicate-check crash during import → silent ignore.
7. `plugin.py:114` — BM25 rebuild crash → silent ignore.
8. `memory_llm.py:390` — Maintenance crash in chat loop → silent ignore.

---

## 4. Dead Code Inventory

**Definition:** Defined functions/classes/methods with zero call sites inside the Python codebase (including tests and scripts).

| File | Line | Name | What It Does | Why It’s Dead |
|------|------|------|--------------|---------------|
| `extensions.py` | 275 | `class EntityExtractor` | spaCy / regex entity extraction (PERSON, ORG, TECH, DATE) | Instantiated at **L351** as `_entity_extractor`, but `extract_entities()` (**L354**) is never called. No other module references it. |
| `extensions.py` | 77 | `HybridRetriever.hybrid_search()` | Fuses BM25 keyword ranks with semantic similarity via Reciprocal Rank Fusion | `HybridRetriever` is instantiated in `plugin.py:99` and `build_bm25_index()` is called, but `hybrid_search()` has **zero** call sites. |
| `extensions.py` | 232 | `apply_importance_decay()` | Ages episode `importance` exponentially by days since creation | Never called. No scheduler or maintenance loop invokes it. |
| `extensions.py` | 177 | `purge_expired()` | Deletes episodes whose `expires_at < now` | Never called. No TTL-reaping cron or maintenance hook exists. |
| `extensions.py` | 166 | `get_expired_episodes()` | Returns IDs of expired episodes | Only called by `purge_expired()` (also dead). |
| `extensions.py` | 501 | `class StoreCallbackManager` | Registry for store-time callbacks | Instantiated in `plugin.py:100`, but `.fire()` (**L517**) is **never** invoked. Consequently `.register()` and `.unregister()` are also unused. |
| `extensions.py` | 189 | `forget_by_query()` | Deletes episodes matching a recall query above a similarity threshold | Never called. The CLI `cmd_forget` in `arn_cli.py` reimplements its own recall-then-delete logic. |
| `extensions.py` | 206 | `forget_by_tag()` | Deletes episodes containing a specific tag in `context_json` | Never called. |
| `extensions.py` | 379 | `get_version_history()` | Walks the `superseded_by` chain for an episode | Never called. |
| `extensions.py` | 409 | `export_memory()` | Exports all episodes + semantics to JSON | Never called. `arn_cli.py:cmd_export()` reimplements export logic inline. |
| `extensions.py` | 446 | `import_memory()` | Imports JSON export, re-vectorizing with the plugin embedder | Never called. `arn_cli.py:cmd_import()` reimplements import logic inline. |
| `storage/persistence.py` | 621 | `StorageEngine.store_entity()` | Inserts / updates an entity and links it to an episode | Never called. The entity tables are created but populated by nobody. |
| `storage/persistence.py` | 658 | `StorageEngine.search_entities()` | Full-text / label search on `entities` table | Never called. |
| `storage/persistence.py` | 677 | `StorageEngine.get_entity_episodes()` | Returns episodes linked to an entity | Never called. |
| `storage/persistence.py` | 688 | `StorageEngine.get_episode_entities()` | Returns entities linked to an episode | Never called. |
| `core/embeddings.py` | 387 | `EmbeddingEngine.batch_similarity()` | Computes dot-product of one query vs. many candidates | Never called. `ARNv9.recall()` uses raw `@` operator inline. |
| `core/embeddings.py` | 462 | `EmbeddingEngine.get_calibrator_stats()` | Returns `SimilarityCalibrator` statistics | Never called. `get_stats()` (**L427**) does not include calibrator data. |
| `api/server.py` | 359 | `check_rate_limit()` | Empty rate-limit middleware stub | Defined but **never** referenced in any route dependency. Actual rate limiting is done inline in each endpoint. |

**Total Dead Functions / Methods:** 18 identified above.  
**Largest Dead Block:** The entire entity-relationship subsystem in `storage/persistence.py` (4 methods, ≈ 80 LOC) and the extensions feature suite (≈ 200 LOC of unused helper functions).

---

## 5. Resource Management Audit

### 5.1 SQLite Connection Lifecycle

- **Factory:** `_ThreadLocalConnection` (`storage/persistence.py` **L32–68**).
- **Per-thread model:** Each thread gets its own `sqlite3.Connection` via `threading.local()` (**L41**). Connections are opened **lazily** on first `get()` call (**L43–55**).
- **Configuration:**
  - `PRAGMA journal_mode=WAL` (**L51**)
  - `PRAGMA synchronous=NORMAL` (**L52**)
  - `PRAGMA cache_size=2000` (**L53**)
  - `check_same_thread=False` (**L49**) — unnecessary because of `threading.local()`, but present.
- **Opening:** `StorageEngine._get_conn()` → `_ThreadLocalConnection.get()`.
- **Closing:** `StorageEngine.close()` → `_ThreadLocalConnection.close()` (**L63–67**). The connection is only closed when `StorageEngine.close()` is invoked (e.g., plugin shutdown, context-manager exit).
- **Shared vs. per-thread:** Strictly per-thread. There is no global shared connection object.
- **WAL checkpointing:** Fully automatic. No manual `PRAGMA wal_checkpoint` calls exist in the codebase. SQLite auto-checkpoints when the WAL file exceeds 1000 pages.

### 5.2 `threading.Lock` Usage

- **Lock instantiation:** `StorageEngine._lock = threading.Lock()` (**L136**).
- **Critical sections:**
  - **`store_episode()`** — lock acquired at **L336** and held for the entire method (hash computation, `MAX(vec_index)` query, memmap write, SQLite `INSERT`, `commit`).
  - **No other method acquires the lock.**
- **Assessment:** The lock is **not** held during a minimal critical section. It serializes all episodic stores, which is correct for preventing `vec_index` collisions, but it does **not** protect semantic stores, reads, or vector expansions from concurrent readers.

### 5.3 Memmap File Handles

- **Files:** `episodic_vectors.npy`, `semantic_vectors.npy`.
- **Opening:** `_init_vectors()` loads them via `np.load(..., mmap_mode='r+')` (**L296–322**).
- **Writing:** Direct assignment to slices: `self._episodic_vectors[vec_index] = vector` (**L358**).
- **Flushing:** `StorageEngine.flush()` calls `self._episodic_vectors.flush()` and `self._semantic_vectors.flush()` (**L780–782**).
- **Closing:** **No explicit `.close()` is called on the memmap arrays.** `StorageEngine.close()` flushes but does not release the file handles. The handles remain open until the `StorageEngine` object is garbage collected.
- **Expansion behavior:** `_expand_episodic_vectors()` creates a new in-memory array, `np.save()` overwrites the file on disk, then `np.load()` re-mmaps the new file (**L751–761**). The old memmap object is dropped; underlying OS file handle may linger until GC.

### 5.4 Other File I/O

- **JSON export / import:** `extensions.py` `export_memory()` opens a file for writing (**L440**); `import_memory()` opens for reading (**L458**). Both use standard `with open(...)`. (Both functions are dead — see §4.)
- **Temp files:** `scripts/memory_editor.py` creates a `NamedTemporaryFile` for editing. It uses `try/finally` to guarantee `os.unlink()` (**L66–82**). No leak risk.
- **Shell redirects:** `core/embeddings.py` `_load_model()` opens `/dev/null` via `os.open()` and manipulates file descriptors to suppress C-level stderr during model load. It restores FDs in a `finally` block (**L182–196**). Correct cleanup.

---

## 6. SQLite Schema Evolution

### Current Schema (v3)

Created by `StorageEngine._init_db()` (`storage/persistence.py` **L141–247**).

| Table | Columns (relevant) |
|-------|-------------------|
| `schema_version` | `version INTEGER PRIMARY KEY` |
| `episodes` | `id`, `vec_index`, `content`, `content_hash`, `context_json`, `importance`, `prediction_error`, `access_count`, `replay_priority`, `created_at`, `last_accessed`, `consolidated`, `source`, `expires_at`, `superseded_by`, `invalidated_at`, `user_id`, `memory_type` |
| `semantic_nodes` | `id`, `vec_index`, `concept_label`, `confidence`, `evidence_count`, `contradiction_log`, `schema_json`, `created_at`, `last_updated`, `access_count` |
| `system_state` | `key TEXT PRIMARY KEY`, `value TEXT` |
| `entities` | `id`, `text`, `label`, `first_seen`, `last_seen`, `mention_count`, `user_id` |
| `entity_episodes` | `entity_id`, `episode_id` (composite PK) |

**Indexes:** `idx_episodes_importance`, `idx_episodes_created`, `idx_episodes_consolidated`, `idx_episodes_hash`, `idx_episodes_expires`, `idx_episodes_user`, `idx_episodes_memory_type`, `idx_semantic_confidence`, `idx_entity_text`, `idx_entity_label`.

### v1 → v2 Migration

**Code:** `storage/persistence.py` **L251–281** (`_migrate_schema`).

- Adds columns to `episodes`:
  - `content_hash TEXT`
  - `expires_at REAL`
  - `superseded_by INTEGER`
  - `invalidated_at REAL`
  - `user_id TEXT`
- Backfills `content_hash` for existing rows using SHA-256 of normalized content (**L267–271**).
- Creates indexes `idx_episodes_hash`, `idx_episodes_expires`, `idx_episodes_user`.

### v2 → v3 Migration

**Code:** `storage/persistence.py` **L283–290**.

- Adds column to `episodes`:
  - `memory_type TEXT DEFAULT 'episode'`
- Creates index `idx_episodes_memory_type`.

### Migration Code Path & Testing

- **Trigger:** `_init_db()` reads `schema_version.version` (**L154**). If `< SCHEMA_VERSION` (3), it calls `_migrate_schema()` and bumps the version.
- **Resilience:** Each `ALTER TABLE` is wrapped in `try/except Exception: pass` (**L261–263**, **L285–289**). If a column already exists, the error is swallowed.
- **Test Coverage:** **No test exercises the migration path.** `tests/test_all.py` `test_persistence()` creates a fresh `StorageEngine` in a temp directory every time. There is no test that:
  - Creates a v1 or v2 database file and asserts the migration succeeds.
  - Validates that backfilled `content_hash` values are correct.
  - Verifies idempotency (re-running `_init_db()` on an already-migrated DB).

---

## 7. Vector Memmap Patterns

### 7.1 Initial Allocation Sizes

| Store | Default Capacity | Shape | File |
|-------|-----------------|-------|------|
| Episodic | 4,096 episodes | `(4096, embedding_dim)` float32 | `episodic_vectors.npy` |
| Semantic | 2,048 nodes | `(2048, embedding_dim)` float32 | `semantic_vectors.npy` |

**Source:** `StorageEngine.__init__()` defaults (**L88–98**); `_init_vectors()` zero-initialization (**L301–304**, **L316–319**).

### 7.2 Expansion Trigger

- **Episodic:** In `store_episode()` (**L328**), `vec_index` is allocated via `MAX(vec_index)+1` (**L349**). If this index exceeds the current memmap shape, `_expand_episodic_vectors()` is called (**L361**).
- **Semantic:** In `store_semantic()` (**L495**), `vec_index` equals `COUNT(*)` of semantic nodes (**L502**). If `vec_index >= shape[0]`, `_expand_semantic_vectors()` is called (**L506**).

### 7.3 Expansion Mechanism

**Episodic expansion** (`storage/persistence.py` **L751–761**):
1. `old_size = self._episodic_vectors.shape[0]`
2. `new_size = old_size * 2`
3. Allocate new zero array `new_vectors` in RAM.
4. `new_vectors[:old_size] = self._episodic_vectors[:]` (copy old data).
5. `np.save(str(self.episodic_vec_path), new_vectors)` — **overwrites** the `.npy` file on disk.
6. `self._episodic_vectors = np.load(..., mmap_mode='r+')` — re-mmaps the new file.

**Semantic expansion** follows the identical pattern (**L763–773**).

### 7.4 Blocking & Thread Safety During Expansion

- **Lock held?** For episodic expansion, `store_episode()` holds `self._lock` (**L336**) for the entire duration, including expansion. So other threads calling `store_episode()` are blocked. Readers (`recall`, `get_episode_vectors`) do **not** acquire the lock, so they may read the old memmap handle while another thread is mid-expansion.
- **For semantic expansion:** `store_semantic()` acquires **no lock**. Two threads could simultaneously trigger expansion or write to the same `vec_index`. This is a **race-condition hazard**.
- **Memmap replacement race:** When `np.save()` overwrites the file, existing readers holding the old memmap FD continue to see the old data (POSIX file semantics). Readers that access `self._episodic_vectors` **after** the reassignment at **L758** will use the new memmap. There is no memory barrier or lock protecting this pointer swap.
- **Flushing:** `flush()` (**L775–782**) calls `.flush()` on both memmaps. This is safe but not atomic with respect to SQLite commits.

### 7.5 Summary of Risks

1. **Semantic store is unprotected** — concurrent semantic stores can corrupt `vec_index` allocation and overwrite each other’s vectors.
2. **Expansion during read** — readers may see stale vectors or a temporarily inconsistent memmap handle during episodic expansion (generally safe on POSIX, but undefined on Windows or network filesystems).
3. **No memmap close** — file handles are leaked until GC. On long-running processes with many expansions, this can accumulate stale FDs.
4. **Expansion doubles size** — growth is exponential (O(n) copies each time), which is acceptable for the 4096→8192→… scale but copies the entire array through RAM on every expansion.

---

*End of report.*
