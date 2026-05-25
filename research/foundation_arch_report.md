# ARN v9 Foundation Architecture Report

**Generated:** 2026-05-16
**Repo:** `/Users/hustle/arn-v9-repo/`
**Scope:** Intentional design vs. actual implementation, feature matrix, deployment readiness, plugin contract, testing landscape, SPOFs, and scaling assumptions.

---

## 1. Intended Architecture vs. Actual Implementation

### 1.1 What the README Claims

| Claim | Source | Actual State |
|-------|--------|--------------|
| Episodic + semantic split (hippocampus vs. neocortex) | README L15 | Implemented. `ARNv9.perceive()` stores episodes; `ConsolidationEngine.consolidate()` clusters them into semantic nodes. |
| Sentence embeddings (all-MiniLM-L6-v2 default) | README L16 | Implemented. `EmbeddingEngine` supports 5 tiers with query/passage asymmetry. |
| Cortical column voting (8 domain columns) | README L17 | Partial. 8 `DomainColumn` objects exist and compute relevance/surprise, but they do not vote on recall output -- only the best-matching column is updated. The "voting" claim is aspirational. |
| Calibrated prediction error (Welford's algorithm) | README L18 | Implemented. `DomainColumn.update_error_stats()` uses Welford; `is_surprising()` checks `mean + 2*sigma`. |
| Consolidation (sleep-like clustering) | README L19 | Implemented. Online threshold-based clustering in `ConsolidationEngine._cluster_episodes()`. |
| Contradiction detection | README L20 | Partial. Two implementations: (a) `ContradictionDetector` at store-time (claim-extraction heuristics), and (b) `_detect_contradictions()` inside consolidation (word-overlap heuristic). README admits it is heuristic-only. |
| Explicit temporal tagging | README L21 | Implemented. `time_context` stored in episode context; `ARNPlugin.recall()` boosts matching temporal context +/-0.3. |
| "Fits in under 50MB on a Raspberry Pi 5" | README L3 | Verified. Stress tests confirm `< 50MB` total disk for typical workloads; RAM depends on model tier (nano ~90MB). |

### 1.2 Gaps Between Claims and Code

| Gap | Claim Location | Issue |
|-----|---------------|-------|
| **Column voting** | README, docstrings | Only `best_domain` is used; no ensemble voting on recall. The other 7 columns are essentially spectators after `perceive()`. |
| **Auto temporal inference** | README L171 | README admits the system "cannot automatically figure out 'this fact is now outdated' without explicit `time_context`". This is a documented limitation, not a hidden gap. |
| **Multi-modal support** | README L172 | Explicitly admitted as missing. No image/audio embeddings exist. |
| **Cross-agent sharing** | README L168 | Explicitly admitted as missing. Each `agent_id` is fully isolated. |
| **BM25 hybrid search** | `extensions.py` docstring | Advertised but never invoked in production recall paths. `ARNPlugin` initializes `HybridRetriever` but does not use it for `recall()`. |
| **Entity extraction** | `extensions.py` docstring | `EntityExtractor` exists but is never called by `ARNv9.perceive()` or `ARNPlugin.store()`. The `entities` and `entity_episodes` tables in SQLite are write-only from extensions; no production path populates them. |
| **Store callbacks** | `extensions.py` docstring | `StoreCallbackManager` is instantiated in `ARNPlugin.__init__` but never fired -- no code calls `self._callbacks.fire()`. |
| **Memory TTL / expiry** | `extensions.py` docstring | `get_expired_episodes()` and `purge_expired()` exist but are never called by any production path or maintenance routine. The `expires_at` column is stored but never checked. |
| **Importance decay** | `extensions.py` docstring | `apply_importance_decay()` exists but is never called by `maintain()` or any scheduler. |

### 1.3 Dead / Stub Code

| File | Dead Code | Why Unused |
|------|-----------|------------|
| `arn_v9/extensions.py` | `HybridRetriever.hybrid_search()` | `ARNPlugin.recall()` calls `self._arn.recall()` directly; never fuses BM25. |
| `arn_v9/extensions.py` | `StoreCallbackManager.fire()` | No caller in `ARNv9.perceive()` or `ARNPlugin.store()`. |
| `arn_v9/extensions.py` | `EntityExtractor.extract()` | Never invoked; entities table stays empty in normal use. |
| `arn_v9/extensions.py` | `get_expired_episodes()`, `purge_expired()` | No cron, scheduler, or `maintain()` hook calls them. |
| `arn_v9/extensions.py` | `apply_importance_decay()` | Not part of `ARNPlugin.maintain()`. |
| `arn_v9/memory_llm.py` | Referenced in `__init__.py` | Not examined in detail for this report. Appears to be a stub or secondary interface. |

---

## 2. Feature Matrix

| Feature | Implemented In | Tested In | Used In Production Paths | Status |
|---------|---------------|-----------|--------------------------|--------|
| **Episodic memory** | `storage/persistence.py::store_episode()` | `test_all.py` | `ARNv9.perceive()`, API `/v1/memory/store` | Complete |
| **Semantic memory (consolidation)** | `core/cognitive.py::ConsolidationEngine` | `test_all.py` | `ARNv9.consolidate()`, `maintain()` | Complete |
| **Working memory** | `core/cognitive.py::WorkingMemory` | `test_all.py` | `ARNv9.perceive()`, `get_context_window()` | Complete |
| **Contradiction detection** | `core/contradictions.py`, `core/cognitive.py::_detect_contradictions()` | `test_all.py` | `ARNv9.perceive()` (store-time), consolidation | Partial -- heuristic only, no NLI |
| **Entity extraction** | `extensions.py::EntityExtractor` | None | Never called | Dead code |
| **BM25 hybrid search** | `extensions.py::HybridRetriever` | None | Never invoked in recall | Dead code |
| **Importance decay** | `extensions.py::apply_importance_decay()` | None | Never called | Dead code |
| **Memory TTL/expiry** | `extensions.py::purge_expired()` | None | Never called | Dead code |
| **Multi-modal memory** | Not implemented | N/A | N/A | Missing |
| **Temporal reasoning** | `plugin.py::recall()` (re-ranking boost) | `benchmarks/stress_test.py` | `ARNPlugin.recall()`, API `/v1/memory/recall` | Complete (requires explicit tags) |
| **Store callbacks** | `extensions.py::StoreCallbackManager` | None | Never fired | Dead code |
| **Export/import** | `scripts/arn_cli.py::cmd_export/cmd_import` | `test_all.py` (indirect via CLI) | `arn_cli.py` | Complete |
| **Memory editor** | `scripts/memory_editor.py` | None | CLI `memory_editor.py` | Complete (standalone script) |
| **Bootstrap script** | `scripts/bootstrap_agent.py` | None | CLI `bootstrap_agent.py` | Complete (standalone script) |
| **Multi-agent isolation** | `api/server.py::AgentPool`, `plugin.py` | `test_stress_strain.py` | API server, `ARNPlugin` init | Complete |
| **API authentication** | `api/server.py::verify_api_key()` | `.github/workflows/tests.yml` (smoke) | API server (optional via `ARN_API_KEY`) | Complete |
| **Rate limiting** | `api/server.py::RateLimiter` | None | API server (per `agent_id`, 300 RPM default) | Partial -- sliding window is in-memory only; resets on restart |

---

## 3. Deployment Model Analysis

### 3.1 Dependencies & Extras (`pyproject.toml`)

| Aspect | Detail |
|--------|--------|
| Core deps | `numpy>=1.24.0`, `sentence-transformers>=2.2.0` |
| Optional `api` | `fastapi>=0.100.0`, `uvicorn>=0.20.0`, `pydantic>=2.0.0` |
| Optional `dev` | `pytest>=7.0`, `pytest-cov>=4.0` |
| Python support | `>=3.10` (classifiers list 3.10, 3.11, 3.12) |
| Entry point | `arn-cli = arn_v9.scripts.arn_cli:main` |

**Assessment:** Minimal core dependency footprint is appropriate for Pi 5. The `torch>=2.0.0` in `requirements.txt` is not listed in `pyproject.toml` dependencies but is pulled in transitively by `sentence-transformers`.

### 3.2 Dockerfile (`arn_v9/Dockerfile`)

| Aspect | Detail |
|--------|--------|
| Base image | `python:3.12-slim` |
| Build step | Pre-downloads `all-MiniLM-L6-v2` at build time so container starts offline |
| Workers | Hardcoded `--workers 1` because embedding model is ~90MB per process |
| Data volume | `/data` with env `ARN_DATA_ROOT=/data` |
| Healthcheck | HTTP GET to `/v1/health` every 30s |
| Port | `8742` |

**Gaps:**
- No multi-stage build; image includes build artifacts.
- `requirements.txt` is copied but `pyproject.toml` is not used for install.
- No non-root user; container runs as root.
- No `ARN_API_KEY` or `ARN_MAX_AGENTS` documentation in image.

### 3.3 Install Script (`install.sh`)

| Aspect | Detail |
|--------|--------|
| Python check | Requires 3.10+ |
| pip flags | Detects `--break-system-packages` automatically |
| Optional deps | Installs `rank_bm25` silently (ignores failure) |
| Env setup | Writes `ARN_DATA_DIR`, `ARN_EMBEDDING_TIER`, `ARN_AGENT_ID`, alias to `~/.bashrc` and `~/.zshrc` |
| Post-install | Runs `arn_cli.py setup` for model pre-download and round-trip test |

**Gaps:**
- GitHub URL is a placeholder (`YOUR_USERNAME`).
- No verification checksum for downloaded code.
- Modifies user shell RC files without asking.
- `--break-system-packages` is used on PEP-668 systems (risky).

### 3.4 CLI (`arn_v9/scripts/arn_cli.py`)

| Command | Implemented | Notes |
|---------|-------------|-------|
| `setup` | Yes | One-command setup with client integration (codex, claude, kimi, openclaw) |
| `store` | Yes | Maps to `ARNPlugin.store()` |
| `recall` | Yes | Maps to `ARNPlugin.recall()` |
| `context` | Yes | Maps to `ARNPlugin.get_context_window()` |
| `forget` | Yes | Recall-then-delete with similarity threshold |
| `maintain` | Yes | Maps to `ARNPlugin.maintain()` |
| `stats` | Yes | Maps to `ARNPlugin.get_stats()` |
| `export` | Yes | JSON export via `storage.get_all_episodes()` |
| `import` | Yes | JSON import via `plugin.store()` |

**Gaps:**
- `forget` command only deletes episodic memories; semantic nodes are never deleted via CLI.
- No `edit` command in CLI (separate `memory_editor.py` script exists but is not integrated).

### 3.5 CI Pipeline (`.github/workflows/tests.yml`)

| Job | Python Versions | What Runs |
|-----|-----------------|-----------|
| `plumbing-tests` | 3.10, 3.11, 3.12 | Minimal deps (numpy only). Degraded-mode detection. `test_all.py` (semantic tests skip). |
| `full-tests` | 3.11, 3.12 | Full deps with embeddings. `check_env.py`, `test_all.py`, `stress_test.py` (nano tier). |
| `api-smoke` | 3.12 | Installs `[api]` extras. Starts uvicorn, runs health + store/recall round-trip via curl. |

**Gaps:**
- No Docker build test in CI.
- No stress test for `base` or `base-e5` tiers in CI.
- `test_stress_strain.py` is not run in CI.
- No linting, type checking, or coverage gates.
- No test for `arn_cli.py` commands.

### 3.6 Deployment Readiness Summary

| Target | Ready? | Blockers |
|--------|--------|----------|
| **Local dev** | Yes | `pip install -e ".[dev,api]"` works. |
| **Pi 5 (ARM64)** | Yes | Nano tier is Pi-optimized. Dockerfile is AMD64-centric but should build on ARM64 with `python:3.12-slim`. |
| **Docker** | Partial | Works, but lacks non-root user, multi-stage build, and proper requirement freezing. |
| **Cloud** | Partial | Single-process design limits throughput. No horizontal scaling docs. Rate limiter is in-memory (non-distributed). No HTTPS/TLS termination guidance. |

---

## 4. OpenClaw Plugin Contract

### 4.1 Hook-to-Endpoint Mapping

The plugin (`openclaw-arn-plugin/index.js`) registers 7 hooks. Each maps to ARN API endpoints as follows:

| # | OpenClaw Hook | ARN Endpoint | HTTP Method | Purpose |
|---|---------------|--------------|-------------|---------|
| 1 | `session_start` | `/v1/memory/recall` (x3) | POST | Loads static persona: identity, preferences, procedures via `arnRecall()` with `memoryType` filters |
| 2 | `message_received` | `/v1/memory/store` | POST | Stores user message as `source="user"`, `memory_type="episode"` |
| 3 | `message_sent` | `/v1/memory/store` | POST | Stores agent reply as `source="me"`, `memory_type="episode"` |
| 4 | `before_tool_call` | `/v1/memory/store` | POST | Stores tool call as `source="tool:{name}"`, `memory_type="procedure"` |
| 5 | `after_tool_call` | `/v1/memory/store` | POST | Stores tool result as `source="tool_result"`, `memory_type="episode"` |
| 6 | `before_prompt_build` | `/v1/memory/store` (fallback) + `/v1/memory/recall` | POST | Fallback auto-store for missed messages; then recalls dynamic memory for prompt injection |
| 7 | `before_compaction` | `/v1/memory/store` | POST | Stores turn summary as `source="compaction"`, `memory_type="episode"` |

### 4.2 Data Contract

**OpenClaw sends (per hook):**
- `event`: `{ content, body, senderName, from, timestamp, toolName, params, toolCallId, result, error, durationMs, messages, prompt }`
- `ctx`: `{ agentId, sessionKey, sessionId, runId }`

**ARN expects (StoreRequest):**
- `agent_id`: string (derived from `ctx.agentId || ctx.sessionKey || ctx.sessionId || "default"`)
- `content`: string (truncated to 2000 chars for tool results)
- `importance`: float (0.5 default, 0.7 for tool calls)
- `source`: string ("user", "me", "tool:{name}", "tool_result", "compaction")
- `memory_type`: string ("episode" or "procedure")
- `context`: object (metadata like `{ sender, timestamp, tool_name }`)

**ARN expects (RecallRequest):**
- `agent_id`, `query`, `top_k` (default 5), optional `memory_type`, optional `memory_types`

**ARN returns (RecallResult):**
- `content`, `score`, `type` ("episodic"/"semantic"), `similarity`, `confidence_tier`, `calibrated_confidence`, `source`, `age_hours`, `memory_type`

### 4.3 Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| **Network** | `arnFetch()` throws on non-OK status; caller catches and logs via `console.warn()`. |
| **ARN unreachable** | Every hook wraps store/recall in `try/catch`. Failures are warnings, not fatal. The agent continues without memory. |
| **Deduplication** | In-memory `sessionStoredMsgs` Set with DJB2-like hash prevents duplicate stores within a session. |
| **Persona load failure** | `loadStaticPersona()` catches errors and returns empty string; prompt injection continues without static context. |

**Gap:** No retry logic, no exponential backoff, no circuit breaker. If ARN is briefly unreachable, every turn logs a warning but does not queue missed stores for later replay.

### 4.4 Configuration Options

| Config Key | Type | Default | Used By |
|------------|------|---------|---------|
| `arnEndpoint` | string | `http://localhost:8742` | All `arnFetch()` calls |
| `apiKey` | string | `""` | `X-API-Key` header (only if non-empty) |
| `topK` | integer | `5` | `arnRecall()` |
| `minScore` | number | `0.35` | Client-side filter on `calibrated_confidence` |
| `tokenBudget` | integer | `1500` | `formatArnMemories()` max chars |
| `storeMessages` | boolean | `true` | Toggles `message_received` / `message_sent` / `before_prompt_build` stores |
| `storeTools` | boolean | `true` | Toggles `before_tool_call` / `after_tool_call` stores |
| `storeCompaction` | boolean | `true` | Toggles `before_compaction` stores |

**Gap:** `openclaw.plugin.json` schema omits `storeTools` (present in `index.js` but missing from JSON schema).

---

## 5. Test Pyramid

### 5.1 Unit Tests (`arn_v9/tests/test_all.py`)

| Tier | Tests | Requires Model | Coverage |
|------|-------|---------------|----------|
| **Tier 1 (Plumbing)** | Embedding basics, persistence, working memory | No | 16 assertions |
| **Tier 2 (Semantic)** | Embedding quality, prediction error, consolidation, contradiction detection, full integration, agent simulation, stress | Yes | ~28 assertions |

**Total:** ~44 checks. When model is unavailable, 7 test functions skip (16+ semantic assertions not counted).

**Gaps:**
- No test for `extensions.py` features (BM25, callbacks, TTL, decay, entities).
- No test for `ARNPlugin` temporal re-ranking.
- No test for `api/server.py` endpoints.
- No test for `arn_cli.py` commands.
- No test for multi-agent isolation at the API layer.

### 5.2 Stress Tests (`arn_v9/tests/test_stress_strain.py`)

| Test | What It Tests |
|------|--------------|
| Volume 2000 episodes | Storage scalability, recall speed |
| Multi-agent isolation (50 agents) | Cross-agent data leakage |
| Very long messages (10K chars) | Embedding truncation / handling |
| Mixed memory types | Type-filtered recall |
| Contradiction flood (100 pairs) | Store-time contradiction volume |
| Rapid fire (100 stores) | Raw throughput |
| Concurrent stores (10 threads x 50) | Thread safety of `StorageEngine` |
| Edge case content | Empty, emoji, XSS, SQL injection, mixed scripts, very long strings |
| Memory growth tracking | Storage growth linearity |
| Consolidation under load | Maintenance latency |
| Recall accuracy at scale | Needle-in-haystack (1000 random + 50 needles) |
| API server load test | Optional; skipped if server not running |

**Gaps:**
- Not run in CI.
- No automated regression tracking (results are printed, not persisted).
- API load test is opportunistic, not deterministic.

### 5.3 Benchmarks (`arn_v9/benchmarks/`)

| File | Purpose | Run in CI? |
|------|---------|------------|
| `stress_test.py` | 7 adversarial scenarios (cross-session, distractor, contradiction, temporal, hallucination refusal, paraphrase, scale) | Yes (nano tier only) |
| `simulate_agent.py` | 5-day agent simulation with recall accuracy assessment | No |

**Gaps:**
- No benchmark for `base` or `base-e5` tiers in CI.
- No latency/throughput benchmarking framework (just manual `time.time()` calls).
- No memory (RAM) usage benchmarking.

### 5.4 CI Coverage

| What Runs | Where |
|-----------|-------|
| Plumbing tests (no embeddings) | `plumbing-tests` job, Python 3.10-3.12 |
| Full tests + adversarial stress | `full-tests` job, Python 3.11-3.12 |
| API smoke (health + store/recall) | `api-smoke` job, Python 3.12 |

**What is NOT Tested in CI:**
- `test_stress_strain.py`
- `simulate_agent.py`
- Docker build
- `arn_cli.py` commands
- `memory_editor.py`
- `bootstrap_agent.py`
- Multi-tier embedding comparisons
- Import/export round-trip

---

## 6. Single Points of Failure

### 6.1 Embedding Model

| Aspect | Detail |
|--------|--------|
| **Failure mode** | Model fails to download/load (no internet, corrupted cache, missing `sentence-transformers`). |
| **Current mitigation** | `EmbeddingEngine` falls back to `_hash_encode()` -- deterministic lexical hash vectors. `api/server.py` refuses to start if degraded (fail-fast). CLI warns but continues in non-strict mode. |
| **Impact** | Without real embeddings, semantic recall is essentially keyword-based. The system is functional but quality collapses. |
| **Recommended mitigation** | (1) Bundle a minimal ONNX model in the package for offline fallback. (2) Add a healthcheck endpoint that reports degradation so load balancers can drain the instance. (3) Cache model in a volume mount, not ephemeral container layers. |

### 6.2 SQLite Database

| Aspect | Detail |
|--------|--------|
| **Failure mode** | DB corruption (power loss, SD card wear on Pi 5, filesystem bugs). |
| **Current mitigation** | WAL mode (`PRAGMA journal_mode=WAL`) provides crash safety. `PRAGMA synchronous=NORMAL` balances durability and speed. Schema migrations exist (`SCHEMA_VERSION = 3`). |
| **Impact** | Total data loss for the agent. No replication or backup. |
| **Recommended mitigation** | (1) Automated periodic `export_memory()` snapshots to secondary path. (2) SQLite `VACUUM` and integrity check in `maintain()`. (3) For cloud: document migration path to PostgreSQL via SQLAlchemy adapter. |

### 6.3 Memmap Vectors (`*.npy`)

| Aspect | Detail |
|--------|--------|
| **Failure mode** | File truncation or corruption (partial write, disk full, power loss during expansion). |
| **Current mitigation** | Vector expansion uses temp-file-then-rename pattern (`np.save` to file, then `np.load(mmap_mode='r+')`). Atomic at the OS level for single saves. |
| **Impact** | If `.npy` is truncated, `np.load()` will raise `ValueError` on next startup, making the agent unstartable. |
| **Recommended mitigation** | (1) Validate `.npy` shape on load and auto-rebuild from SQLite content hashes if corrupted. (2) Keep a `.npy.backup` before expansion. (3) Add `try/except` around `_init_vectors()` with fallback to zero-initialized array. |

### 6.4 API Server

| Aspect | Detail |
|--------|--------|
| **Failure mode** | Process crash (OOM, unhandled exception, model loading failure). |
| **Current mitigation** | FastAPI global exception handler returns 500 JSON. Healthcheck endpoint for orchestrators. Docker `HEALTHCHECK` every 30s. |
| **Impact** | All agents served by this instance lose access to memory. In-memory rate limiter state resets. |
| **Recommended mitigation** | (1) Run behind a reverse proxy (nginx/traefik) with retry logic. (2) Document Kubernetes deployment with liveness/readiness probes. (3) Consider stateless design: load agent data on-demand instead of caching `ARNPlugin` instances in `AgentPool`. |

### 6.5 OpenClaw Plugin

| Aspect | Detail |
|--------|--------|
| **Failure mode** | ARN API unreachable (network partition, ARN crashed, wrong endpoint config). |
| **Current mitigation** | All hooks catch errors and log warnings. Agent continues without memory injection. |
| **Impact** | Agent operates as if it has no long-term memory. No data loss, but degraded user experience. |
| **Recommended mitigation** | (1) Add retry with exponential backoff in `arnFetch()`. (2) Maintain a small in-memory write-ahead buffer so missed stores can be replayed when ARN returns. (3) Surface degraded-memory status to the user or agent meta-cognition layer. |

---

## 7. Bottleneck Assumptions

### 7.1 Identified Scaling Limiters

| Assumption | Location | Impact | Mitigation Status |
|------------|----------|--------|-------------------|
| **"Single SQLite DB per agent"** | `storage/persistence.py` | Concurrent writes to the same agent block behind SQLite's file lock. On the API server, multiple requests for the same `agent_id` serialize. | No mitigation. Thread-local connections help multi-threading within one process but not across processes. |
| **"Embedding model loaded once per process"** | `api/server.py::AgentPool` | `AgentPool` shares the embedding model across agents *within* one process, but scaling horizontally requires loading it in every worker/container. At ~90MB (nano) to ~500MB (base), this is the primary RAM cost. | Documented. Workers=1 in Dockerfile. No model-sharing mechanism across processes. |
| **"Consolidation runs on main thread"** | `ARNv9.perceive()` L789 | If `auto_consolidate=True` and unconsolidated count >= threshold, `perceive()` blocks until consolidation completes. At 500+ episodes this can take seconds. | README admits this (L169). No background thread or queue implemented. |
| **"Vector memmap doubles on expansion"** | `storage/persistence.py::_expand_episodic_vectors()` | When capacity is exceeded, the `.npy` file doubles in size. For large corpora this causes sudden disk usage spikes and write amplification. | No incremental growth strategy. |
| **"Recall is brute-force O(N)"** | `ARNv9.recall()` | Every recall computes dot product against ALL episodic vectors and ALL semantic vectors. No approximate nearest neighbor (ANN) index. | This is by design for Pi 5 simplicity, but limits scale to ~10K episodes before latency degrades. |
| **"Rate limiter is in-memory"** | `api/server.py::RateLimiter` | Per-agent request counters are stored in a `defaultdict(list)` of timestamps. Restarts clear all limits. No distributed rate limiting. | No mitigation. Acceptable for single-instance deployments. |
| **"AgentPool caches plugins indefinitely"** | `api/server.py::AgentPool` | Plugins are evicted only on LRU overflow (`MAX_AGENTS=100`). Until eviction, the plugin holds open file handles (SQLite + memmap) for every loaded agent. | Could exhaust OS file descriptors or RAM if many agents are active. |

### 7.2 Hidden Assumptions

| Assumption | Risk |
|------------|------|
| **All vectors share the same dimension** | `storage/persistence.py` infers dimension from existing `.npy` files. Switching tiers after data exists requires deleting the data directory. No runtime dimension migration. |
| **Episodic vector indices are monotonic** | `store_episode()` uses `MAX(vec_index)+1`. If the DB is manually edited or restored from backup without vectors, indices desynchronize. |
| **Semantic node `vec_index` is sequential** | `store_semantic()` uses `COUNT(*)` as vec_index. Deleting semantic nodes does not reclaim vector slots, leading to sparse memmap waste. |
| **Content hash is SHA-256 of normalized lowercase text** | Any whitespace-normalization change breaks deduplication backward compatibility. |

---

## Appendix A: File Reference

| File | Role |
|------|------|
| `arn_v9/core/cognitive.py` | Main `ARNv9` class, `WorkingMemory`, `ConsolidationEngine`, `DomainColumn` |
| `arn_v9/core/embeddings.py` | `EmbeddingEngine`, `SimilarityCalibrator`, model registry |
| `arn_v9/core/contradictions.py` | `ContradictionDetector`, `ClaimExtractor` |
| `arn_v9/storage/persistence.py` | `StorageEngine` (SQLite + memmap vectors) |
| `arn_v9/plugin.py` | `ARNPlugin` (OpenClaw-compatible wrapper) |
| `arn_v9/api/server.py` | FastAPI server, `AgentPool`, `RateLimiter` |
| `arn_v9/extensions.py` | BM25, callbacks, TTL, decay, entities, export/import (mostly dead code) |
| `arn_v9/scripts/arn_cli.py` | CLI tool |
| `arn_v9/scripts/bootstrap_agent.py` | Markdown-to-ARN migration |
| `arn_v9/scripts/memory_editor.py` | Human-in-the-loop memory editor |
| `arn_v9/scripts/test_openclaw_simulation.py` | OpenClaw simulation script (not examined) |
| `openclaw-arn-plugin/index.js` | OpenClaw plugin implementation |
| `openclaw-arn-plugin/openclaw.plugin.json` | OpenClaw plugin manifest |
| `arn_v9/tests/test_all.py` | Main test suite (44 checks) |
| `arn_v9/tests/test_stress_strain.py` | Stress/strain test suite (12 tests) |
| `arn_v9/benchmarks/stress_test.py` | 7-scenario adversarial benchmark |
| `arn_v9/benchmarks/simulate_agent.py` | 5-day agent simulation |

---

## Appendix B: Confidence Summary

| Area | Confidence | Notes |
|------|------------|-------|
| Core memory architecture | High | Code is well-structured, docstrings are accurate. |
| Extension features (BM25, callbacks, TTL, decay) | High | They exist but are not wired into production paths. Verified via grep. |
| OpenClaw plugin contract | High | Read both `index.js` and `openclaw.plugin.json` in full. |
| Deployment readiness | High | Read Dockerfile, install.sh, pyproject.toml, CI workflow. |
| Test coverage | High | Read all test and benchmark files. |
| SPOFs and bottlenecks | High | Inferred from code patterns and documented limitations. |
