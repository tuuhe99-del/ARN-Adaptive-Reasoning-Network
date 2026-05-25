# ARN v9 Research Roadmap -- Synthesis

**Date:** 2026-05-18
**Scope:** 7 research areas across code, architecture, security, and testing
**Foundation Agents:** 2 (Code Archaeologist + Architecture Cartographer)
**Research Agents:** 7 (specialized deep-dive)
**Deliverables:** 11 documents, 1 synthesis

---

## Priority Matrix

| Rank | Area | Risk | Effort | Impact | Fix Key | Owner |
|------|------|------|--------|--------|---------|-------|
| 1 | Plugin Integration (Area 5) | CRITICAL | 12-20h | **Highest** | Add retry + test suite | Research Agent 5 |
| 2 | Data Integrity (Area 1) | HIGH | 8-12h | High | Atomic expansion + recovery | Research Agent 1 |
| 3 | Memmap Concurrency (Area 7) | HIGH | 8-16h | High | Pre-alloc + lock narrowing | Research Agent 7 |
| 4 | CI Gaps (Area 4) | HIGH | 4-8h | High | Nightly stress tests + plugin CI | Research Agent 4 |
| 5 | API Security (Area 6) | MEDIUM | 4-8h | Medium | Harden for network exposure | Research Agent 6 |
| 6 | Resource Exhaustion (Area 3) | MEDIUM | 6-10h | Medium | Disk checks + size limits | Research Agent 3 |
| 7 | Embedding Drift (Area 2) | MEDIUM | 16-24h | Low-Medium | Model version tracking | Research Agent 2 |

---

## Quick Wins (1-2 hours each)

These can be implemented immediately with minimal risk:

1. **Add request size limit** (Area 6)
   - File: `api/server.py`
   - Change: Add `max_length=50000` to `StoreRequest.content`
   - Time: 15 minutes

2. **Fix timing-safe API key comparison** (Area 6)
   - File: `api/server.py`
   - Change: `hmac.compare_digest(x_api_key, API_KEY)`
   - Time: 15 minutes

3. **Add global rate limit** (Area 6)
   - File: `api/server.py`
   - Change: Track `_global_window` in `RateLimiter`
   - Time: 30 minutes

4. **Pre-allocate larger vector sizes** (Area 7)
   - File: `storage/persistence.py`
   - Change: Initial sizes 4096/2048 -> 65536/16384
   - Time: 15 minutes

5. **Fix schema migration error handling** (Area 1)
   - File: `storage/persistence.py`
   - Change: Replace `except: pass` with specific `sqlite3.OperationalError` handling
   - Time: 30 minutes

6. **Add CI nightly stress test job** (Area 4)
   - File: `.github/workflows/tests.yml`
   - Change: Add scheduled job for `test_stress_strain.py`
   - Time: 30 minutes

---

## Medium Projects (4-8 hours each)

1. **Atomic vector expansion** (Area 1)
   - Temp-file + rename pattern
   - Corrupted file recovery
   - Files: `storage/persistence.py`

2. **Plugin retry logic + circuit breaker** (Area 5)
   - Exponential backoff on ARN API failures
   - Failure threshold before skipping calls
   - File: `openclaw-arn-plugin/index.js`

3. **Plugin Jest test suite** (Area 5)
   - Mock ARN server
   - Test all 7 hooks
   - File: `openclaw-arn-plugin/plugin.test.js`

4. **Narrow lock scope in store_episode** (Area 7)
   - Split into allocate/write/commit phases
   - File: `storage/persistence.py`

5. **Add CI plugin test job** (Area 4)
   - Node.js test runner
   - Mock OpenClaw SDK
   - File: `.github/workflows/tests.yml`

---

## Large Projects (12-24 hours each)

1. **Model version tracking + migration** (Area 2)
   - Add `model_version` column to episodes
   - Detect model changes on startup
   - Background re-embedding job
   - Files: `storage/persistence.py`, `core/embeddings.py`

2. **Comprehensive security hardening** (Area 6)
   - CORS middleware
   - Request audit logging
   - TLS guidance documentation
   - Files: `api/server.py`, docs

3. **Lock-free memmap architecture** (Area 7)
   - Double-buffering or RCU pattern
   - Reader-writer separation
   - Files: `storage/persistence.py`

---

## Cross-Cutting Themes

### Theme 1: Silent Failures
Multiple areas share the same anti-pattern: `try/except` blocks that swallow exceptions:
- Schema migrations (Area 1)
- Plugin API calls (Area 5)
- Vector loading (Area 1)

**Recommendation:** Audit every `except` block. Log at `ERROR` level, not `WARN`. Fail fast when possible.

### Theme 2: Dead Code
Multiple features exist but are never called:
- Entity extraction
- BM25 hybrid search
- Memory TTL
- Importance decay
- Store callbacks

**Recommendation:** Either wire them up or remove them. Dead code creates maintenance burden and false confidence.

### Theme 3: Test Gaps
The most critical production paths are untested:
- OpenClaw plugin hooks
- Crash recovery
- Concurrent expansion
- Network-exposed security

**Recommendation:** Prioritize integration tests over unit tests for these paths.

---

## Files Changed Summary

| File | Purpose | Status |
|------|---------|--------|
| `research/foundation_code_report.md` | Complete code analysis | Done |
| `research/foundation_arch_report.md` | Architecture vs reality | Done |
| `research/research_briefs.md` | 7 agent briefs | Done |
| `research/research_01_data_integrity.md` | Crash recovery findings | Done |
| `research/research_02_embedding_drift.md` | Model migration findings | Done |
| `research/research_03_resource_exhaustion.md` | Disk/OOM findings | Done |
| `research/research_04_concurrency_ci.md` | CI gap findings | Done |
| `research/research_05_plugin_integration.md` | Plugin test findings | Done |
| `research/research_06_api_security.md` | Security audit findings | Done |
| `research/research_07_memmap_concurrency.md` | Expansion findings | Done |
| `research/arn_v9_research_roadmap.md` | This file -- synthesis | Done |

---

## Next Steps

1. **Implement Quick Wins** (1 day) -- low risk, immediate value
2. **Implement Medium Projects** (1 week) -- significant reliability improvements
3. **Plan Large Projects** (backlog) -- architectural improvements
4. **Deploy to Pi 5** -- test in production-like environment
5. **Re-run stress tests after fixes** -- validate improvements
