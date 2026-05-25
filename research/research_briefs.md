# Research Briefs for 7 Specialized Agents

## Agent 1 — Data Integrity & Crash Recovery Specialist

### Area
Long-term data integrity under crash conditions (WAL mode untested under crash)

### Files to Investigate
- arn_v9/storage/persistence.py:249-322 (schema migration + vector loading)
- arn_v9/storage/persistence.py:358-400 (store_episode lock scope)
- arn_v9/storage/persistence.py:740-780 (vector expansion)

### Hypotheses to Test
1. If the process dies during SQLite WAL write, does WAL recovery work on restart?
2. If the process dies during memmap expansion (between np.save and np.load), is the .npy file recoverable?
3. Can episodic_vectors.npy and SQLite become inconsistent (vec_index mismatch)?
4. Is PRAGMA synchronous set appropriately for durability vs performance?

### Acceptance Criteria
- Design at least 2 crash-recovery test scenarios
- Document SQLite WAL checkpoint behavior
- Assess memmap expansion atomicity
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate

---

## Agent 2 — Embedding Model Drift Analyst

### Area
Embedding model stability across upgrades

### Files to Investigate
- arn_v9/core/embeddings.py:1-100 (model loading, tier selection)
- arn_v9/core/embeddings.py:200-250 (dimension detection)
- arn_v9/core/embeddings.py:393-425 (hash fallback)
- arn_v9/storage/persistence.py:161-180 (schema — no model version stored)

### Hypotheses to Test
1. Mixing vectors from different models breaks similarity scores completely
2. content_hash can be used to detect "needs re-embedding" after model change
3. SimilarityCalibrator thresholds are model-specific and need reset on model change
4. There is no schema field tracking which model generated each vector

### Acceptance Criteria
- Document practical impact of model drift
- Design a migration strategy for re-embedding all memories
- Assess whether vector normalization differences matter
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate

---

## Agent 3 — Resource Exhaustion Engineer

### Area
Disk-full and OOM handling

### Files to Investigate
- arn_v9/storage/persistence.py:all (every file write operation)
- arn_v9/core/embeddings.py:50-100 (model memory footprint per tier)
- arn_v9/api/server.py:60-80 (request size limits)

### Hypotheses to Test
1. SQLite INSERT fails gracefully when disk is full
2. NumPy memmap expansion corrupts or crashes when disk is full
3. FastAPI request body size is unbounded — can OOM the server
4. No pre-flight disk space checks exist before any write

### Acceptance Criteria
- Document failure modes for disk-full and OOM
- Design graceful degradation strategy
- Calculate resource budgets (episodes/GB, agents/GB RAM)
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate

---

## Agent 4 — Concurrency & CI Forensics Expert

### Area
ThreadLocalConnection bug root cause + CI gap

### Files to Investigate
- arn_v9/storage/persistence.py:100-150 (_ThreadLocalConnection class)
- arn_v9/storage/persistence.py:250-290 (context manager exit)
- .github/workflows/tests.yml (CI pipeline)
- arn_v9/tests/test_all.py (test list)
- arn_v9/tests/test_stress_strain.py (stress tests)

### Hypotheses to Test
1. The _ThreadLocalConnection __exit__ bug still exists in some code path
2. test_stress_strain.py is excluded from CI due to duration
3. CI only runs a subset of tests — regressions in skipped tests go unnoticed
4. No CI job exists for Node.js plugin tests

### Acceptance Criteria
- Root cause analysis of connection lifecycle bug
- Exact list of which tests run in CI vs which don't
- Concrete proposal for CI improvements
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate

---

## Agent 5 — Plugin Integration Test Engineer

### Area
OpenClaw plugin hooks (most critical gap — completely untested)

### Files to Investigate
- openclaw-arn-plugin/index.js:all (all 7 hooks)
- openclaw-arn-plugin/openclaw.plugin.json (manifest)
- arn_v9/api/server.py:all (API surface the plugin calls)

### Hypotheses to Test
1. session_start persona preload works with mocked ARN API
2. message_received deduplication prevents duplicate stores
3. before_prompt_build fallback store captures messages when other hooks miss
4. before_prompt_build context injection returns properly formatted memories
5. Error handling silently swallows all failures — no alerting

### Acceptance Criteria
- Test scenario design for each of the 7 hooks
- Mock test harness design (no live OpenClaw gateway needed)
- Error handling audit
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate

---

## Agent 6 — API Security Auditor

### Area
Authentication and API exposure risks

### Files to Investigate
- arn_v9/api/server.py:40-60 (verify_api_key)
- arn_v9/api/server.py:200-250 (AgentPool — agent ID validation)
- arn_v9/api/server.py:270-290 (RateLimiter)
- arn_v9/api/server.py:1-30 (server startup, binding)

### Hypotheses to Test
1. Agent ID validation is weak — pattern allows traversal but filesystem isolates
2. Rate limiter can be bypassed by creating new agent IDs
3. No request body size limit — can OOM with huge content
4. API binds to 0.0.0.0 by default — network exposed
5. No CORS restrictions — any origin can call the API

### Acceptance Criteria
- Vulnerability assessment
- Security hardening plan for network-exposed deployment
- Deployment security guide
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate

---

## Agent 7 — Concurrent Memmap Expansion Engineer

### Area
Vector memmap expansion under concurrent load

### Files to Investigate
- arn_v9/storage/persistence.py:740-780 (_expand_episodic_vectors)
- arn_v9/storage/persistence.py:780-820 (_expand_semantic_vectors)
- arn_v9/storage/persistence.py:320-400 (store_episode lock scope)
- arn_v9/tests/test_stress_strain.py:200-300 (concurrent test)

### Hypotheses to Test
1. The concurrent test (10 threads x 50 stores) hit capacity ceiling, not expansion boundary
2. memmap expansion is not atomic — crash during expansion corrupts file
3. Lock scope is too broad — holds lock during SQLite write + vector allocation
4. Pre-allocating larger initial sizes would reduce expansion frequency

### Acceptance Criteria
- Expansion behavior analysis
- Concurrent test design that hits expansion boundary
- Fix strategies (RCU, pre-allocation, lock narrowing)
- Risk rating: Critical/High/Medium/Low
- Effort to fix: hours estimate
