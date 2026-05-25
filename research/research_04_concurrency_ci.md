# Research Report 4 -- Concurrency & CI Forensics

**Agent:** Concurrency & CI Forensics Expert
**Area:** ThreadLocalConnection bug root cause + CI gap
**Risk Level:** HIGH
**Effort to Fix:** 4-8 hours

---

## Executive Summary

The _ThreadLocalConnection class was previously buggy (connections incorrectly closed in __exit__), but this only affected test code using the `with StorageEngine(...) as s:` context manager. The production server never calls close() on storage, so it was not affected. The real issue is that test_stress_strain.py is completely excluded from CI, and the custom test framework makes it hard to integrate standard tooling.

---

## Root Cause Analysis

### 1. _ThreadLocalConnection Lifecycle (persistence.py:32-61)

```python
class _ThreadLocalConnection:
    def __init__(self, db_path: Path, row_factory=sqlite3.Row):
        self.db_path = db_path
        self.row_factory = row_factory
        self._local = threading.local()
    
    def get(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = self.row_factory
            self._local.conn = conn
        return conn
```

The class creates a new SQLite connection per thread on first access. Connections are never explicitly closed in the server code path. They persist for the lifetime of the thread.

### 2. CI Pipeline Analysis (.github/workflows/tests.yml)

| Job | Tests Run | Tests Skipped |
|-----|-----------|---------------|
| plumbing-tests | test_all.py Tier 1 | Tier 2 |
| full-tests | test_all.py Tier 1+2 + benchmarks/stress_test.py | test_stress_strain.py |
| api-smoke | Health + store/recall curl | Everything else |

### 3. test_stress_strain.py Exclusion

The file `test_stress_strain.py` contains 12 stress tests including:
- Volume: 2000 episodes
- 50-agent isolation
- Concurrent stores (10 threads x 50)
- Contradiction flood (100 pairs)
- Edge cases (XSS, SQL injection, emoji)

**Why excluded:** Duration. The volume test alone takes several minutes. The concurrent test with model loading could take 10+ minutes.

### 4. Custom Test Framework

`test_all.py` uses a hand-rolled test runner, not pytest. This means:
- No pytest fixtures
- No pytest plugins (coverage, xdist, etc.)
- No standard CI integration patterns
- `pytest-cov` is listed in dev dependencies but unused

---

## CI Enhancement Proposal

### Option A: Add Stress Tests to CI (Nightly)
```yaml
stress-tests:
  name: Stress tests
  runs-on: ubuntu-latest
  schedule:
    - cron: '0 3 * * *'  # 3 AM daily
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - name: Cache HuggingFace models
      uses: actions/cache@v4
      with:
        path: ~/.cache/huggingface
        key: hf-cache-${{ runner.os }}-minilm-v1
    - name: Install dependencies
      run: pip install -e ".[dev]"
    - name: Run stress tests
      run: python arn_v9/tests/test_stress_strain.py
      timeout-minutes: 30
```

### Option B: Migrate to pytest
Convert `test_all.py` to use pytest. This enables:
- `pytest-cov` for coverage reporting
- `pytest-xdist` for parallel execution
- Standard CI integration
- Better error reporting

---

## Recommended Fixes

### Fix 1: Add Nightly Stress Test Job (High Priority)
Add a scheduled GitHub Actions job that runs `test_stress_strain.py` once per day.

### Fix 2: Add Plugin Test Job (High Priority)
Add a Node.js test job for the OpenClaw plugin:
```yaml
plugin-tests:
  name: OpenClaw plugin tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with: { node-version: '20' }
    - name: Install plugin deps
      run: cd openclaw-arn-plugin && npm install
    - name: Run plugin tests
      run: cd openclaw-arn-plugin && npm test
```

### Fix 3: Add Docker Build Test (Medium Priority)
```yaml
docker-build:
  name: Docker build
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -f arn_v9/Dockerfile -t arn-v9:test .
    - name: Test container starts
      run: docker run --rm -p 8742:8742 arn-v9:test &
```

### Fix 4: Migrate Tests to pytest (Low Priority, Long-term)
Gradually convert custom test framework to pytest for better tooling integration.
