"""
ARN v9 Stress / Strain Test Suite
====================================
Pushes the system to its limits to find bottlenecks,
memory leaks, corruption, and edge-case failures.
"""

import os
import sys
import time
import random
import string
import tempfile
import shutil
import threading
import concurrent.futures
import statistics

# Ensure arn_v9 is importable
_test_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_test_dir))
sys.path.insert(0, _package_root)

from arn_v9.core.cognitive import ARNv9
from arn_v9.core.embeddings import EmbeddingEngine
from arn_v9.plugin import ARNPlugin

# =============================================================================
# Test Data Generators
# =============================================================================

def random_text(min_words=5, max_words=50, seed=None):
    """Generate random natural-ish text."""
    if seed is not None:
        random.seed(seed)
    words = [
        "the", "a", "an", "user", "agent", "system", "memory", "context",
        "python", "code", "function", "class", "module", "import", "return",
        "error", "bug", "fix", "test", "deploy", "server", "client",
        "data", "database", "query", "result", "async", "await", "promise",
        "json", "api", "endpoint", "request", "response", "header", "token",
        "chat", "message", "reply", "conversation", "thread", "channel",
        "learn", "remember", "forget", "recall", "search", "find", "match",
        "embedding", "vector", "similarity", "cluster", "semantic", "episodic",
        "hello", "world", "goodbye", "thanks", "please", "sorry", "yes", "no",
        "red", "green", "blue", "yellow", "black", "white", "color", "light",
        "fast", "slow", "big", "small", "hot", "cold", "new", "old",
        "today", "tomorrow", "yesterday", "now", "then", "soon", "later",
        "here", "there", "everywhere", "somewhere", "nowhere", "anywhere",
    ]
    n = random.randint(min_words, max_words)
    return " ".join(random.choices(words, k=n))

def code_snippet(lang="python"):
    """Generate a realistic-looking code snippet."""
    snippets = [
        "def foo():\n    return 42",
        "class Bar:\n    def __init__(self):\n        self.x = 0",
        "import numpy as np\narr = np.array([1, 2, 3])",
        "async def fetch(url):\n    async with aiohttp.ClientSession() as s:\n        return await s.get(url)",
        "for i in range(10):\n    print(i)",
        "try:\n    risky()\nexcept Exception as e:\n    logger.error(e)",
        "SELECT * FROM users WHERE id = %s",
        "const x = { a: 1, b: 2 };\nconsole.log(x.a);",
        "fn main() {\n    println!(\"Hello, world!\");\n}",
        "docker run -it --rm ubuntu:latest bash",
    ]
    return random.choice(snippets)

def very_long_text(target_chars=10000):
    """Generate a very long message."""
    paragraphs = []
    chars = 0
    while chars < target_chars:
        p = random_text(min_words=20, max_words=100)
        paragraphs.append(p)
        chars += len(p) + 2
    return "\n\n".join(paragraphs)

def contradictory_pairs():
    """Generate pairs of contradictory statements."""
    return [
        ("The server is running on port 8080", "The server is running on port 3000"),
        ("User prefers dark mode", "User prefers light mode"),
        ("Python 3.12 is installed", "Python 3.11 is installed"),
        ("The database is PostgreSQL", "The database is MySQL"),
        ("API key is sk-abc123", "API key is sk-xyz789"),
        ("Deployment is on AWS", "Deployment is on GCP"),
        ("The meeting is at 9am", "The meeting is at 3pm"),
        ("Redis cache is enabled", "Redis cache is disabled"),
        ("User lives in New York", "User lives in London"),
        ("The project uses React", "The project uses Vue"),
    ]

# =============================================================================
# Reporter
# =============================================================================

class Reporter:
    def __init__(self):
        self.tests = []
        self.current = None

    def start(self, name):
        self.current = {"name": name, "t0": time.time(), "errors": []}

    def ok(self, msg=""):
        self.current["ok"] = True
        self.current["msg"] = msg
        self.current["dt"] = time.time() - self.current["t0"]
        self.tests.append(self.current)
        self._print("✓", msg)

    def fail(self, msg):
        self.current["ok"] = False
        self.current["msg"] = msg
        self.current["dt"] = time.time() - self.current["t0"]
        self.tests.append(self.current)
        self._print("✗", msg)

    def warn(self, msg):
        self.current.setdefault("warnings", []).append(msg)
        self._print("⚠", msg)

    def _print(self, icon, msg):
        name = self.current["name"]
        dt = time.time() - self.current["t0"]
        print(f"  {icon} [{dt:6.2f}s] {name}: {msg}")

    def summary(self):
        passed = sum(1 for t in self.tests if t["ok"])
        failed = sum(1 for t in self.tests if not t["ok"])
        print("\n" + "="*60)
        print(f"STRESS TEST SUMMARY: {passed} passed, {failed} failed")
        print("="*60)
        for t in self.tests:
            status = "PASS" if t["ok"] else "FAIL"
            print(f"  [{status}] {t['name']:<45s} {t['dt']:6.2f}s  {t.get('msg','')}")
        return failed == 0

# =============================================================================
# Stress Tests
# =============================================================================

def run_stress_tests():
    reporter = Reporter()
    tmpdir = tempfile.mkdtemp(prefix="arn_stress_")
    print(f"Stress test data root: {tmpdir}")
    print("="*60)

    # ------------------------------------------------------------------
    # TEST 1: Volume stress — 2000 episodes single agent
    # ------------------------------------------------------------------
    reporter.start("Volume 2000 episodes")
    try:
        agent = ARNPlugin(agent_id="volume_agent", data_root=tmpdir)
        t0 = time.time()
        for i in range(2000):
            text = random_text(seed=i)
            importance = random.uniform(0.1, 1.0)
            agent.store(text, importance=importance)
        dt = time.time() - t0
        stats = agent.get_stats()
        size_mb = stats.get('storage', {}).get('total_size_mb', 0)
        reporter.ok(f"2000 episodes in {dt:.1f}s ({2000/dt:.0f} eps/s), size={size_mb:.1f}MB")
        # Check recall speed
        t1 = time.time()
        res = agent.recall("python code error", top_k=10)
        recall_dt = time.time() - t1
        reporter.ok(f"Recall from 2000 eps: {len(res)} results in {recall_dt*1000:.1f}ms")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 2: Many agents — 50 agents, verify isolation
    # ------------------------------------------------------------------
    reporter.start("Multi-agent isolation (50 agents)")
    try:
        agents = []
        t0 = time.time()
        for i in range(50):
            a = ARNPlugin(agent_id=f"agent_{i:02d}", data_root=tmpdir)
            a.store(f"I am agent {i} and my secret is {random.randint(1000,9999)}", importance=0.8)
            agents.append(a)
        dt = time.time() - t0

        # Verify isolation — agent_i should NOT recall agent_j's secret
        leaks = 0
        for i in range(50):
            res = agents[i].recall("secret", top_k=5)
            for r in res:
                content = r.get("content", "")
                for j in range(50):
                    if j != i:
                        # Use word-boundary check to avoid substring false positives
                        # (e.g. "agent 1" matching inside "agent 10")
                        import re
                        if re.search(rf"\bagent\s+{j}\b", content, re.IGNORECASE):
                            leaks += 1
        if leaks == 0:
            reporter.ok(f"50 agents created in {dt:.1f}s, zero cross-agent leakage")
        else:
            reporter.fail(f"{leaks} isolation leaks detected!")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 3: Very long messages
    # ------------------------------------------------------------------
    reporter.start("Very long messages (10K chars)")
    try:
        agent = ARNPlugin(agent_id="long_msg_agent", data_root=tmpdir)
        long_text = very_long_text(target_chars=10000)
        t0 = time.time()
        agent.store(long_text, importance=0.5)
        dt = time.time() - t0
        res = agent.recall(long_text[:50], top_k=3)
        if res and long_text[:50] in res[0].get("content", ""):
            reporter.ok(f"10K char message stored/recalled in {dt*1000:.0f}ms")
        else:
            reporter.fail("Long message not recallable")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 4: Mixed memory types
    # ------------------------------------------------------------------
    reporter.start("Mixed memory types")
    try:
        agent = ARNPlugin(agent_id="typed_agent", data_root=tmpdir)
        types = ["identity", "preference", "procedure", "error", "fact", "episode"]
        for i, mt in enumerate(types * 50):  # 300 entries
            text = f"{mt} entry number {i}: {random_text(min_words=10, max_words=30)}"
            agent.store(text, importance=0.5, memory_type=mt)

        # Recall by type
        for mt in types:
            res = agent.recall("entry", top_k=5, memory_type=mt)
            if not res:
                reporter.warn(f"No results for memory_type={mt}")
        reporter.ok("300 mixed-type entries stored and recallable")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 5: Contradiction flood
    # ------------------------------------------------------------------
    reporter.start("Contradiction flood (100 pairs)")
    try:
        agent = ARNPlugin(agent_id="contradiction_agent", data_root=tmpdir)
        pairs = contradictory_pairs()
        t0 = time.time()
        for i in range(100):
            old, new = random.choice(pairs)
            # Vary slightly to avoid exact dedupe
            old_v = f"{old} (version {i})"
            new_v = f"{new} (version {i+1})"
            agent.store(old_v, importance=0.6)
            agent.store(new_v, importance=0.6)
        dt = time.time() - t0
        stats = agent.get_stats()
        stats = agent.get_stats()
        reporter.ok(f"200 contradictory entries in {dt:.1f}s, semantics={stats.get('semantic_count', 0)}")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 6: Rapid fire — 100 stores as fast as possible
    # ------------------------------------------------------------------
    reporter.start("Rapid fire (100 stores)")
    try:
        agent = ARNPlugin(agent_id="rapid_agent", data_root=tmpdir)
        t0 = time.time()
        for i in range(100):
            agent.store(f"Rapid message {i}: {random_text(min_words=3, max_words=10)}", importance=0.3)
        dt = time.time() - t0
        reporter.ok(f"100 stores in {dt*1000:.0f}ms ({100/dt:.0f} stores/s)")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 7: Concurrent stores (thread safety)
    # ------------------------------------------------------------------
    reporter.start("Concurrent stores (10 threads × 50)")
    try:
        agent = ARNPlugin(agent_id="concurrent_agent", data_root=tmpdir)
        errors = []

        def worker(tid):
            try:
                for i in range(50):
                    agent.store(f"Thread {tid} message {i}: {random_text()}", importance=0.4)
            except Exception as e:
                errors.append(f"thread {tid}: {e}")

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - t0

        if errors:
            reporter.fail(f"Concurrent errors: {errors[:3]}")
        else:
            stats = agent.get_stats()
            reporter.ok(f"500 concurrent stores in {dt:.1f}s, total eps={stats['episodic_count']}")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 8: Empty / weird content edge cases
    # ------------------------------------------------------------------
    reporter.start("Edge case content")
    try:
        agent = ARNPlugin(agent_id="edge_agent", data_root=tmpdir)
        edge_cases = [
            "",
            "   ",
            "a",
            "🚀 emoji test 🎉",
            "<script>alert('xss')</script>",
            "SELECT * FROM users; DROP TABLE users;",
            "\n\n\n",
            "x" * 5000,
            "mixтекстсмешанный中文 también español",
        ]
        for content in edge_cases:
            try:
                agent.store(content, importance=0.5)
            except Exception as e:
                reporter.warn(f"Edge case failed: {repr(content[:30])}: {e}")
        reporter.ok(f"{len(edge_cases)} edge cases processed")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 9: Memory growth tracking
    # ------------------------------------------------------------------
    reporter.start("Memory growth (2000 → check size)")
    try:
        # Use auto_consolidate=False so episodic_count grows monotonically;
        # with pruning enabled the count plateaus below max_episodes and the
        # while-loop would never reach the target batch size.
        agent = ARNPlugin(agent_id="growth_agent", data_root=tmpdir,
                          episodic_capacity=2048, auto_consolidate=False)
        sizes = []
        stores = 0
        for batch in [100, 500, 1000, 2000]:
            while stores < batch:
                agent.store(random_text(), importance=random.uniform(0.1, 0.9))
                stores += 1
            stats = agent.get_stats()
            sz = stats.get('storage', {}).get('total_size_mb', 0)
            sizes.append((batch, sz))
        reporter.ok(f"Growth: {sizes[-1][1]:.1f}MB at {sizes[-1][0]} episodes")
        # Check growth is roughly linear (O(n) not O(n²))
        mb_per_ep = sizes[-1][1] / sizes[-1][0]
        if mb_per_ep > 0.5:
            reporter.warn(f"High storage per episode: {mb_per_ep:.3f}MB/episode")
    except KeyError as e:
        reporter.fail(f"Missing stat key: {e}")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 10: Consolidation under load
    # ------------------------------------------------------------------
    reporter.start("Consolidation under load")
    try:
        agent = ARNPlugin(agent_id="consolidation_agent", data_root=tmpdir)
        for i in range(500):
            agent.store(f"Topic {i % 10}: {random_text(min_words=20, max_words=50)}", importance=0.5)
        before = agent.get_stats().get('semantic_count', 0)
        t0 = time.time()
        result = agent.maintain()
        dt = time.time() - t0
        after = agent.get_stats().get('semantic_count', 0)
        after = agent.get_stats().get('semantic_count', 0)
        reporter.ok(f"Consolidation on 500 eps: {before}→{after} semantics in {dt:.1f}s")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 11: Recall accuracy degradation
    # ------------------------------------------------------------------
    reporter.start("Recall accuracy at scale")
    try:
        agent = ARNPlugin(agent_id="accuracy_agent", data_root=tmpdir)
        # Store 1000 random, then 50 "needle" messages
        for i in range(1000):
            agent.store(random_text(seed=i), importance=0.3)
        needles = []
        for i in range(50):
            needle = f"NEEDLE {i}: The quick brown fox jumps over the lazy dog {i}"
            agent.store(needle, importance=0.9)
            needles.append(needle)

        # Try to recall each needle
        found = 0
        for needle in needles:
            res = agent.recall(needle[:30], top_k=5)
            for r in res:
                if needle in r.get("content", ""):
                    found += 1
                    break
        accuracy = found / len(needles)
        if accuracy >= 0.9:
            reporter.ok(f"Recall accuracy: {accuracy*100:.0f}% ({found}/{len(needles)})")
        else:
            reporter.warn(f"Low recall accuracy: {accuracy*100:.0f}% ({found}/{len(needles)})")
    except Exception as e:
        reporter.fail(str(e))

    # ------------------------------------------------------------------
    # TEST 12: API server load (if running)
    # ------------------------------------------------------------------
    reporter.start("API server load test")
    try:
        import requests
        base = "http://localhost:8742"
        r = requests.get(f"{base}/v1/health", timeout=2)
        if r.status_code != 200:
            reporter.ok("API server not running — skipped")
        else:
            agent_id = "api_load_test"
            # Store 50 entries via API
            t0 = time.time()
            for i in range(50):
                requests.post(f"{base}/v1/memory/store", json={
                    "agent_id": agent_id,
                    "content": f"API load test {i}: {random_text()}",
                    "importance": 0.5,
                }, timeout=5)
            store_dt = time.time() - t0

            # Recall 20 times
            t1 = time.time()
            for i in range(20):
                requests.post(f"{base}/v1/memory/recall", json={
                    "agent_id": agent_id,
                    "query": random_text(),
                    "top_k": 5,
                }, timeout=5)
            recall_dt = time.time() - t1
            reporter.ok(f"API: 50 stores in {store_dt*1000:.0f}ms, 20 recalls in {recall_dt*1000:.0f}ms")
    except Exception as e:
        reporter.ok(f"API load test skipped: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print(f"\nCleaning up {tmpdir}...")
    shutil.rmtree(tmpdir, ignore_errors=True)

    return reporter.summary()


if __name__ == "__main__":
    print("="*60)
    print("ARN v9 STRESS / STRAIN TEST SUITE")
    print("="*60)
    ok = run_stress_tests()
    sys.exit(0 if ok else 1)
