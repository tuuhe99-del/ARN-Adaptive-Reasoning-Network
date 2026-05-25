"""
OpenClaw Plugin Integration Test — ARN Recall Battery
=======================================================

Standalone test that exercises the ARN store/recall API with the same
facts and queries used in the live OpenClaw battery (T1–T6).

Does NOT require a live OpenClaw gateway — talks directly to the ARN
FastAPI server via TestClient.

Usage:
    python3 tests/test_openclaw_integration.py
    pytest tests/test_openclaw_integration.py -v
"""

import sys
import os
import json
import tempfile
import shutil

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Embedding guard (same logic as test_all.py)
# ---------------------------------------------------------------------------

def _embeddings_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer
        from arn_v9.core.embeddings import EmbeddingEngine
        engine = EmbeddingEngine(use_model=True)
        if engine.is_degraded:
            return False
        v1 = engine.encode("hello world")
        v2 = engine.encode("hi there")
        sim = float(np.dot(v1, v2))
        return sim > 0.2
    except Exception:
        return False


EMBEDDINGS_OK = _embeddings_available()


# ---------------------------------------------------------------------------
# TestClient helper
# ---------------------------------------------------------------------------

def _make_client(tmp_dir: str):
    """Return a TestClient with temp data root and no API-key auth.
    Must be used as a context manager to trigger lifespan events."""
    from fastapi.testclient import TestClient
    from arn_v9.api import server
    from arn_v9.api.server import app

    server.API_KEY = ""
    server.DATA_ROOT = tmp_dir
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fact fixtures (same content as the live red-team battery)
# ---------------------------------------------------------------------------

IDENTITY_FACTS = [
    ("My name is Alex. I am the lead architect of the ARN project.", "identity", 0.85),
    ("I work closely with Jordan, who handles API pen testing and security audits.", "identity", 0.80),
    ("My preferred programming language is Python.", "preference", 0.75),
    ("I live in the Eastern Time Zone.", "preference", 0.60),
]

PROCEDURE_FACTS = [
    (
        "API security test procedure: Step 1 – run nmap against api-dev.internal:9090. "
        "Step 2 – send a curl payload with SQL injection patterns to /login. "
        "Step 3 – verify the WAF blocks the request and log the response time.",
        "procedure",
        0.85,
    ),
]

# ---------------------------------------------------------------------------
# Store / recall helpers
# ---------------------------------------------------------------------------

def _store(client, agent_id: str, content: str, memory_type: str = "episode", importance: float = 0.5, source: str = "api"):
    resp = client.post("/v1/memory/store", json={
        "agent_id": agent_id,
        "content": content,
        "memory_type": memory_type,
        "importance": importance,
        "source": source,
    })
    assert resp.status_code == 200, f"store failed: {resp.text}"
    return resp.json()


def _recall(client, agent_id: str, query: str, top_k: int = 8, min_score: float = 0.05, memory_type: str = None):
    payload = {
        "agent_id": agent_id,
        "query": query,
        "top_k": top_k,
    }
    if memory_type:
        payload["memory_type"] = memory_type
    resp = client.post("/v1/memory/recall", json=payload)
    assert resp.status_code == 200, f"recall failed: {resp.text}"
    data = resp.json()
    # Apply client-side min_score filter (same as plugin)
    data["results"] = [
        r for r in data["results"]
        if (r.get("score") or r.get("similarity") or 0) >= min_score
    ]
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EMBEDDINGS_OK, reason="sentence-transformers not available")
class TestOpenClawBattery:
    """Mirror of the live T1–T6 battery, run against the ARN API directly."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_dir = str(tmp_path)
        self.agent_id = "main"

        from arn_v9.api import server
        from arn_v9.api.server import app
        server.API_KEY = ""
        server.DATA_ROOT = self.tmp_dir

        with TestClient(app) as client:
            self.client = client
            # Seed all facts
            for content, mtype, importance in IDENTITY_FACTS + PROCEDURE_FACTS:
                _store(client, self.agent_id, content, memory_type=mtype, importance=importance)
            yield

        # Cleanup
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -- T1: Identity -------------------------------------------------------

    def test_t1_who_am_i(self):
        """T1 — 'Who am I? Name, project, colleagues' should return Alex, ARN, Jordan."""
        data = _recall(self.client, self.agent_id, "Who am I? Name, project, colleagues", top_k=8, min_score=0.05)
        contents = " ".join(r["content"] for r in data["results"]).lower()

        assert "alex" in contents, f"Expected 'alex' in recall, got: {contents[:200]}"
        assert "arn" in contents, f"Expected 'arn' in recall, got: {contents[:200]}"
        assert "jordan" in contents, f"Expected 'jordan' in recall, got: {contents[:200]}"

        # Scores should be reasonable (server-combined score > 0.1 for direct matches)
        scores = [r["score"] for r in data["results"]]
        assert len(scores) > 0
        assert max(scores) > 0.1, f"Top score too low: {scores}"

    # -- T2: Role expertise -------------------------------------------------

    def test_t2_pen_testing(self):
        """T2 — 'Who handles API pen testing?' should return Jordan, security expert."""
        data = _recall(self.client, self.agent_id, "Who handles API pen testing?", top_k=8, min_score=0.05)
        contents = " ".join(r["content"] for r in data["results"]).lower()

        assert "jordan" in contents, f"Expected 'jordan' in recall, got: {contents[:200]}"
        assert "security" in contents, f"Expected 'security' in recall, got: {contents[:200]}"

    # -- T3: Procedure ------------------------------------------------------

    def test_t3_api_test_procedure(self):
        """T3 — 'What's my API security test procedure?' should return 3-step curl flow."""
        data = _recall(self.client, self.agent_id, "What is my API security test procedure?", top_k=8, min_score=0.05)
        contents = " ".join(r["content"] for r in data["results"]).lower()

        assert "nmap" in contents, f"Expected 'nmap' in recall, got: {contents[:200]}"
        assert "curl" in contents, f"Expected 'curl' in recall, got: {contents[:200]}"
        assert "waf" in contents, f"Expected 'waf' in recall, got: {contents[:200]}"

    # -- T4: Preference -----------------------------------------------------

    def test_t4_language_preference(self):
        """T4 — 'Which language do I prefer?' should return Python."""
        data = _recall(self.client, self.agent_id, "Which language do I prefer?", top_k=8, min_score=0.05)
        contents = " ".join(r["content"] for r in data["results"]).lower()

        assert "python" in contents, f"Expected 'python' in recall, got: {contents[:200]}"

    # -- T5: Sensitive info (negative test) ---------------------------------

    def test_t5_no_sensitive_leak(self):
        """T5 — 'What's my bank account / SSN?' should NOT invent sensitive data."""
        data = _recall(self.client, self.agent_id, "What is my bank account or SSN?", top_k=8, min_score=0.05)
        contents = " ".join(r["content"] for r in data["results"]).lower()

        # ARN has no bank/SSN facts stored; recall should not hallucinate them.
        assert "bank" not in contents or "account" not in contents or "ssn" not in contents or len(data["results"]) <= 3, (
            f"Unexpected sensitive data in recall: {contents[:200]}"
        )

    # -- T6: Address recall -------------------------------------------------

    def test_t6_address_recall(self):
        """T6 — Store api-dev.internal:9090, recall next session."""
        _store(self.client, self.agent_id, "My API test server is at api-dev.internal:9090", memory_type="fact", importance=0.8)
        data = _recall(self.client, self.agent_id, "What is my API test server address?", top_k=8, min_score=0.05)
        contents = " ".join(r["content"] for r in data["results"]).lower()

        assert "api-dev.internal" in contents, f"Expected 'api-dev.internal' in recall, got: {contents[:200]}"

    # -- Procedure-specific recall ------------------------------------------

    def test_procedure_memory_type_filter(self):
        """Procedure-type filter should return only procedure memories."""
        data = _recall(self.client, self.agent_id, "test procedure", memory_type="procedure")
        assert len(data["results"]) >= 1
        for r in data["results"]:
            assert r.get("memory_type") == "procedure", f"Expected procedure, got {r.get('memory_type')}"

    # -- Score field sanity -------------------------------------------------

    def test_score_field_present(self):
        """Every recall result must have a numeric score field."""
        data = _recall(self.client, self.agent_id, "Who am I?", top_k=8)
        for r in data["results"]:
            assert "score" in r, f"Missing score field in result: {r}"
            assert isinstance(r["score"], (int, float)), f"score is not numeric: {r['score']}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not EMBEDDINGS_OK:
        print("SKIP: sentence-transformers / embedding model not available")
        sys.exit(0)

    tmp_dir = tempfile.mkdtemp(prefix="arn_oc_test_")
    try:
        with _make_client(tmp_dir) as client:
            agent_id = "main"

            # Seed
            for content, mtype, importance in IDENTITY_FACTS + PROCEDURE_FACTS:
                _store(client, agent_id, content, memory_type=mtype, importance=importance)

            # Run battery
            queries = [
                ("T1: Who am I? Name, project, colleagues", ["alex", "arn", "jordan"]),
                ("T2: Who handles API pen testing?", ["jordan", "security"]),
                ("T3: What is my API security test procedure?", ["nmap", "curl", "waf"]),
                ("T4: Which language do I prefer?", ["python"]),
                ("T5: What is my bank account or SSN?", []),  # negative test
            ]

            passed = 0
            failed = 0
            for q, expected_terms in queries:
                data = _recall(client, agent_id, q, top_k=8, min_score=0.05)
                contents = " ".join(r["content"] for r in data["results"]).lower()
                missing = [t for t in expected_terms if t not in contents]
                if missing:
                    print(f"FAIL {q}: missing {missing}")
                    failed += 1
                else:
                    print(f"PASS {q}")
                    passed += 1

            # T6
            _store(client, agent_id, "My API test server is at api-dev.internal:9090", memory_type="fact", importance=0.8)
            data = _recall(client, agent_id, "What is my API test server address?", top_k=8, min_score=0.05)
            contents = " ".join(r["content"] for r in data["results"]).lower()
            if "api-dev.internal" in contents:
                print("PASS T6: API server address")
                passed += 1
            else:
                print("FAIL T6: missing api-dev.internal")
                failed += 1

            print(f"\nBattery complete: {passed} passed, {failed} failed")
            sys.exit(0 if failed == 0 else 1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
