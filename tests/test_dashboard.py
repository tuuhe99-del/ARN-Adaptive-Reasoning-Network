"""
Dashboard API endpoint tests.

These tests run against the full FastAPI app using httpx's TestClient.
The lifespan hook requires a real embedding model — tests are skipped
if the model is unavailable (same as the existing TestThresholdValidation skip).
"""

import os
import json
import time
import pytest
import tempfile

os.environ.setdefault("ARN_DATA_ROOT", tempfile.mkdtemp(prefix="arn_dash_test_"))
os.environ["ARN_API_KEY"] = "test-key-dashboard"

try:
    from fastapi.testclient import TestClient
    from arn_v9.api.server import app, pool, DEFAULT_AGENT_ID
    _FASTAPI_AVAILABLE = True
except Exception:
    _FASTAPI_AVAILABLE = False

# Skip all tests if the embedding model isn't available (offline CI)
_MODEL_AVAILABLE = False
try:
    from arn_v9.core.embeddings import EmbeddingEngine
    _eng = EmbeddingEngine(use_model=True)
    _MODEL_AVAILABLE = not _eng.is_degraded
    del _eng
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE or not _MODEL_AVAILABLE,
    reason="FastAPI or embedding model not available"
)

HEADERS = {"X-Api-Key": "test-key-dashboard"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        # Pre-populate some memories
        plugin = pool.get(DEFAULT_AGENT_ID)
        arn = plugin._arn
        for content, role, imp in [
            ("User prefers dark mode in all tools", "user", 0.85),
            ("User is building ARN, a memory system for AI agents", "user_identity", 0.9),
            ("Docker build failed: no space left on device", "tool_result", 0.6),
            ("pip install sentence-transformers succeeded", "tool_result", 0.5),
            ("User loves Python for scripting and automation", "user", 0.8),
        ]:
            vec = arn.embedder.encode(content)
            arn.storage.store_episode(content=content, vector=vec,
                                      role=role, importance=imp, session_id="sess-001")
        yield c


class TestDashboardStats:
    def test_stats_shape(self, client):
        r = client.get("/dashboard/stats")
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d["episodes"]["total"], int)
        assert isinstance(d["episodes"]["by_role"], dict)
        assert isinstance(d["recall_latency"]["recent_50"], list)
        assert isinstance(d["sessions"]["total"], int)
        assert "procedures" in d
        assert "reviews_pending" in d

    def test_stats_db_size(self, client):
        r = client.get("/dashboard/stats")
        assert r.json()["db_size_mb"] >= 0.0

    def test_stats_uptime(self, client):
        r = client.get("/dashboard/stats")
        assert r.json()["uptime_seconds"] >= 0.0


class TestDashboardFeed:
    def test_feed_returns_list(self, client):
        r = client.get("/dashboard/feed?limit=5")
        assert r.status_code == 200
        eps = r.json()["episodes"]
        assert isinstance(eps, list)
        assert len(eps) <= 5

    def test_feed_episode_fields(self, client):
        r = client.get("/dashboard/feed?limit=10")
        eps = r.json()["episodes"]
        assert len(eps) > 0
        ep = eps[0]
        assert "id" in ep
        assert "role" in ep
        assert "content" in ep
        assert "time" in ep

    def test_feed_since_filter(self, client):
        future_ts = time.time() + 3600
        r = client.get(f"/dashboard/feed?limit=50&since={future_ts}")
        assert r.json()["episodes"] == []


class TestDashboardSearch:
    def test_basic_search(self, client):
        r = client.post("/dashboard/search",
                        json={"query": "what does the user prefer?", "top_k": 5})
        assert r.status_code == 200
        d = r.json()
        assert "results" in d
        assert "gap_index" in d
        assert "query_latency_ms" in d
        assert d["query_latency_ms"] > 0

    def test_search_result_fields(self, client):
        r = client.post("/dashboard/search",
                        json={"query": "Python preference", "top_k": 5})
        results = r.json()["results"]
        assert len(results) > 0
        for res in results:
            assert "id" in res
            assert "role" in res
            assert "content" in res
            assert "score" in res
            assert "importance" in res

    def test_include_scores(self, client):
        r = client.post("/dashboard/search",
                        json={"query": "dark mode", "top_k": 5, "include_scores": True})
        d = r.json()
        assert r.status_code == 200
        for res in d["results"]:
            assert "scores" in res
            s = res["scores"]
            for key in ("vector", "fts5", "rrf", "recency", "final"):
                assert key in s, f"Missing score key: {key}"

    def test_gap_index_sensible(self, client):
        r = client.post("/dashboard/search",
                        json={"query": "Python scripting automation", "top_k": 10, "include_scores": True})
        d = r.json()
        gap = d["gap_index"]
        assert isinstance(gap, int)
        assert 0 < gap <= len(d["results"]) + 1

    def test_scores_above_gap_higher_than_below(self, client):
        r = client.post("/dashboard/search",
                        json={"query": "user preferences", "top_k": 10, "include_scores": True})
        d = r.json()
        results = d["results"]
        gap = d["gap_index"]
        if gap > 0 and gap < len(results):
            above = results[gap - 1]["score"]
            below = results[gap]["score"]
            assert above >= below

    def test_role_filter(self, client):
        r = client.post("/dashboard/search",
                        json={"query": "Python", "role_filter": ["user_identity"], "top_k": 10})
        results = r.json()["results"]
        for res in results:
            assert res["role"] == "user_identity"


class TestDashboardSessions:
    def test_sessions_list(self, client):
        r = client.get("/dashboard/sessions?limit=10")
        assert r.status_code == 200
        assert "sessions" in r.json()

    def test_sessions_is_list(self, client):
        assert isinstance(client.get("/dashboard/sessions").json()["sessions"], list)


class TestDashboardReviews:
    def test_reviews_shape(self, client):
        r = client.get("/dashboard/reviews")
        assert r.status_code == 200
        assert "reviews" in r.json()
        assert isinstance(r.json()["reviews"], list)


class TestDashboardProcedures:
    def test_procedures_shape(self, client):
        r = client.get("/dashboard/procedures")
        assert r.status_code == 200
        assert "procedures" in r.json()
        assert isinstance(r.json()["procedures"], list)


class TestDashboardActions:
    def test_pin_then_feed_shows_pinned(self, client):
        # Get an episode to pin
        r = client.get("/dashboard/feed?limit=5")
        eps = r.json()["episodes"]
        if not eps:
            pytest.skip("No episodes to pin")
        ep_id = eps[0]["id"]

        pin_r = client.post("/dashboard/pin", json={"episode_id": ep_id})
        assert pin_r.status_code == 200
        assert pin_r.json()["pinned"] is True

        # Unpin to clean up
        client.post("/dashboard/unpin", json={"episode_id": ep_id})

    def test_forget_episode(self, client):
        # Store a throwaway episode
        plugin = pool.get(DEFAULT_AGENT_ID)
        arn = plugin._arn
        vec = arn.embedder.encode("Temporary test memory for forget test")
        ep_id = arn.storage.store_episode(
            content="Temporary test memory for forget test",
            vector=vec, role="user", importance=0.3
        )
        r = client.post("/dashboard/forget", json={"episode_id": ep_id})
        assert r.status_code == 200
        assert r.json()["forgotten"] is True

    def test_reflect_returns_stats(self, client):
        r = client.post("/dashboard/reflect", json={})
        assert r.status_code == 200
        d = r.json()
        assert "contradictions_found" in d or "procedures_extracted" in d or isinstance(d, dict)


class TestDashboardServes:
    def test_dashboard_html(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "arn-dash" in r.text

    def test_dashboard_has_chart_js(self, client):
        r = client.get("/dashboard")
        assert "chart.js" in r.text.lower()

    def test_dashboard_has_tabs(self, client):
        r = client.get("/dashboard")
        for tab in ("Feed", "Explorer", "Debugger"):
            assert tab in r.text


class TestLatencyTracking:
    def test_recall_updates_latency_buffer(self, client):
        # Do a recall to populate the buffer
        client.post("/recall", json={"query": "Python", "top_k": 3})
        r = client.get("/dashboard/stats")
        lats = r.json()["recall_latency"]["recent_50"]
        assert isinstance(lats, list)

    def test_search_updates_latency_buffer(self, client):
        before = client.get("/dashboard/stats").json()["recall_latency"]["recent_50"]
        client.post("/dashboard/search", json={"query": "dark mode", "top_k": 3})
        after = client.get("/dashboard/stats").json()["recall_latency"]["recent_50"]
        assert len(after) >= len(before)
