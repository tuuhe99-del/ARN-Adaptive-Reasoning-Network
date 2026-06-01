"""
API key authentication tests.

server.API_KEY is always set (auto-generated on first boot if ARN_API_KEY
is not provided).  When the key is non-empty, every authenticated endpoint
must reject requests that supply a missing or wrong X-API-Key header with
a 401 response and pass requests that supply the correct key.

Test tiers
----------
PLUMBING (always runs, no embedding model needed):
  Rejection tests (401) work without triggering the lifespan because auth
  enforcement happens inside FastAPI's dependency layer before the route
  handler — and pool.get() — are ever reached.

SEMANTIC (skipped if sentence-transformers model unavailable):
  Acceptance tests (200) invoke endpoints that call pool.get(), so the
  full lifespan must run, which requires the embedding model.
"""

import pytest
from fastapi.testclient import TestClient

from arn_v9.api import server
from arn_v9.api.server import app

_TEST_KEY = "test-secret-key-12345"
_WRONG_KEY = "wrong-key-67890"
_LIST_BODY = {"agent_id": "auth-test-agent", "limit": 1}


def _embeddings_available() -> bool:
    try:
        from arn_v9.core.embeddings import EmbeddingEngine
        return not EmbeddingEngine(use_model=True).is_degraded
    except Exception:
        return False


_NEEDS_MODEL = pytest.mark.skipif(
    not _embeddings_available(),
    reason="sentence-transformers model not available",
)


# ──────────────────────────────────────────────────────────────────────────────
# Public endpoints (no auth, no pool — always passes)
# ──────────────────────────────────────────────────────────────────────────────

def test_health_is_public():
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200


def test_dashboard_is_public():
    client = TestClient(app)
    r = client.get("/dashboard")
    assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Missing key → 401  (plumbing — no lifespan needed)
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_key_is_rejected():
    server.API_KEY = _TEST_KEY
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/memory/list", json=_LIST_BODY)
    assert r.status_code == 401


def test_missing_key_on_store_is_rejected():
    server.API_KEY = _TEST_KEY
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(
        "/v1/memory/store",
        json={"agent_id": "auth-test-agent", "content": "Should be rejected."},
    )
    assert r.status_code == 401


def test_missing_key_on_recall_is_rejected():
    server.API_KEY = _TEST_KEY
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(
        "/v1/memory/recall",
        json={"agent_id": "auth-test-agent", "query": "anything", "top_k": 5},
    )
    assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# Wrong key → 401  (plumbing — no lifespan needed)
# ──────────────────────────────────────────────────────────────────────────────

def test_wrong_key_is_rejected():
    server.API_KEY = _TEST_KEY
    client = TestClient(
        app, headers={"X-API-Key": _WRONG_KEY}, raise_server_exceptions=False
    )
    r = client.post("/v1/memory/list", json=_LIST_BODY)
    assert r.status_code == 401


def test_wrong_key_error_detail_is_present():
    server.API_KEY = _TEST_KEY
    client = TestClient(
        app, headers={"X-API-Key": _WRONG_KEY}, raise_server_exceptions=False
    )
    r = client.post("/v1/memory/list", json=_LIST_BODY)
    body = r.json()
    assert "detail" in body
    assert len(body["detail"]) > 0


def test_empty_string_key_is_rejected():
    server.API_KEY = _TEST_KEY
    client = TestClient(
        app, headers={"X-API-Key": ""}, raise_server_exceptions=False
    )
    r = client.post("/v1/memory/list", json=_LIST_BODY)
    assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# Correct key → 200  (semantic — needs pool + embedding model)
# ──────────────────────────────────────────────────────────────────────────────

@_NEEDS_MODEL
def test_correct_key_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))
    server.API_KEY = _TEST_KEY

    with TestClient(app, headers={"X-API-Key": _TEST_KEY}) as client:
        r = client.post("/v1/memory/list", json=_LIST_BODY)
    assert r.status_code == 200


@_NEEDS_MODEL
def test_correct_key_allows_store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))
    server.API_KEY = _TEST_KEY

    with TestClient(app, headers={"X-API-Key": _TEST_KEY}) as client:
        r = client.post(
            "/v1/memory/store",
            json={
                "agent_id": "auth-test-agent",
                "content": "Test memory for auth verification.",
            },
        )
    assert r.status_code == 200
    assert r.json()["stored"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Disabled auth (API_KEY == "") → requests pass through  (semantic)
# ──────────────────────────────────────────────────────────────────────────────

@_NEEDS_MODEL
def test_disabled_auth_allows_any_request(tmp_path, monkeypatch):
    """Setting API_KEY = '' disables enforcement — used throughout other tests."""
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))
    server.API_KEY = ""

    with TestClient(app) as client:
        r = client.post("/v1/memory/list", json=_LIST_BODY)
    assert r.status_code == 200
