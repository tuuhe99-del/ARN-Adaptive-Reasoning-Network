"""
Multi-tenancy isolation tests for the ARN API.

The server is advertised as providing "No cross-agent data leakage."
These tests verify that promise at the API boundary: memories stored
under agent_A must never appear in recalls or stats for agent_B.

All tests here require the full FastAPI lifespan (pool + embedding model).
They are skipped automatically in the plumbing tier (no sentence-transformers).
"""

import pytest
from fastapi.testclient import TestClient

from arn_v9.api import server
from arn_v9.api.server import app


def _embeddings_available() -> bool:
    try:
        from arn_v9.core.embeddings import EmbeddingEngine
        return not EmbeddingEngine(use_model=True).is_degraded
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _embeddings_available(),
    reason="sentence-transformers model not available",
)


# ──────────────────────────────────────────────────────────────────────────────
# Recall isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_recalled_results_do_not_cross_agent_boundary(tmp_path, monkeypatch):
    """Memories stored for agent_A must not appear in agent_B's recall."""
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    with TestClient(app) as client:
        server.API_KEY = ""

        stored = client.post(
            "/v1/memory/store",
            json={
                "agent_id": "agent_A",
                "content": "CONFIDENTIAL: agent_A authentication token is abc-xyz-789.",
                "memory_type": "identity",
                "importance": 1.0,
            },
        )
        assert stored.status_code == 200

        recalled = client.post(
            "/v1/memory/recall",
            json={
                "agent_id": "agent_B",
                "query": "authentication token",
                "top_k": 10,
            },
        )
        assert recalled.status_code == 200
        results = recalled.json()["results"]
        assert results == [], (
            "agent_B should not see any of agent_A's memories, "
            f"but got: {[r['content'] for r in results]}"
        )


def test_agent_can_recall_its_own_memory(tmp_path, monkeypatch):
    """Positive control: an agent should find what it stored."""
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    with TestClient(app) as client:
        server.API_KEY = ""

        client.post(
            "/v1/memory/store",
            json={
                "agent_id": "solo_agent",
                "content": "User prefers the Rust programming language.",
                "memory_type": "preference",
                "importance": 0.9,
            },
        )

        recalled = client.post(
            "/v1/memory/recall",
            json={
                "agent_id": "solo_agent",
                "query": "programming language preference",
                "top_k": 5,
            },
        )
        assert recalled.status_code == 200
        results = recalled.json()["results"]
        assert len(results) > 0
        assert any("Rust" in r["content"] for r in results)


def test_multiple_agents_stored_content_stays_isolated(tmp_path, monkeypatch):
    """Three agents each store unique content; each recalls only their own."""
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    agents = {
        "alice": "Alice uses Python for data science work.",
        "bob": "Bob prefers TypeScript for frontend projects.",
        "carol": "Carol specialises in Rust systems programming.",
    }

    with TestClient(app) as client:
        server.API_KEY = ""

        for agent_id, content in agents.items():
            r = client.post(
                "/v1/memory/store",
                json={"agent_id": agent_id, "content": content, "importance": 0.9},
            )
            assert r.status_code == 200

        for agent_id, own_content in agents.items():
            recalled = client.post(
                "/v1/memory/recall",
                json={"agent_id": agent_id, "query": "programming language", "top_k": 10},
            )
            assert recalled.status_code == 200
            contents = [r["content"] for r in recalled.json()["results"]]

            assert any(own_content in c for c in contents), (
                f"{agent_id} should recall its own memory"
            )

            for other_id, other_content in agents.items():
                if other_id == agent_id:
                    continue
                assert not any(other_content in c for c in contents), (
                    f"{agent_id} must not see {other_id}'s memory"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Memory-list isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_list_returns_only_own_agent_memories(tmp_path, monkeypatch):
    """The /v1/memory/list endpoint must scope results to the requesting agent."""
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    with TestClient(app) as client:
        server.API_KEY = ""

        client.post(
            "/v1/memory/store",
            json={"agent_id": "owner_agent", "content": "Owner memory."},
        )

        r = client.post(
            "/v1/memory/list",
            json={"agent_id": "stranger_agent", "limit": 50},
        )
        assert r.status_code == 200
        episodes = r.json().get("episodes", [])
        assert episodes == [], (
            "stranger_agent should have an empty episode list, "
            f"but got {episodes}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Delete isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_delete_agent_does_not_affect_other_agent(tmp_path, monkeypatch):
    """Deleting agent_A's data must leave agent_B's data intact."""
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    with TestClient(app) as client:
        server.API_KEY = ""

        for agent_id in ("keeper", "doomed"):
            client.post(
                "/v1/memory/store",
                json={"agent_id": agent_id, "content": f"Memory for {agent_id}."},
            )

        client.request(
            "DELETE",
            "/v1/memory/agent",
            json={"agent_id": "doomed", "confirm": True},
        )

        recalled = client.post(
            "/v1/memory/recall",
            json={"agent_id": "keeper", "query": "memory", "top_k": 5},
        )
        assert recalled.status_code == 200
        results = recalled.json()["results"]
        assert len(results) > 0, "keeper's memories should survive after doomed was deleted"
