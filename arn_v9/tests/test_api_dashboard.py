from fastapi.testclient import TestClient

from arn_v9.api import server
from arn_v9.api.server import app


def _with_test_auth(client):
    """Temporarily disable API-key enforcement for test client requests."""
    server.API_KEY = ""
    return client


def test_dashboard_returns_html():
    client = TestClient(app)
    server.API_KEY = ""

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ARN Dashboard" in response.text
    assert "/v1/memory/list" in response.text
    assert '"""' not in response.text
    assert '\\"": "&quot;"' in response.text


def test_relations_tab_present():
    client = TestClient(app)
    server.API_KEY = ""

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "Relations" in response.text
    assert "/v1/memory/link" in response.text
    assert "/v1/memory/links" in response.text
    assert "relationsView" in response.text
    assert "connectToggle" in response.text


def test_graph_mode_injected():
    client = TestClient(app)
    server.API_KEY = ""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "ARN_GRAPH_MODE" in response.text
    assert "relNeuronGraph" in response.text
    assert "relCardGrid" in response.text


def test_node_content_color_explicit():
    client = TestClient(app)
    server.API_KEY = ""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "color: var(--text)" in response.text


def test_memory_link_api_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    with TestClient(app) as client:
        server.API_KEY = ""
        first = client.post(
            "/v1/memory/store",
            json={
                "agent_id": "relations_test",
                "content": "Identity: user prefers local-first memory.",
                "memory_type": "identity",
            },
        )
        second = client.post(
            "/v1/memory/store",
            json={
                "agent_id": "relations_test",
                "content": "Procedure: check disk space before vector expansion.",
                "memory_type": "procedure",
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200

        first_id = first.json()["episode_id"]
        second_id = second.json()["episode_id"]

        link_payload = {
            "agent_id": "relations_test",
            "from_episode_id": first_id,
            "to_episode_id": second_id,
            "relation_type": "used_by",
            "confidence": 0.75,
        }
        created = client.post("/v1/memory/link", json=link_payload)
        duplicate = client.post("/v1/memory/link", json=link_payload)
        assert created.status_code == 200
        assert duplicate.status_code == 200
        assert duplicate.json()["link_id"] == created.json()["link_id"]

        links = client.post(
            "/v1/memory/links",
            json={"agent_id": "relations_test", "episode_id": first_id},
        )
        assert links.status_code == 200
        body = links.json()
        assert body["count"] == 1
        assert body["links"][0]["from_episode_id"] == first_id
        assert body["links"][0]["to_episode_id"] == second_id
        assert body["links"][0]["relation_type"] == "used_by"

        unlinked = client.post(
            "/v1/memory/unlink",
            json={"agent_id": "relations_test", "link_id": created.json()["link_id"]},
        )
        assert unlinked.status_code == 200

        after = client.post(
            "/v1/memory/links",
            json={"agent_id": "relations_test", "episode_id": first_id},
        )
        assert after.status_code == 200
        assert after.json()["count"] == 0


def test_delete_agent_route_not_shadowed_by_episode_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    with TestClient(app) as client:
        server.API_KEY = ""
        stored = client.post(
            "/v1/memory/store",
            json={
                "agent_id": "delete_agent_test",
                "content": "Temporary memory for agent deletion route test.",
            },
        )
        assert stored.status_code == 200

        deleted = client.request(
            "DELETE",
            "/v1/memory/agent",
            json={"agent_id": "delete_agent_test", "confirm": True},
        )

        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "agent_id": "delete_agent_test"}


def test_auto_link_connects_related_live_session_memories(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", str(tmp_path))

    memories = [
        ("My name is Red Team Alpha. I test web applications for security vulnerabilities.", "agent", "identity"),
        ("Target system: internal dashboard at 127.0.0.1:8745. Testing for XSS, open redirects, auth bypass.", "user", "episode"),
        ("Tool call: curl_http - GET http://127.0.0.1:8745/dashboard", "tool:curl_http", "procedure"),
        ("Finding: Dashboard loads without authentication. No login required to view memories.", "agent", "episode"),
    ]

    with TestClient(app) as client:
        server.API_KEY = ""
        for content, source, memory_type in memories:
            stored = client.post(
                "/v1/memory/store",
                json={
                    "agent_id": "auto_link_live_test",
                    "content": content,
                    "source": source,
                    "memory_type": memory_type,
                    "importance": 0.8,
                },
            )
            assert stored.status_code == 200

        links = client.post(
            "/v1/memory/links",
            json={"agent_id": "auto_link_live_test"},
        )

        assert links.status_code == 200
        body = links.json()
        assert body["count"] > 0
        assert any(link["relation_type"] == "relates_to" for link in body["links"])
