# ARN Dashboard — Relations & Connections Tab

## Goal

Add a **Relations** tab to the existing ARN local dashboard where the user can see memories as nodes and manually wire them together. This turns the dashboard from a read-only browser into a lightweight graph editor for linking related identity, tool-call/procedure, and fact memories.

## User Need

The user wants to interconnect ARN memories visually:
- **Identity nodes** — "My name is Mohamed", "I work at X"
- **Tool-call / Procedure nodes** — "How to deploy ARN", "OpenClaw setup steps"
- **Fact nodes** — "ARN uses SQLite", "Pi 5 has 8GB RAM"

The user should be able to:
1. Switch to the Relations tab
2. See memories rendered as draggable cards/nodes
3. Click two nodes and "wire" them with a labeled connection (e.g., `relates_to`, `used_by`, `part_of`, `leads_to`)
4. Save those wires persistently
5. Click a node to see its immediate neighborhood

## Current State

- Dashboard exists at `GET /dashboard` in `arn_v9/api/server.py` (inline HTML/CSS/JS)
- Dashboard already has: memory list, search/recall, stats, detail panel, related-memories panel
- No persistent link/edge storage exists in the ARN schema
- SQLite schema in `arn_v9/storage/persistence.py` has `episodes` and `semantic_nodes` tables
- The API has `/v1/memory/list`, `/v1/memory/recall`, `/v1/memory/store`, `/v1/memory/edit`, `/v1/memory/delete`

## Scope

### In Scope

1. **Schema**: Add a `memory_links` table to SQLite:
   - `id INTEGER PRIMARY KEY`
   - `agent_id TEXT`
   - `from_episode_id INTEGER`
   - `to_episode_id INTEGER`
   - `relation_type TEXT` (e.g., `relates_to`, `used_by`, `part_of`, `leads_to`, `contradicts`)
   - `created_at REAL`
   - `confidence REAL DEFAULT 1.0`
   - Unique constraint on `(agent_id, from_episode_id, to_episode_id, relation_type)`

2. **Storage layer**: Add methods to `StorageEngine` in `persistence.py`:
   - `create_link(agent_id, from_id, to_id, relation_type, confidence=1.0)`
   - `get_links_for_episode(agent_id, episode_id)` — returns outgoing + incoming
   - `delete_link(agent_id, link_id)`
   - `get_all_links(agent_id)` — for graph export

3. **API endpoints** in `server.py`:
   - `POST /v1/memory/link` — create a link
   - `POST /v1/memory/unlink` — delete a link
   - `POST /v1/memory/links` — list links for an episode (or all for agent)

4. **Dashboard UI**: Add a "Relations" tab to the existing inline HTML:
   - Node list (same as memory list but rendered as cards)
   - Canvas or DOM-based wire rendering between connected nodes
   - "Connect" mode: select source → select target → pick relation type
   - Neighborhood view: click a node, see directly linked nodes highlighted
   - Keep it dependency-free (vanilla JS, no D3/Cytoscape)

5. **Tests**:
   - Storage tests for link CRUD
   - API tests for link endpoints
   - Dashboard test verifying the Relations tab HTML is present

### Out of Scope

- Automatic link inference (AI-suggested connections)
- Graph layout algorithms (force-directed, etc.) — manual placement or simple grid is fine
- Multi-agent link sharing (links are per `agent_id` for now)
- Link types beyond the predefined set
- Visual graph export (SVG/PNG)

## Agent Split

Per user request:
1. **Kimi**: Analyze codebase, design schema + API + UI plan, write task brief and handoff
2. **Codex**: Review plan, add implementation notes and edge-case handling
3. **Claude**: Review plan, add architectural safety notes and UX refinements
4. **Claude (implementation)**: Implement the approved plan

## Success Criteria

- `POST /v1/memory/link` creates a persistent link between two episodes
- `POST /v1/memory/links` returns links for a given episode
- Dashboard has a visible "Relations" tab
- Relations tab shows nodes and allows connecting two memories
- Links survive server restart (persisted in SQLite)
- Existing dashboard tests still pass
- New link storage tests pass
- No secrets or API keys exposed in new HTML/JS

## Suggested Verification

```bash
python3 -m py_compile arn_v9/api/server.py
python3 -m pytest arn_v9/tests/test_collab.py arn_v9/tests/test_collab_runner.py
python3 -m pytest arn_v9/tests/test_all.py -k link
```

## Files Expected to Change

- `arn_v9/storage/persistence.py` — add `memory_links` table + link CRUD methods
- `arn_v9/api/server.py` — add link API endpoints + Relations tab UI
- `arn_v9/tests/test_all.py` — add link storage tests
- `arn_v9/tests/test_api_dashboard.py` — add Relations tab HTML assertions
- `arn_v9/plugin.py` — optional convenience methods for link creation

## Proposed Relation Types (v1)

```text
relates_to   — general association
used_by      — tool/procedure used in context of identity/fact
part_of      — fact is part of a larger procedure/concept
leads_to     — one step leads to another
contradicts  — explicit contradiction (supplements automatic contradiction detection)
```

## Follow-Up Task (if not in this run)

If automatic link inference or graph layout is not implemented, the handoff must propose:

```text
Proposed task:
- Problem: User wants AI-suggested memory connections and better graph layout.
- Evidence: Manual wiring works but is tedious for large memory sets.
- Files: arn_v9/api/server.py, arn_v9/core/cognitive.py
- Success criteria: Dashboard offers "Suggest links" button using embedding similarity.
- Verification: Manual test + unit test for suggestion API.
- Suggested owner: claude
```
