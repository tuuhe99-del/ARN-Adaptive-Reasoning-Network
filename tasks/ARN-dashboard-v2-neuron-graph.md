# ARN Dashboard v2 — Neuron Graph, Bug Fixes & Packaging

## Task ID
`ARN-dashboard-v2-neuron-graph`

## Priority
Critical — this is the user-facing surface of ARN. If the dashboard is broken or
the Relations tab doesn't match the "neurons firing and wiring" vision, nothing
downstream matters.

## Review Chain
```
kimi (analysis + build) → claude (plan review + refinement) → codex (verify)
```
Note: The collab system enforces unique agents per chain. Kimi handles both the
codebase extraction phase AND the full implementation. Claude reviews Kimi's approach
and handles any tricky architectural pieces. Codex does final verification.

---

## Background & Evidence

A full live inspection of the ARN dashboard at `http://127.0.0.1:8745/dashboard`
was performed using browser automation. The following bugs and design gaps were
observed directly in the running system, confirmed against `arn_v9/api/server.py`
(DASHBOARD_HTML, ~1440 lines) and `arn_v9/storage/persistence.py`.

### Bug 1 — Relations tab always blank (CRITICAL, patch already applied)
`switchTab('relations')` set `relationsView.style.display = ""` which reverts to
the CSS rule `#relationsView { display: none; }`. Both panels ended up hidden.
**Fix already applied**: changed to `"block"`. Verify this is in the file and
passes the dashboard test.

### Bug 2 — Stats counter mismatch ("0 total" vs "5 episodic")
`total_experiences` is an in-memory counter (`self.total_experiences += 1` in
`cognitive.py` line 784) that resets to zero on every server restart. The 5
episodes exist durably in SQLite (`episodic_count` is a live COUNT query), but
`total_experiences` was never persisted for them. The dashboard labels this
volatile counter "total" — wrong and misleading.
**Files**: `arn_v9/core/cognitive.py` (lines 652, 784, 992, 1007, 1024),
`arn_v9/api/server.py` (stats endpoint ~line 1340).

### Bug 3 — All memories tagged `source: me`, importance hardcoded at 0.7
The OpenClaw Node.js plugin (`openclaw-arn-plugin/index.js`) tags every message
(both user and agent) as `source: "me"` and hardcodes `importance: 0.7`. This
makes user messages and agent responses indistinguishable in the memory store and
prevents any dynamic importance scoring.

### Bug 4 — Error messages stored as durable memories
The redteam agent's memory contains:
- `"⚠️ Something went wrong while processing your request..."`
- `"⚠️ API provider returned a billing error..."`
- `"You are not authorized to use this command."`

These are transient system errors with zero semantic value. The plugin has no
filter for them. They pollute recall results and distort embedding space.
**File**: `openclaw-arn-plugin/index.js`

### Bug 5 — Node content invisible on unselected cards
Relations tab node cards (white background) show only the type badge and ID.
The `.node-content` text (12px) is invisible because there is no explicit color
on the element and the JPEG compression/rendering makes small near-white text
disappear. Selected/colored cards (blue, amber) make the text readable.
**Fix**: Add `color: var(--text)` explicitly to `.node-content` in DASHBOARD_HTML.

### Bug 6 — memory_links stored but never used in recall
`memory_links` table exists in SQLite (schema v4) and the link CRUD API works.
However `arn_v9/core/cognitive.py` never reads `memory_links` during recall or
scoring. Wiring two memories together has zero effect on what gets retrieved.
**File**: `arn_v9/core/cognitive.py` (recall function, ~line 800+)

### Design Gap — Relations tab is a card grid, not a neuron graph
The user's vision: "dot-like nodes you can connect to each other like connecting
thoughts or memory — like neurons firing and wiring together."
Current implementation: a static card grid with a tiny SVG sidebar showing only
the selected node's neighbors. There are no edge lines between nodes in the main
area, no spatial positioning, no draggable placement.

---

## Scope

### 1. Fix all confirmed bugs
See bugs 1-6 above. Minimum verifiable changes.

### 2. Relations Tab v2 — Interactive Neuron Graph (TWO MODES)

#### Design goal
Full-page SVG/Canvas interactive graph where memories appear as draggable neuron
circles connected by visible curved edges. The user can:
- Drag nodes to any position on the canvas
- Click a node to select it (highlight it and its connections)
- Toggle "Connect" mode, click source → click target → edge drawn instantly
- Click an edge label to delete the link
- Pan the canvas (drag on empty space)
- See the relation type as a label on each edge

#### Version A — High-RAM (≥ 4 GB detected RAM, e.g. MacBook)
- Full SVG graph rendered on the whole page below the toolbar
- Nodes: circles (~52px diameter) with type badge and truncated content label
- Edges: smooth quadratic Bézier curves with animated "pulse" on hover
- Draggable nodes (mousedown + mousemove + mouseup)
- Canvas pan (drag on empty background)
- Node positions persist in `localStorage` keyed by `agent_id + episode_id`
- Selected node and its neighbors glow (drop-shadow filter)
- Connect mode: source node glows amber, draw a dashed preview line to cursor
  until target is clicked

#### Version B — Low-RAM (< 4 GB detected RAM, e.g. Raspberry Pi 5 8 GB is
actually OK — threshold is really < 2 GB for very constrained devices)
- Keep the existing card grid layout
- Add SVG edge-line overlay on top of the grid using absolute positioning
- Edges drawn as straight lines between card centers (no Bézier)
- No drag, no pan, no animations
- Same connect/select interaction as Version A but simpler rendering
- Existing `.node-card` cards remain the primary node representation

Both versions must:
- Show all 5 relation types (relates_to, used_by, part_of, leads_to, contradicts)
  as color-coded edge labels
- Allow deleting a link by clicking × on the edge or in a side panel
- Auto-load nodes when the Relations tab is opened
- Work with vanilla JS, no D3/Cytoscape/React

#### System detection (server-side + client-side)
Server side (`server.py`):
```python
import psutil
RAM_GB = psutil.virtual_memory().total / (1024 ** 3)
GRAPH_MODE = "full" if RAM_GB >= 4.0 else "lite"
```
Pass `GRAPH_MODE` into the dashboard HTML as a JS constant:
```html
<script>const ARN_GRAPH_MODE = "{{ graph_mode }}";</script>
```
The dashboard JS reads `ARN_GRAPH_MODE` and renders accordingly. If `psutil` is
not installed, default to `"lite"`.

### 3. Fix `total_experiences` stat
Option A (preferred): On `ARNBrain` init, if `total_experiences == 0` in persisted
state but `episodic_count > 0`, seed `total_experiences` from
`SELECT COUNT(*) FROM episodes WHERE agent_id=?` to correct the drift.
Option B: Rename the dashboard label from "total" to "lifetime stores" to make
clear it may reset, and show `episodic_count` prominently as "episodes in memory".
Either option is acceptable. Option A preferred for data accuracy.

### 4. OpenClaw plugin fixes (`openclaw-arn-plugin/index.js`)
- Tag `message_received` events as `source: "user"`, `message_sending/sent` as
  `source: "agent"`
- Filter content before storing: skip if content matches any of:
  - Starts with `⚠️`
  - Contains `"Something went wrong"`
  - Contains `"billing error"`
  - Contains `"not authorized"`
  - Content length < 15 characters (noise threshold)
- Derive importance dynamically: `Math.min(0.9, 0.4 + content.length / 800)`
  (longer = more important, capped at 0.9, minimum 0.4)

### 5. Link boosting in recall
In `arn_v9/core/cognitive.py`, after computing the initial ranked recall list:
- For each returned episode, fetch its links via storage
- If any linked episode is also in the top-N recall results, apply a small boost
  (`+0.05`) to both episodes' scores
- Re-sort after boosting
- Keep this optional and off by default (feature flag in config or env var
  `ARN_LINK_BOOST=1`)

### 6. ARN Packaging (`pyproject.toml`)
Add a proper Python package definition so ARN can be installed with `pip install .`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "arn-v9"
version = "0.9.0"
description = "Adaptive Recall Network — local-first persistent memory for AI agents"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "numpy>=1.26",
    "sentence-transformers>=2.7",
    "psutil>=5.9",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "httpx>=0.27"]

[project.scripts]
arn-server = "arn_v9.api.server:main"
arn-cli    = "arn_v9.cli:main"
```

Also add `arn_v9/api/server.py:main()` entry point (uvicorn programmatic start)
and `arn_v9/cli.py:main()` entry point (wrapper around existing CLI).

---

## Agent Responsibilities

### Kimi — Step 1: Analysis + Full Build (Lead Engineer)

Kimi is the lead implementer. Work in two strict phases within one session.

**PHASE 1 — Analysis (read-only, do this first before any edits)**

1. Run `arn collab status` and claim with `arn collab claim --agent kimi`
2. Read and extract the exact relevant code sections from:
   - `arn_v9/api/server.py`: DASHBOARD_HTML block (lines 373–1090 approx),
     stats endpoint (~1333–1352), switchTab JS (lines 867–874), Relations HTML
     (lines 670–693), psutil import (check if present)
   - `arn_v9/core/cognitive.py`: `__init__` total_experiences (line 652),
     `store` (line 784), `recall` method, `get_stats` (line 992),
     `_load_state` (line 1007), `_save_state` (line 1021)
   - `arn_v9/storage/persistence.py`: memory_links schema, `get_links_for_episode`,
     `create_link`, `get_all_links` method signatures
   - `openclaw-arn-plugin/index.js`: all hook handlers that store content
3. Build a private extraction map (in your working notes):
   - Exact line ranges for every section that needs to change
   - Current behavior vs required behavior for each bug
   - Dependencies and import changes required

**PHASE 2 — Implementation (execute all changes)**

Implement in this order (run `python3 -m py_compile arn_v9/api/server.py` after
each section):

   a. OpenClaw plugin: source tag fix (`user`/`agent`), error filter, dynamic importance
   b. `cognitive.py`: total_experiences seed from DB count on init
   c. `server.py`: add psutil import + RAM detection + `GRAPH_MODE` variable
   d. `server.py` DASHBOARD_HTML: inject `ARN_GRAPH_MODE` JS constant, fix
      `.node-content` color, fix stats label
   e. `server.py` DASHBOARD_HTML: Relations tab — replace card grid with
      Version A full SVG neuron graph (high-RAM) and Version B lite overlay
      (low-RAM), conditioned on `ARN_GRAPH_MODE`
   f. `cognitive.py`: optional link-boost in recall (behind `ARN_LINK_BOOST` env)
   g. `pyproject.toml`: new file with package definition + entry points
   h. Tests: update `test_api_dashboard.py` for new assertions

After all edits:
```bash
python3 -m py_compile arn_v9/api/server.py
python3 -m py_compile arn_v9/core/cognitive.py
python3 -m pytest arn_v9/tests/ -x -q
pip install -e . --quiet && echo "package OK"
```

Write handoff with: all changed files, test output, and what to review in Claude's pass.

---

### Claude — Step 2: Plan Review + Refinement

**Primary role: review Kimi's implementation, fix any UX or architectural issues.**

1. Run `arn collab status` and claim with `arn collab claim --agent claude`
2. Read Kimi's handoff carefully — focus on the graph implementation and any gaps
3. Inspect the changed files directly (do not rely solely on the handoff summary)
4. Check specifically:
   - Does the Version A SVG graph match the neuron vision? Nodes as circles,
     edges as Bézier curves, draggable, full-page below toolbar?
   - Does Version B work on a card grid with SVG line overlay?
   - Is `ARN_GRAPH_MODE` correctly injected and branching the right code?
   - Are node content labels readable (not empty, not white-on-white)?
   - Does the connect mode draw a preview line while picking the target?
   - Is the pyproject.toml complete and correct?
5. Fix any concrete issues found. If the graph implementation is incomplete or
   broken, rebuild the missing piece with a full working implementation.
6. Add architectural safety notes in the handoff: what edge cases are unhandled,
   what the next follow-up should be.

**Verification**:
```bash
python3 -m py_compile arn_v9/api/server.py
python3 -m pytest arn_v9/tests/ -x -q
grep -c "ARN_GRAPH_MODE" arn_v9/api/server.py  # should be > 1
grep "color: var(--text)" arn_v9/api/server.py  # should appear in .node-content
```

---

### Codex — Step 3: Verification

**Do NOT edit code unless you find a breaking bug.**

1. Run `arn collab status` and claim with `arn collab claim --agent codex`
2. Read Kimi's build handoff
3. Run the full verification suite:
```bash
# Syntax check
python3 -m py_compile arn_v9/api/server.py
python3 -m py_compile arn_v9/core/cognitive.py
python3 -m py_compile arn_v9/storage/persistence.py

# Unit + integration tests
python3 -m pytest arn_v9/tests/ -v 2>&1 | tail -30

# Collab tests
python3 -m pytest arn_v9/tests/test_collab.py arn_v9/tests/test_collab_runner.py -v

# Dashboard HTML assertions
python3 -m pytest arn_v9/tests/test_api_dashboard.py -v

# Package build
pip install -e . --quiet && arn-server --help 2>&1 | head -5
```
4. Specifically check:
   - The `switchTab` fix uses `"block"` not `""`
   - Version A neuron graph: SVG element exists, draggable nodes, Bézier edges
   - Version B lite graph: card grid intact, SVG overlay for edges
   - `ARN_GRAPH_MODE` JS constant is injected into HTML
   - `.node-content` has explicit `color: var(--text)` 
   - OpenClaw plugin: no `source: "me"` for user messages, no error strings stored
   - `total_experiences` seeds from DB count on init (or label corrected)
   - `pyproject.toml` exists, `pip install -e .` succeeds
   - No secrets or API keys appear anywhere in HTML/JS
5. If any of the above fail, fix the specific issue and re-verify
6. Write final handoff with pass/fail for each check, and a summary of what ARN
   can now do vs what it could do before

---

## Files Expected to Change

| File | What Changes |
|------|-------------|
| `arn_v9/api/server.py` | switchTab fix (verify), node-content color, psutil detection, graph_mode injection, Relations tab v2 (both modes) |
| `arn_v9/core/cognitive.py` | total_experiences seed from DB on init, optional link-boost in recall |
| `arn_v9/storage/persistence.py` | No change expected (links API already complete) |
| `openclaw-arn-plugin/index.js` | source tag fix, error filter, dynamic importance |
| `pyproject.toml` | New file — package definition |
| `arn_v9/tests/test_api_dashboard.py` | Add assertions: graph mode constant present, node-content color, both mode containers exist |
| `arn_v9/tests/test_all.py` | Add test for total_experiences seeding from DB |

## Files Must NOT Change
- `arn_v9/storage/persistence.py` memory_links schema (already correct)
- `arn_v9/tests/test_collab.py` / `test_collab_runner.py` (collab infra)
- `COLLAB.md`, `docs/collab-protocol.md`

---

## Success Criteria

- [ ] `switchTab('relations')` shows the Relations view (block display)
- [ ] Both graph mode containers exist in HTML (`#relNeuronGraph`, `#relCardGrid`)
- [ ] `ARN_GRAPH_MODE` JS constant injected by server based on detected RAM
- [ ] Version A: SVG neuron graph renders draggable circles with edges
- [ ] Version B: card grid renders with SVG line overlay
- [ ] Node content text readable in both modes (unselected state)
- [ ] Connect mode creates a persistent link and draws an edge immediately
- [ ] Link deletion removes edge immediately
- [ ] Node positions persist in localStorage (Version A)
- [ ] OpenClaw plugin stores `source: "user"` for user messages
- [ ] OpenClaw plugin skips ⚠️-prefixed and error-message content
- [ ] Stats "total" label corrected or seeded from DB
- [ ] `pip install -e .` succeeds
- [ ] `python3 -m pytest arn_v9/tests/ -x -q` passes (all existing tests green)
- [ ] No secrets in HTML/JS output

## Verification Commands

```bash
python3 -m py_compile arn_v9/api/server.py
python3 -m py_compile arn_v9/core/cognitive.py
python3 -m pytest arn_v9/tests/ -x -q
python3 -m pytest arn_v9/tests/test_api_dashboard.py -v
pip install -e . --quiet && echo "package install OK"
python3 -c "import psutil; print('psutil ok, RAM:', round(psutil.virtual_memory().total/1e9,1), 'GB')"
```

## Proposed Follow-Up Tasks (if not in this run)

```text
Proposed task:
- Problem: memory_links are stored but never influence recall scoring.
- Evidence: cognitive.py recall function has no reference to memory_links table.
- Files: arn_v9/core/cognitive.py (recall), arn_v9/storage/persistence.py (get_links_for_episode)
- Success criteria: ARN_LINK_BOOST=1 boosts linked episodes in recall results by 0.05.
- Verification: unit test storing two linked episodes and asserting the linked one ranks higher.
- Suggested owner: claude

Proposed task:
- Problem: No Docker container for ARN server.
- Evidence: No Dockerfile in repo.
- Files: Dockerfile, docker-compose.yml
- Success criteria: docker build and docker run serve /dashboard on port 8742.
- Verification: curl -s http://localhost:8742/dashboard | grep "ARN Dashboard"
- Suggested owner: codex
```
