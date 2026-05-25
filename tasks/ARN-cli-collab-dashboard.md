# ARN CLI Collaboration Dashboard

## Goal

Build a CLI-based dashboard for the ARN collaboration system so the user can view agent status, feed information, and dispatch messages to Codex, Claude, and Kimi from a single CLI run.

## User Need

The user wants one command-line interface to:
1. **See the big picture** — current collab state, which agent is active, recent handoffs, pending tasks
2. **Feed information** — broadcast a message, context, or task brief to one or all agents
3. **Talk to agents** — send a prompt to a specific agent and (eventually) see the response
4. **Trigger cycles** — kick off a collaboration run manually without waiting for the scheduled morning/night launchd jobs

This is an **operator console**, not a replacement for the individual agent CLIs.

## Current State

- Collaboration system exists:
  - `arn_v9/collab.py` — state machine, handoffs
  - `arn_v9/collab_runner.py` — serial cycle runner
  - `arn collab` CLI commands: `init`, `status`, `next`, `claim`, `release`, `handoff`, `validate-handoff`
  - `COLLAB.md` and `docs/collab-protocol.md` document the system
- The runner can launch agents but only via `python3 -m arn_v9.collab_runner --execute`
- There is no unified "feed" or "broadcast" mechanism
- There is no human-facing CLI summary view

## Scope

### In Scope

1. **`arn collab dashboard`** — Live console view (similar to `htop` or `docker ps`):
   - Current task ID and status
   - Review chain (e.g., `codex → claude → kimi`)
   - Which agent has the lock, for how long, whether stale
   - Last 3 handoffs with timestamps and status
   - Recent cycle report path
   - Agent binary health (codex, claude, kimi paths exist? auth ok?)
   - Refresh every N seconds or single-shot

2. **`arn collab feed --agent <agent>|all --message "..."`** — Broadcast context:
   - Stores the message as an ARN memory episode under a special `agent_id` (e.g., `collab_hub`)
   - Tags it with `source: human_feed`, `target_agent: codex|claude|kimi|all`
   - The next time that agent claims a task, the runner prompt includes the feed context
   - Optionally writes a lightweight "feed" file in `$ARN_DATA_DIR/collab/feeds/`

3. **`arn collab run --task-id <id>`** — Manual cycle trigger:
   - Wrapper around `collab_runner.py` that sets sensible defaults
   - Shows live progress (claimed → running → handoff → done)
   - Returns exit code 0 if cycle completes, 2 if blocked

4. **`arn collab agents`** — Agent health check:
   - Shows binary paths, versions, auth status
   - Kimi OAuth expiry check (reuse `kimi_auth_status()` from `collab_runner.py`)
   - Codex/Claude path existence checks

5. **`arn collab history`** — Show recent handoffs:
   - Lists last N handoff files
   - Shows agent, status, timestamp, one-line summary
   - Option to cat a specific handoff

### Out of Scope

- Real-time bidirectional chat with agents (agents are batch processes, not daemons)
- WebSocket or streaming interface
- Modifying the runner to keep agents alive between tasks
- Auth management (just report status, don't rotate keys)
- Backwards-incompatible changes to existing `arn collab` commands

## Agent Split

Per user request:
1. **Kimi**: Analyze codebase, design CLI commands + dashboard layout, write task brief and handoff
2. **Codex**: Review plan, add implementation notes and edge-case handling
3. **Claude**: Review plan, add architectural safety notes and UX refinements
4. **Claude (implementation)**: Implement the approved plan
5. **Codex + Kimi (testing)**: Run verification tests, validate CLI output, check integration

## Success Criteria

- `arn collab dashboard` renders a readable status board in the terminal
- `arn collab feed` stores human input where agents can access it
- `arn collab run` executes a full collaboration cycle and reports progress
- `arn collab agents` reports health for all three agent binaries
- `arn collab history` lists recent handoffs
- Existing collab tests still pass
- New tests for dashboard/feed/run commands pass
- No secrets exposed in CLI output or feed storage

## Suggested Verification

```bash
python3 -m py_compile arn_v9/scripts/arn_cli.py
python3 -m pytest arn_v9/tests/test_collab.py arn_v9/tests/test_collab_runner.py
arn collab status
arn collab agents
arn collab history
arn collab dashboard --once
```

## Files Expected to Change

- `arn_v9/scripts/arn_cli.py` — add new `collab` subcommands: `dashboard`, `feed`, `run`, `agents`, `history`
- `arn_v9/collab_runner.py` — expose helper functions for `run` command, inject feed context into prompts
- `arn_v9/collab.py` — add `feeds_dir()`, `list_handoffs()`, `agent_health()` helpers
- `arn_v9/tests/test_collab.py` — add CLI command tests
- `arn_v9/tests/test_collab_runner.py` — add feed-injection tests
- `COLLAB.md` — document new CLI commands
- `docs/collab-protocol.md` — document human-in-the-loop feed mechanism

## Dashboard Layout Sketch

```text
┌─────────────────────────────────────────────────────────────┐
│ ARN Collaboration Dashboard              Refresh: 14:32:05  │
├─────────────────────────────────────────────────────────────┤
│ Task: ARN-dashboard-relations-connections                   │
│ Status: HANDOFF_CODEX        Chain: codex → claude → kimi   │
│                                                             │
│ Agent Status                                                │
│ ───────────                                                 │
│ codex   ● ready     /Users/hustle/.nvm/.../bin/codex       │
│ claude  ● ready     /Users/hustle/Library/.../claude       │
│ kimi    ● auth OK   expires in 3521s                        │
│                                                             │
│ Lock                                                        │
│ ────                                                        │
│ Locked by: codex                                            │
│ Since: 2026-05-18T08:15:00Z                                 │
│ Stale: no (45m / 120m)                                      │
│                                                             │
│ Recent Handoffs                                             │
│ ───────────────                                             │
│ 08:15  codex   complete  "Added link storage schema"        │
│ 07:00  kimi    no_issues "Reviewed schema, no issues"       │
│ 06:00  claude  complete  "Added API endpoints and UI"       │
│                                                             │
│ Recent Report                                               │
│ ─────────────                                               │
│ /Users/hustle/.arn_data/collab/reports/...-cycle-report.md  │
└─────────────────────────────────────────────────────────────┘
```

## Feed Storage Design

Feeds should be lightweight operational notes, not full memories:

```json
{
  "feed_version": "1.0",
  "timestamp": "2026-05-18T14:32:05Z",
  "from": "human",
  "to": "claude",
  "message": "Focus on edge cases in the link schema"
}
```

Stored as a JSON line in `$ARN_DATA_DIR/collab/feeds/YYYY-MM-DD.jsonl`.

The runner injects the last N feeds into the agent prompt under a "Human Context" section.

## Follow-Up Task (if not in this run)

If real-time chat or persistent agent daemons are not implemented:

```text
Proposed task:
- Problem: User wants near-real-time interaction with agents, not batch cycles.
- Evidence: Batch cycles have 2-hour timeouts and cannot handle quick Q&A.
- Files: arn_v9/collab_runner.py, arn_v9/scripts/arn_cli.py
- Success criteria: `arn collab ask --agent kimi "question"` returns a response within 60s.
- Verification: Manual test + timeout test.
- Suggested owner: codex
```
