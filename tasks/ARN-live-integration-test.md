# ARN Task: Live Integration Test & Red Team Validation

## Task ID
`ARN-live-integration-test`

## Review Chain
```
claude → kimi → codex
```

## What Was Fixed in the Prior Cycle (do not repeat)
See `ARN-operational-audit-fix-result.md` for full details. Summary:
- `calibrated_confidence` fixed (0.078 → 0.503+ for relevant memories)
- DELETE `/v1/memory/{episode_id}` endpoint added
- Source badges (user=blue, agent=green, tool=orange) added to dashboard
- Trash button added to dashboard memory rows
- Auto-linking on store: new episodes auto-link to similar existing ones (similarity ≥ 0.6)
- Junk ⚠️ memories deleted from DB
- Servers restarted — port 8742 (OpenClaw) and 8745 (dashboard) are fresh with new code
- 33 tests passing

## Goal for This Cycle

**Validate ARN works end-to-end as a real memory system across sessions.**
Then fix anything that doesn't work.

The ARN promise: when an agent starts a new session, it gets relevant memories
from past sessions injected into its context. It should remember identities,
decisions, findings, and procedures — not just store them blindly.

## Claude's Specific Tasks

### 1. Run a real multi-session simulation (REQUIRED)

Use the ARN API (http://localhost:8742) to simulate what actually happens when
the red team agent runs across 3 separate sessions.

**Session 1 — Identity & Setup:**
Store these as agent_id="Website_Red_Team":
- "My name is Red Team Alpha. I test web applications for security vulnerabilities." (source=agent, importance=0.9)
- "Target system: internal dashboard at 127.0.0.1:8745. Testing for XSS, open redirects, auth bypass." (source=user, importance=0.8)
- "Tool call: curl_http — GET http://127.0.0.1:8745/dashboard" (source=tool:curl_http, memory_type=procedure, importance=0.6)
- "Finding: Dashboard loads without authentication. No login required to view memories." (source=agent, importance=0.85)

Then recall: query="who am I and what am I testing" — verify identity + target surface correctly

**Session 2 — Finding & Fix:**
Store:
- "Second session. Continuing red team of internal dashboard." (source=agent, importance=0.5)
- "Found: Relations tab SVG neuron graph loads nodes from /v1/memory/recall without authentication. Could expose agent memories to network-adjacent attackers." (source=agent, importance=0.9)
- "Recommended fix: add API key requirement or localhost-only bind for production deployments." (source=agent, importance=0.8)

Then recall: query="what vulnerabilities did I find" — verify finding from session 1 also surfaces

**Session 3 — Recall Quality Check:**
WITHOUT storing anything new, do:
- recall query="red team identity and name" → should get identity memory from session 1
- recall query="security vulnerabilities dashboard" → should get both findings
- recall query="tool calls made during testing" → should get the curl_http procedure memory
- recall query="authentication bypass findings" → should return the auth findings

For each recall, record: top result, its similarity, calibrated_confidence, and source.

If calibrated_confidence < 0.3 for any clearly relevant result (similarity > 0.4),
that's a recall quality failure — investigate and fix.

### 2. Test auto-linking created real links

After session 1 and 2 stores, query the memory_links table via the ARN API or
directly in SQLite to verify that auto-linking created relates_to connections
between related memories (e.g., the identity memory linked to the target memory,
the finding memories linked to each other).

Use: `GET /v1/agents/Website_Red_Team/links` or check the DB directly:
```bash
sqlite3 ~/.arn_data/agents/Website_Red_Team/memory.db \
  "SELECT from_episode_id, to_episode_id, relation_type, confidence FROM memory_links;"
```

If no links exist after multiple related stores, the auto-link is not firing in
the real code path (it may only work in test client, not real server). Fix it.

### 3. Verify dashboard source badges and trash button are live

Navigate to http://127.0.0.1:8745/dashboard after storing the red team memories.
The Memories tab should show:
- Blue "user" badge for user-sourced memories
- Green "agent" badge for agent-sourced memories  
- Orange "tool:curl_http" badge for procedure memories
- A trash icon (🗑️ or ×) on each row that calls DELETE /v1/memory/{id}

If the badges or trash button are not visible, find the bug. The server was
restarted so the new HTML should be live.

### 4. Check OpenClaw plugin endpoint compatibility

The plugin calls:
- POST /v1/memory/store  
- POST /v1/memory/recall
- The response must include `id` field in recall results (plugin uses it for dedup)

Verify the response schema from port 8742 matches what the plugin expects.
Check `openclaw-arn-plugin/index.js` lines for what fields it reads from results.

If `id` is missing from recall results (Codex noted this was a pre-existing issue
on the old server — check if the new server fixes it), add `id` to RecallResult
in server.py.

### 5. Fix any issues found. Run tests. Write handoff.

Handoff command:
```bash
python3 arn_v9/scripts/arn_cli.py collab handoff \
  --agent claude \
  --status complete \
  --task "ARN live integration test and red team validation" \
  --changes "..." \
  --verification "Session 1/2/3 recall results: [paste actual scores]. Auto-link: [found/not found]. Dashboard: [badges visible/not]. id field: [present/missing]." \
  --concerns "..." \
  --next-focus "Kimi: verify the changes, run pytest, confirm dashboard renders correctly, check the OpenClaw plugin can connect and store a real message end-to-end using the test script at arn_v9/scripts/test_openclaw_simulation.py"
```

## Success Criteria

- [ ] Session 1 recall returns correct identity + target (calibrated_confidence > 0.3)
- [ ] Session 2 recall returns both sessions' findings
- [ ] Session 3 cross-session recall works for all 4 query types
- [ ] Auto-links exist in memory_links table after stores
- [ ] Dashboard shows source badges and trash button (live, not just static check)
- [ ] Recall results include `id` field
- [ ] 33+ tests passing
- [ ] Handoff written with actual recall scores documented
