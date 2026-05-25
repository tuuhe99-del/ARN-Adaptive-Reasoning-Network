# ARN Task: Plugin Recall Quality — Review & Harden Production Fixes

## Task ID
`ARN-plugin-recall-quality`

## Priority
**HIGH** — Core recall pipeline is now functional but needs review, hardening,
and edge-case testing before this can be considered production-stable.

## Review Chain
```
codex → claude → kimi
```

## Context — What Was Fixed Today (2026-05-22)

Live no-MD test battery revealed 6 additional bugs in the plugin on top of the
4 already fixed by kimi in the previous collab cycle. All 6 were fixed and
verified with a full test battery (T1–T6, all PASS). The fixes are in:

`/Users/hustle/arn-v9-repo/openclaw-arn-plugin/index.js`

### FIX A — getAgentId() namespace fragmentation (CRITICAL)
**Problem:** `getAgentId()` used the full sanitized sessionKey as the agent_id
(e.g. `"agent:main:explicit:session-123"` → `"agent_main_explicit_session-123"`).
Each session created a new isolated ARN namespace, making cross-session recall
completely impossible.

**Fix:** Extracts the stable agent name from the session key pattern
`"agent:<name>:<mode>:<sessionId>"`:
```js
const parts = raw.split(":");
if (parts.length >= 2 && parts[0] === "agent" && parts[1]) {
  return parts[1].replace(/[^a-zA-Z0-9_\-]/g, "_");
}
```
All sessions for `--agent main` now consistently use `agent_id = "main"`.

### FIX B — arnRecall() filter used wrong score field (HIGH)
**Problem:** `arnRecall()` filtered results by `(r.calibrated_confidence || r.similarity) >= minScore`.
In JavaScript, if `calibrated_confidence = 0.038` (truthy), the OR short-circuits
and never reads `similarity`. Identity facts have low `calibrated_confidence`
(they're "expected", low prediction error) but high `r.score` (0.47–0.83).
Result: Jordan with `score=0.47` but `cal_conf=0.038` was filtered out even with
`minScore=0.05`.

**Fix:** Filter by `r.score` — the server's combined score (similarity × importance × recency):
```js
const effectiveScore = r.score ?? r.similarity ?? 0;
return effectiveScore >= minScore;
```

### FIX C — Sort used wrong score field (HIGH)
**Problem:** Same bug in the sort: `merged.sort((a,b) => (b.calibrated_confidence || b.similarity) - ...)`.
High-value facts with low `calibrated_confidence` sorted to the bottom, getting
cut off by the `tokenBudget` before reaching the agent.

**Fix:**
```js
merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
```

### FIX D — isRecallNoise() — noise filter too weak (MEDIUM)
**Problem:** The injected context was full of:
- Timestamped user question echoes: `[Fri 2026-05-22 19:18 EDT] Who am I?` scoring
  0.75+ when the current query was "Who am I?" — drowning out real facts
- OpenClaw runtime context blocks
- Raw tool call logs and tool results
- Old LLM self-identification outputs ("I chose the name Ash")

**Fix:** Added `isRecallNoise(r)` function that filters:
- OpenClaw runtime context blocks
- Tool call / tool result / after_tool_call source entries
- Ash self-naming patterns in llm_output
- Timestamped messages with body < 60 chars (short questions)
- Messages under 12 chars

Intentionally KEEPS timestamped messages with factual bodies (>60 chars) like
`[Fri ...] Remember: my API test server is at api-dev.internal:9090`.

### FIX E — formatArnMemories skips noise (MEDIUM)
**Problem:** `formatArnMemories()` formatted all results including noise.

**Fix:** Calls `isRecallNoise(r)` before adding each result to the formatted block:
```js
if (isRecallNoise(r)) continue;
```

### FIX F — sqlite-vec installed (LOW)
`pip install sqlite-vec` (v0.1.9) was run on the system. NOTE: The ARN Python
server does NOT currently reference `sqlite_vec` anywhere in its codebase — the
"chunks_vec not updated" warning is from openclaw's internal memory system
(bundled in the gateway), not from the ARN plugin. Installing sqlite-vec does
not affect ARN server behavior. This is a no-op for ARN but may help openclaw's
built-in memory subsystem.

## Test Battery Results (all MDs zeroed, ARN as sole memory)

| Test | Query | Expected | Result |
|------|-------|----------|--------|
| T1 | "Who am I? Name, project, colleagues" | Alex, ARN project, Jordan | ✅ PASS |
| T2 | "Who handles API pen testing?" | Jordan, security expert | ✅ PASS |
| T3 | "What's my API security test procedure?" | 3-step curl flow | ✅ PASS |
| T4 | "Which language do I prefer?" | Python | ✅ PASS |
| T5 | "What's my bank account / SSN?" | Refuse to guess | ✅ PASS |
| T6 | Store api-dev.internal:9090, recall next session | Exact address | ✅ PASS |

## What Verifiers Should Do

### codex (step 1 — code review)
1. Read the current plugin: `/Users/hustle/arn-v9-repo/openclaw-arn-plugin/index.js`
2. Verify all 5 code fixes (A–E) are correctly implemented
3. Check edge cases:
   - What if `parts[0] !== "agent"` in getAgentId? Does fallback work?
   - What if `r.score` is null/undefined in the filter and sort? Are defaults correct?
   - Does isRecallNoise accidentally filter legitimate factual messages?
   - Is the 60-char threshold for timestamp body filtering appropriate?
4. Run `node --check openclaw-arn-plugin/index.js`
5. Run `pytest arn_v9/tests/ -x -q`
6. Look for any regression in the existing fixes from the prior cycle

### claude (step 2 — integration review)
1. Review codex's handoff
2. Check that the `getAgentId()` fix doesn't break telegram or other channels:
   - Telegram session key: `"agent:main:telegram:direct:6196798335"` → should return "main"
   - Gateway direct: `"agent:main:explicit:session-id"` → should return "main"
   - Non-standard formats: fallback to full sanitized key (check the fallback path)
3. Verify the `isRecallNoise` filter doesn't over-filter:
   - Test content: `"[Fri 2026-05-22 19:40 EDT] Remember: my new API test server is at api-dev.internal:9090"` → should NOT be filtered (body > 60 chars)
   - Test content: `"[Fri 2026-05-22 19:18 EDT] Who am I?"` → SHOULD be filtered (body < 60 chars)
4. Consider: should tool_result episodes be filtered from recall entirely, or only from
   the formatted output? They may carry useful tool output context.

### kimi (step 3 — harden and extend)
1. Add minScore and topK to the redteam openclaw.json config to expose better tuning:
   - Suggest `minScore: 0.25` (lower than current 0.3 to catch more facts)
   - Suggest `topK: 8` (more candidates before noise filter)
2. Add a dedicated `session_start` identity recall for the `main` agent namespace:
   - On session start, explicitly recall "user identity name preferences" with minScore=0.05
   - Cache in `sessionPersonaCache` so it's available for all turns
3. Consider adding `memory_type: "procedure"` boost in recall — procedural facts should
   rank higher when the query contains words like "procedure", "steps", "how do I"
4. Write a standalone test script `tests/test_openclaw_integration.py` that:
   - Stores identity/procedure/preference facts via API
   - Recalls them with the same queries used in the battery
   - Asserts scores and content match expectations
   - Can be run without a live openclaw gateway

## Files To Review
- `/Users/hustle/arn-v9-repo/openclaw-arn-plugin/index.js` — all fixes applied here
- `/Users/hustle/.openclaw-redteam/openclaw.json` — plugin config (minScore, topK)
- `/Users/hustle/arn-v9-repo/arn_v9/tests/` — existing tests

## Known Remaining Issues (out of scope for this cycle)

1. **Base tier migration** — Server runs `nano` (384-dim) on an 8GB machine. Upgrading
   to `base` (768-dim) requires re-embedding all stored episodes. Separate task needed.

2. **WS concurrency** — Parallel CLI calls cause gateway handshake timeouts and zombie
   processes. See `tasks/ARN-gateway-ws-concurrency.md`.

3. **llm_input arn_memory_context re-storage** — Plugin stores `[llm_input]` messages
   which include the injected `<arn_memory_context>` block. The `shouldSkipContent`
   guard catches `<arn_memory_context>` but `[llm_input]` wraps it — check if
   these are leaking through.

4. **sqlite-vec not integrated into ARN server** — The Python server uses numpy memmap
   for vectors. sqlite-vec was installed but isn't used by ARN code. If we want
   true vector SQL queries, ARN persistence layer needs sqlite-vec integration.

## Verification After This Cycle
```bash
# Zero all MDs
for f in USER.md MEMORY.md IDENTITY.md AGENTS.md SOUL.md TOOLS.md HEARTBEAT.md; do
  > "$HOME/.openclaw-redteam/workspace/$f"
done
> "$HOME/.openclaw-redteam/workspace/memory/2026-05-22.md"

# Run battery
for i in 1 2 3 4 5; do
  openclaw --profile redteam agent --agent main \
    --session-id "verify-battery-t$i" \
    --message "$(echo 'Who am I?|Who handles API pen testing?|What is my API test procedure?|What language do I prefer?|What is my home address?' | cut -d'|' -f$i)" \
    --json 2>&1 | grep '"text":' | head -1
  echo "---"
done

# Restore MDs
cp /tmp/md-backup-full/USER.md ~/.openclaw-redteam/workspace/USER.md
cp /tmp/md-backup-full/MEMORY.md ~/.openclaw-redteam/workspace/MEMORY.md
```
