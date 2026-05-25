# ARN Task: Operational Audit & Full Fix

## Task ID
`ARN-operational-audit-fix`

## Review Chain
```
claude → kimi → codex
```

## Context: What Was Already Found (READ THIS FIRST)

The previous collab cycle (ARN-dashboard-v2-neuron-graph) had Codex verify
implementation quality. Its full findings (from stdout log) were:
- 33 tests pass, syntax clean, GRAPH_MODE injection works, all dashboard
  checklist items PASS
- `pip install -e .` only fails in no-network sandbox (not a real bug)
- Codex could not write its own handoff due to sandbox write permissions
  (THIS HAS BEEN FIXED — `danger-full-access` is now set for Codex)

## What Claude Has Confirmed Before Launching This Task

Critical issues discovered through live inspection that NO prior agent addressed:

### Issue 1 — Junk Memories Polluting Recall
Two `⚠️` error memories are in the database (episode IDs 1 and 2):
- `⚠️ Something went wrong while processing your request...`
- `⚠️ API provider returned a billing error...`
These were stored BEFORE the `shouldSkipContent` filter was added.
The filter prevents new ones but these two come back on every recall.
`source: "me"` on all 6 existing memories (old format — new format uses
"user" / "agent"). The recall formatter already handles "me" → "I said:"
but old memories should be cleaned up or source migrated.

### Issue 2 — `calibrated_confidence` Near Zero (0.078)
Recalls work but `calibrated_confidence` = 0.078 for a clearly relevant
memory. The `calibrate_similarity` function in the embedder computes this.
Confidence tiers show "low" for similarities of 0.33–0.53. This means
the plugin's `minScore: 0.35` filter may be too low OR the calibration
thresholds are miscalibrated for the embedding model in use. Investigate
`embedder.calibrate_similarity()` and `confidence_tier()` — understand if
the calibration is intentionally conservative or genuinely broken.

### Issue 3 — No Auto-Linking on Store
When a memory is stored, similar existing memories are NOT automatically
linked. The `memory_links` table (relations graph) requires manual wiring
via the dashboard. For ARN to be useful as a knowledge graph, links should
be inferred at store time: find top-3 similar existing memories (similarity
> 0.6) and auto-create `relates_to` links between them.

### Issue 4 — Four Server Instances Running Simultaneously
Ports 8742, 8743, 8744, 8745 all running uvicorn with the same server code.
Plugin uses 8742 (confirmed working). Dashboard was recently restarted on
8745. 8743 and 8744 appear to be stale processes from Monday. Do NOT kill
any processes — just document the situation and ensure the launchd config
(if any) points to one canonical port.

### Issue 5 — Dashboard: No Delete Button for Individual Memories
The dashboard can VIEW memories but cannot delete individual ones. The old
⚠️ memories can only be purged via API or DB directly. Add a delete button
(trash icon) to each memory row in the Memories tab that calls DELETE or a
store endpoint that removes by episode_id.

### Issue 6 — Dashboard: Source Badge Not Shown
Memory rows show content, type, importance, timestamp. They do NOT show
the `source` field (user / agent / tool:xyz / compaction). Add a small
source badge next to the type tag so it's immediately obvious who said
what.

## Primary Goals for This Run

1. **Test the full store → recall → inject loop with the red team agent.**
   Send real messages through the ARN API as agent_id="Website_Red_Team"
   (or whichever ID the red team agent uses in OpenClaw — check the
   `openclaw.plugin.json` and `SKILL.md` for how agentId is derived).
   Verify that after storing 3–4 episodic memories across different topics
   (identity, a tool call, a finding, a fix), a recall query surfaces the
   right ones with meaningful scores.

2. **Clean up junk memories.** Either:
   - Add `DELETE /v1/memory/{episode_id}` endpoint to the FastAPI server
     (simplest) and call it to remove episodes 1 and 2, OR
   - Add a `clean_junk()` method to the storage layer that removes memories
     matching `shouldSkipContent` criteria
   Then add a trash button to the dashboard memory list.

3. **Investigate and fix calibrated_confidence.**
   Read `arn_v9/core/cognitive.py` around `calibrate_similarity()` and
   `confidence_tier()`. Understand why similarities of 0.33–0.53 yield
   calibrated confidence of 0.078. If the calibration thresholds are
   wrong for the embedding model in use, fix them. Target: a clearly
   relevant memory (similarity > 0.4) should have calibrated_confidence
   > 0.3.

4. **Implement auto-linking on store.**
   In `plugin.py` or `cognitive.py`, after storing a new episodic memory,
   find the top-3 most similar EXISTING memories (similarity ≥ 0.6).
   For each, create a `memory_link` with relation_type="relates_to" if one
   doesn't already exist. This makes the neuron graph self-populate.

5. **Add source badge to dashboard memory rows.**
   In `server.py`, in the memory list HTML, add a small `<span>` badge
   showing `source` next to the existing type badge. Style it simply —
   different color per source type (user=blue, agent=green, tool=orange,
   compaction=gray).

6. **Run all tests after changes.** Must pass:
   ```bash
   python3 -m py_compile arn_v9/api/server.py arn_v9/core/cognitive.py arn_v9/plugin.py
   python3 -m pytest arn_v9/tests/ -x -q
   ```

## Agent Instructions

Read `COLLAB.md` and `docs/collab-protocol.md` first.

Do NOT expand scope. Fix exactly what is listed above, nothing else.

After each file change run py_compile on that file.

End by writing a handoff via:
```bash
python3 arn_v9/scripts/arn_cli.py collab handoff \
  --agent claude \
  --status complete \
  --task "ARN operational audit and fix" \
  --changes "..." \
  --verification "..." \
  --concerns "..." \
  --next-focus "Kimi: run live integration test with red team agent using real OpenClaw session, verify memories inject into next session prompt, test 3 sessions on different topics"
```

## Success Criteria

- [ ] ⚠️ junk memories deleted from DB
- [ ] DELETE /v1/memory/{episode_id} endpoint exists
- [ ] Trash button visible in dashboard memory list
- [ ] Source badge visible in dashboard memory rows
- [ ] `calibrated_confidence` > 0.25 for clearly relevant memories (similarity > 0.4)
- [ ] Auto-linking: storing a new memory creates `relates_to` links to similar existing ones
- [ ] Full store → recall loop tested with red team agent_id, results documented in handoff
- [ ] All tests pass (≥ 33)
- [ ] No secrets, no new server instances started
