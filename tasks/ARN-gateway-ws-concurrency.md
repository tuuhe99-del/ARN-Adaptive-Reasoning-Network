# ARN Task: Fix Gateway WebSocket Concurrency / Zombie CLI Processes

## Task ID
`ARN-gateway-ws-concurrency`

## Review Chain
```
claude → codex → kimi
```

## Context: What Was Found (READ THIS FIRST)

During live test battery execution, sending multiple `openclaw agent --json`
CLI calls in parallel against the redteam gateway (port 18790) caused the
gateway to drop connections with handshake timeouts. Each dropped connection
left an orphaned `openclaw-agent` node process running indefinitely, which
further saturated the gateway — a compounding failure cascade.

### Symptoms Observed
```
# Gateway log (multiple concurrent connections):
2026-05-22T13:49:57 handshake timeout conn=1bf2eca3 peer=127.0.0.1:50603->127.0.0.1:18790
2026-05-22T13:49:57 handshake timeout conn=820002bd peer=127.0.0.1:50605->127.0.0.1:18790
2026-05-22T13:49:57 handshake timeout conn=2c4b619f peer=127.0.0.1:50606->127.0.0.1:18790
2026-05-22T13:49:57 gateway connect failed: Error: gateway closed (1000):
2026-05-22T13:49:57 EMBEDDED FALLBACK: Gateway agent failed; running embedded agent
```
- 4 parallel CLI calls → 3 handshake timeouts + fallback to embedded agent
- After that, even sequential calls started timing out (gateway backlogged)
- `pgrep -la openclaw` showed 22+ node processes, most orphaned CLIs
- Killing the zombies (`kill <pids>`) restored normal gateway responsiveness

### Why It Matters
The `openclaw agent --json` CLI is the primary way to drive automated test
sessions for ARN. If the gateway can't handle even 2 simultaneous CLI turns,
automated test scripts must be artificially serialised with no error recovery,
and any stall leaves zombie processes that corrupt subsequent runs.

### What Is NOT Broken
- The gateway handles one request at a time correctly
- The gateway WS server itself is healthy (HTTP health endpoint always 200)
- Single sequential CLI calls succeed (though they take 2–8 min each under Codex)

## What To Fix

### Option A — Gateway-side connection queue (preferred)
Investigate whether the openclaw gateway has a configurable WS connection
queue / backlog size. If it rejects connections immediately when busy, add
a queue or increase the backlog so concurrent CLI requests wait rather than fail.

Config file: `~/.openclaw-redteam/openclaw.json` (look for gateway connection
or queue settings).

### Option B — CLI-side retry with backoff
If the gateway can't queue, update the test runner or add a wrapper script
that retries `openclaw agent --json` on `gateway closed (1000)` with
exponential backoff (1s, 2s, 4s), max 3 retries.

### Option C — Process cleanup on CLI exit
Ensure orphaned `openclaw-agent` node children are reaped when the parent CLI
exits (or times out). Check if the CLI already handles SIGTERM/SIGINT cleanup
and whether the agent subprocess is spawned with `detached: true` unexpectedly.

## Files Relevant
- `~/.openclaw-redteam/openclaw.json` — gateway config
- `/tmp/openclaw/openclaw-2026-05-22.log` — gateway logs showing the timeouts
- Any test runner scripts that invoke `openclaw agent --json` in parallel

## Agent Instructions

- Read `COLLAB.md` and `docs/collab-protocol.md` first.
- Claim your step, do minimal correct work, write a handoff.
- Start with Option A (config investigation) before touching any code.
- If no gateway config knob exists, implement Option B as a wrapper script.
- Do NOT modify openclaw's installed node_modules.

## Verification
```bash
# Fire 3 concurrent agent turns and check that all 3 complete without 401/timeout
for i in 1 2 3; do
  openclaw --profile redteam agent --agent main \
    --session-id "concurrency-test-$i" \
    --message "Say PONG$i." --json > /tmp/conc$i.json 2>&1 &
done
wait
grep -l 'runId' /tmp/conc1.json /tmp/conc2.json /tmp/conc3.json | wc -l
# Expected: 3
```
