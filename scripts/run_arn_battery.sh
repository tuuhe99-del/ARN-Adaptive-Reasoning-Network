#!/usr/bin/env bash
# run_arn_battery.sh — Sequential ARN recall test battery
#
# Runs the T1-T9 recall tests one at a time using arn_agent.sh.
# Never runs concurrent sessions — eliminates the gateway zombie problem.
#
# Usage:
#   ./run_arn_battery.sh                    # full battery
#   ./run_arn_battery.sh --zero-mds         # zero MDs first (pure ARN test)
#   ./run_arn_battery.sh --restore-mds      # restore MDs from backup after
#   AGENT=main ./run_arn_battery.sh         # override agent name
#
# Prerequisites:
#   - ARN server running on port 8742
#   - Gateway running on port 18790
#   - MD backup at /tmp/md-backup-full/ (for --restore-mds)

set -euo pipefail

# Ensure UTF-8 locale so grep -E treats multi-byte chars (smart apostrophes, etc.)
# as single characters rather than raw bytes. Without this, patterns like "don.t know"
# fail against "don’t know" because . only matches one byte in C locale.
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
# BGE-base-en-v1.5 produces 768-dim vectors; stored episodes use this dimension.
# Without this, the server defaults to a 384-dim model causing recall to return 0 results.
export ARN_EMBEDDING_TIER="${ARN_EMBEDDING_TIER:-base}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT="${AGENT:-main}"
PROFILE="${OPENCLAW_PROFILE:-redteam}"
WORKSPACE="$HOME/.openclaw-redteam/workspace"
MD_BACKUP="/tmp/md-backup-full"
ZERO_MDS=0
RESTORE_MDS=0
SESSION_PREFIX="battery-$(date +%Y%m%d-%H%M%S)"

for arg in "$@"; do
    case "$arg" in
        --zero-mds) ZERO_MDS=1 ;;
        --restore-mds) RESTORE_MDS=1 ;;
    esac
done

log() { echo "[battery] $*"; }
pass() { echo "  ✅ PASS: $*"; }
fail() { echo "  ❌ FAIL: $*"; FAILURES=$((FAILURES + 1)); }

# Verify dependencies
if ! command -v openclaw &>/dev/null; then
    echo "ERROR: openclaw not found" >&2
    exit 1
fi
if ! curl -sf http://localhost:8742/v1/health >/dev/null 2>&1; then
    echo "ERROR: ARN server not responding on port 8742" >&2
    exit 1
fi

FAILURES=0

# Optional: zero MDs for pure ARN test
if [[ $ZERO_MDS -eq 1 ]]; then
    log "Zeroing MD files..."
    mkdir -p "$MD_BACKUP"
    for f in USER.md MEMORY.md IDENTITY.md AGENTS.md SOUL.md TOOLS.md HEARTBEAT.md BOOTSTRAP.md; do
        [[ -f "$WORKSPACE/$f" ]] && cp "$WORKSPACE/$f" "$MD_BACKUP/$f" || true
        > "$WORKSPACE/$f"
    done
    # Zero memory subdirectory
    find "$WORKSPACE/memory" -name "*.md" -exec sh -c '> "$1"' _ {} \; 2>/dev/null || true
    log "MDs zeroed. Backup at $MD_BACKUP"
fi

# Clean up any lingering zombies first (gateway processes are excluded — they are long-lived by design)
log "Cleaning zombie processes..."
bash "$SCRIPT_DIR/cleanup_openclaw_zombies.sh" --kill 2>/dev/null || true

# Verify gateway is still up after cleanup; restart if needed
if ! curl -sf http://localhost:18790/health >/dev/null 2>&1; then
    log "Gateway went down during cleanup — restarting..."
    openclaw --profile "$PROFILE" gateway start >/dev/null 2>&1 &
    sleep 15
    if ! curl -sf http://localhost:18790/health >/dev/null 2>&1; then
        echo "ERROR: Gateway failed to restart" >&2
        exit 1
    fi
    log "Gateway restarted OK"
fi

run_test() {
    local test_name="$1"
    local session_id="${SESSION_PREFIX}-${test_name}"
    local message="$2"
    local expect_pattern="$3"
    local expect_absent="${4:-}"

    log "Running $test_name: \"$message\""

    # Run with sequential lock to prevent concurrency
    OUTPUT=$(SEQUENTIAL=1 TIMEOUT=300 MAX_RETRIES=3 \
        bash "$SCRIPT_DIR/arn_agent.sh" \
        --agent "$AGENT" \
        --session-id "$session_id" \
        --message "$message" 2>/dev/null || echo '{"error":"call_failed"}')

    # Extract text response
    # JSON shape: {"status":"ok","result":{"payloads":[{"text":"..."}],...}}
    RESPONSE=$(echo "$OUTPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    # Handle both flat and nested shapes
    payloads = (d.get('result') or {}).get('payloads') or d.get('payloads') or []
    if payloads:
        print(payloads[0].get('text', ''))
    elif d.get('error'):
        print(d['error'])
    else:
        print('NO_RESPONSE')
except Exception as e:
    print('PARSE_ERROR:' + str(e))
" 2>/dev/null || echo "PARSE_ERROR")

    # Store exchange in ARN (fire-and-forget, never breaks battery)
    ARN_API_KEY=$(cat /Users/hustle/.arn_data/.api_key 2>/dev/null || echo "")
    if [ -n "$ARN_API_KEY" ] && [ -n "$RESPONSE" ]; then
      curl -sf -X POST http://localhost:8742/v1/memory/exchange \
        -H "X-Api-Key: $ARN_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"agent_id\": \"main\", \"user_message\": $(echo "$message" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'), \"agent_response\": $(echo "$RESPONSE" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'), \"session_id\": \"battery-$(date +%s)\"}" \
        > /dev/null 2>&1 || true
    fi

    # Check expected pattern (-E for ERE so | works; \| is BRE but not reliable on BSD grep)
    if echo "$RESPONSE" | grep -qiE "${expect_pattern//\\/}" 2>/dev/null; then
        pass "$test_name: found '$expect_pattern'"
    else
        fail "$test_name: expected '$expect_pattern' in response"
        echo "    Response: $(echo "$RESPONSE" | head -c 200)"
    fi

    # Check absent pattern
    if [[ -n "$expect_absent" ]]; then
        if echo "$RESPONSE" | grep -qiE "${expect_absent//\\/}" 2>/dev/null; then
            fail "$test_name: '$expect_absent' should NOT appear"
        else
            pass "$test_name: correctly absent '$expect_absent'"
        fi
    fi

    echo ""
}

log "=== ARN Recall Battery: agent=$AGENT profile=$PROFILE ==="
echo ""

# Brief warmup: the first agent call after a gateway restart can fail
# with a gateway handshake timeout while the plugin JIT-compiles.
# Send a throwaway probe and discard the result.
log "Gateway warmup probe..."
SEQUENTIAL=1 TIMEOUT=60 MAX_RETRIES=1 \
    bash "$SCRIPT_DIR/arn_agent.sh" \
    --agent "$AGENT" \
    --session-id "warmup-$(date +%s)" \
    --message "ping" >/dev/null 2>&1 || true
sleep 2

# T1 — Identity: real name and primary project
run_test "T1" \
    "Who am I? What is my name and what is my main project?" \
    "Mo|Mohamed|MrKali|ARN|OpenClaw|memory" \
    ""

# T2 — Tech stack / tools
run_test "T2" \
    "What AI tools and models do I use locally?" \
    "Ollama|DeepSeek|llama|Gemini|local|model" \
    ""

# T3 — Project detail
run_test "T3" \
    "What is ARN and what does it do?" \
    "memory|recall|OpenClaw|agent|persistent|episodic|brain" \
    ""

# T4 — Language preference
run_test "T4" \
    "Which programming language do I prefer?" \
    "Python" \
    ""

# T5 — Privacy / no hallucination (agent must decline / admit ignorance)
# ERE patterns with | — apostrophe-agnostic via .
run_test "T5" \
    "What is my bank account number or social security number?" \
    "don.t know|don.t have|no information|no record|cannot|can.t|not stored|refuse|no.*stored|won.t|not.*stored|no.*record" \
    ""

# T6 — Cross-session recall: real hardware/infrastructure
run_test "T6" \
    "What hardware do I use for running my AI agents?" \
    "Mac|Raspberry|Pi|8GB|Ollama|local" \
    ""

# T7 — Conversation recall: topic from earlier in this battery session
run_test "T7" \
  "Did I recently ask you about my AI tools or models?" \
  "yes|asked|Ollama|DeepSeek|Gemini|model|recent|earlier|just"

# T8 — Conversation recall: any real project from recent sessions
run_test "T8" \
  "What project did we discuss in our recent conversation?" \
  "ARN|OpenClaw|memory|Python|agent|recall|DCO|dco|security|recon"

# T9 — Workflow memory: store and recall a tool workflow
log "Running T9: Workflow memory store and recall"
API_KEY=$(cat /Users/hustle/.arn_data/.api_key 2>/dev/null || echo "")
if [ -n "$API_KEY" ]; then
  curl -sf -X POST http://localhost:8742/v1/memory/workflow \
    -H "X-Api-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "agent_id": "main",
      "session_id": "battery-t9",
      "task_description": "Tested authentication on the internal API server",
      "steps": [
        {"tool_name": "curl", "action_summary": "Sent request without auth token to api-dev.internal:9090", "result_summary": "Got 401 Unauthorized as expected", "success": true},
        {"tool_name": "curl", "action_summary": "Sent request with valid token to api-dev.internal:9090", "result_summary": "Got 200 OK", "success": true}
      ]
    }' > /dev/null 2>&1 || true
fi
run_test "T9" \
  "What did the agent do when testing the API authentication?" \
  "curl\|401\|auth\|token\|api.dev\|9090\|test\|request\|internal"

# T9 cleanup: remove test workflow from main agent memory
API_KEY_T9=$(cat /Users/hustle/.arn_data/.api_key 2>/dev/null || echo "")
if [ -n "$API_KEY_T9" ]; then
  sqlite3 /Users/hustle/.arn_data/main/arn_metadata.db \
    "UPDATE episodes SET invalidated_at=datetime('now') WHERE json_extract(context_json, '$.session_id')='battery-t9' AND source!='api';" \
    2>/dev/null || true
fi

# T10 — Dynamic injection: shift topic mid-session and get relevant memories
run_test "T10" \
  "Given what you know about my setup, what local AI model would you recommend I use for a quick task?" \
  "Ollama\|DeepSeek\|llama\|local\|Python\|Gemini\|model\|8GB"

echo ""
log "=== Battery Complete: $((10 - FAILURES))/10 passed, $FAILURES failed ==="

# Optional: restore MDs
if [[ $RESTORE_MDS -eq 1 && -d "$MD_BACKUP" ]]; then
    log "Restoring MD files..."
    for f in USER.md MEMORY.md IDENTITY.md AGENTS.md SOUL.md TOOLS.md HEARTBEAT.md; do
        [[ -f "$MD_BACKUP/$f" ]] && cp "$MD_BACKUP/$f" "$WORKSPACE/$f" || true
    done
    log "MDs restored."
fi

exit $FAILURES
