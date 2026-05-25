#!/usr/bin/env bash
# arn_agent.sh — Reliable openclaw agent CLI wrapper
#
# Wraps `openclaw --profile redteam agent --json` with:
#   1. Exponential backoff retry on gateway handshake failures (up to 3 attempts)
#   2. Child-process cleanup on exit so zombies don't accumulate
#   3. Optional lock file for enforcing sequential execution
#   4. Per-call timeout (default 300s)
#
# Usage:
#   ./arn_agent.sh --agent main --session-id my-session --message "Hello"
#   SEQUENTIAL=1 ./arn_agent.sh --agent main ...   # uses lock file
#   TIMEOUT=600 ./arn_agent.sh ...                  # override timeout
#
# Env vars:
#   MAX_RETRIES     max retry attempts (default: 3)
#   RETRY_BASE_MS   base backoff in ms (default: 2000)
#   TIMEOUT         per-attempt timeout in seconds (default: 300)
#   SEQUENTIAL      if set to 1, acquire a lock file before running
#   LOCK_FILE       lock file path (default: /tmp/arn_agent.lock)
#   OPENCLAW_PROFILE  openclaw profile (default: redteam)

set -euo pipefail

MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BASE_MS="${RETRY_BASE_MS:-2000}"
TIMEOUT="${TIMEOUT:-300}"
SEQUENTIAL="${SEQUENTIAL:-0}"
LOCK_FILE="${LOCK_FILE:-/tmp/arn_agent.lock}"
PROFILE="${OPENCLAW_PROFILE:-redteam}"

# Track child PIDs for cleanup
CHILD_PID=""

cleanup() {
    local exit_code=$?
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        sleep 0.5
        kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
    # Release lock if we hold it
    if [[ "$SEQUENTIAL" == "1" ]]; then
        rm -f "$LOCK_FILE"
    fi
    exit $exit_code
}
trap cleanup EXIT INT TERM

# Acquire lock for sequential mode
if [[ "$SEQUENTIAL" == "1" ]]; then
    LOCK_TIMEOUT=120
    WAITED=0
    while ! (set -o noclobber; echo "$$:$(date +%s)" > "$LOCK_FILE") 2>/dev/null; do
        # Check if lock holder is still alive
        LOCK_PID=$(cut -d: -f1 "$LOCK_FILE" 2>/dev/null || echo "")
        if [[ -n "$LOCK_PID" ]] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
            rm -f "$LOCK_FILE"
            continue
        fi
        if [[ $WAITED -ge $LOCK_TIMEOUT ]]; then
            echo "[arn_agent] ERROR: lock timeout after ${LOCK_TIMEOUT}s" >&2
            exit 1
        fi
        sleep 2
        WAITED=$((WAITED + 2))
    done
fi

# Kill any lingering openclaw processes from a previous failed run
ZOMBIES=$(pgrep -f "openclaw.*agent" 2>/dev/null | grep -v "^$$\$" || true)
if [[ -n "$ZOMBIES" ]]; then
    echo "[arn_agent] Cleaning up ${#ZOMBIES} orphaned openclaw process(es)" >&2
    echo "$ZOMBIES" | xargs kill -TERM 2>/dev/null || true
    sleep 1
fi

ATTEMPT=0
while [[ $ATTEMPT -lt $MAX_RETRIES ]]; do
    ATTEMPT=$((ATTEMPT + 1))
    if [[ $ATTEMPT -gt 1 ]]; then
        BACKOFF_MS=$(( RETRY_BASE_MS * (2 ** (ATTEMPT - 2)) ))
        BACKOFF_S=$(( BACKOFF_MS / 1000 ))
        echo "[arn_agent] Retry $ATTEMPT/$MAX_RETRIES — waiting ${BACKOFF_S}s before retry..." >&2
        sleep "$BACKOFF_S"
    fi

    # Run openclaw with timeout, capture output.
    # Note: GNU `timeout` is not available on macOS. Use a background sleep
    # watchdog instead.
    # Capture stdout (JSON payload) and stderr (logs/warnings) separately so
    # the caller receives clean JSON even when openclaw emits plugin warnings
    # on stderr.
    TMPOUT=$(mktemp /tmp/arn_agent_out.XXXXXX)
    TMPERR=$(mktemp /tmp/arn_agent_err.XXXXXX)
    set +e
    openclaw --profile "$PROFILE" agent --json "$@" > "$TMPOUT" 2> "$TMPERR" &
    CHILD_PID=$!

    # Watchdog: kill the child after $TIMEOUT seconds
    ( sleep "$TIMEOUT" && kill -TERM "$CHILD_PID" 2>/dev/null ) &
    WATCHDOG_PID=$!

    wait "$CHILD_PID"
    EXIT_CODE=$?
    kill "$WATCHDOG_PID" 2>/dev/null || true   # cancel watchdog if child finished first
    wait "$WATCHDOG_PID" 2>/dev/null || true
    CHILD_PID=""
    set -e

    OUTPUT=$(cat "$TMPOUT")
    ERROUT=$(cat "$TMPERR")
    rm -f "$TMPOUT" "$TMPERR"

    # Check for gateway failure patterns in stderr (where gateway errors land)
    COMBINED="$OUTPUT $ERROUT"
    if echo "$COMBINED" | grep -qE "gateway.*closed|handshake timeout|gateway connect failed|gateway request timeout" 2>/dev/null; then
        if [[ $ATTEMPT -lt $MAX_RETRIES ]]; then
            echo "[arn_agent] Gateway failure on attempt $ATTEMPT — will retry" >&2
            echo "$COMBINED" | grep -E "gateway|EMBEDDED|timeout" | head -3 >&2
            continue
        fi
    fi

    # Success or non-retryable failure
    echo "$OUTPUT"
    exit $EXIT_CODE
done

echo "[arn_agent] All $MAX_RETRIES attempts failed" >&2
exit 1
