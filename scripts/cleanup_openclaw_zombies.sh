#!/usr/bin/env bash
# cleanup_openclaw_zombies.sh — Kill orphaned openclaw-agent node processes
#
# Usage:
#   ./cleanup_openclaw_zombies.sh           # dry-run: list zombies
#   ./cleanup_openclaw_zombies.sh --kill    # kill them
#   ./cleanup_openclaw_zombies.sh --kill --gateway  # also restart the gateway
#
# A process is considered a zombie if:
#   - It matches the openclaw agent pattern
#   - It has been running > ZOMBIE_AGE_SECONDS (default: 600s)

set -euo pipefail

KILL_MODE=0
RESTART_GATEWAY=0
ZOMBIE_AGE="${ZOMBIE_AGE_SECONDS:-600}"
PROFILE="${OPENCLAW_PROFILE:-redteam}"

for arg in "$@"; do
    case "$arg" in
        --kill) KILL_MODE=1 ;;
        --gateway) RESTART_GATEWAY=1 ;;
    esac
done

echo "=== OpenClaw Zombie Cleanup ==="
echo "Mode: $([ $KILL_MODE -eq 1 ] && echo 'KILL' || echo 'DRY RUN')"
echo "Age threshold: ${ZOMBIE_AGE}s"
echo ""

NOW=$(date +%s)
ZOMBIE_COUNT=0
ZOMBIE_PIDS=""

while IFS= read -r line; do
    PID=$(echo "$line" | awk '{print $1}')
    START=$(ps -p "$PID" -o lstart= 2>/dev/null || echo "")
    if [[ -z "$START" ]]; then continue; fi

    # Convert start time to epoch
    START_EPOCH=$(date -j -f "%a %b %d %T %Y" "$START" +%s 2>/dev/null || echo "0")
    AGE=$((NOW - START_EPOCH))

    CMD=$(ps -p "$PID" -o command= 2>/dev/null | head -c 80 || echo "unknown")

    # Gateway processes are long-lived by design — never treat them as zombies.
    # The pgrep name may be "node" so we must check the full CMD, not the pgrep line.
    if echo "$CMD" | grep -qi "gateway"; then
        echo "  GATEWAY PID=$PID age=${AGE}s cmd=${CMD} (skipped — gateway is long-lived)"
        continue
    fi

    if [[ $AGE -gt $ZOMBIE_AGE ]]; then
        echo "  ZOMBIE PID=$PID age=${AGE}s cmd=${CMD}"
        ZOMBIE_PIDS="$ZOMBIE_PIDS $PID"
        ZOMBIE_COUNT=$((ZOMBIE_COUNT + 1))
    else
        echo "  ACTIVE PID=$PID age=${AGE}s cmd=${CMD}"
    fi
done < <(pgrep -la "openclaw" 2>/dev/null | grep -i "agent\|node" || true)

echo ""
echo "Found $ZOMBIE_COUNT zombie(s) older than ${ZOMBIE_AGE}s"

if [[ $KILL_MODE -eq 1 && $ZOMBIE_COUNT -gt 0 ]]; then
    echo "Killing zombies..."
    for PID in $ZOMBIE_PIDS; do
        kill -TERM "$PID" 2>/dev/null && echo "  Sent SIGTERM to $PID" || echo "  $PID already gone"
    done
    sleep 2
    # Force kill any survivors
    for PID in $ZOMBIE_PIDS; do
        if kill -0 "$PID" 2>/dev/null; then
            kill -KILL "$PID" 2>/dev/null && echo "  Force-killed $PID"
        fi
    done
    echo "Done."
fi

if [[ $RESTART_GATEWAY -eq 1 ]]; then
    echo ""
    echo "Restarting gateway (profile: $PROFILE)..."
    openclaw --profile "$PROFILE" gateway restart 2>&1 | head -5
    echo "Gateway restart initiated."
fi
