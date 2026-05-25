# ARN Collaboration Protocol

This protocol coordinates Codex, Claude, and Kimi as serial coding reviewers.
It is intentionally file-based and local-first.

## Runtime Layout

```text
$ARN_DATA_DIR/collab/
  state.json
  handoffs/
  reports/
  logs/
```

## State Flow

```text
IDLE
CLAIMED_CODEX
HANDOFF_CODEX
CLAIMED_CLAUDE
HANDOFF_CLAUDE
CLAIMED_KIMI
DONE
```

The review chain can rotate, so the exact agent names may differ.

## Commands

Initialize a cycle:

```bash
arn collab init --task-id ARN-001 --force
```

Check state:

```bash
arn collab status
arn collab next
```

Claim the next step:

```bash
arn collab claim --agent codex
```

Write a handoff:

```bash
arn collab handoff \
  --agent codex \
  --status complete \
  --task "Add collaboration state machine" \
  --changes "Added file-based lock/state handling and CLI commands" \
  --verification "python3 -m pytest arn_v9/tests/test_collab.py" \
  --concerns "Full ARN test suite not run" \
  --next-focus "Review stale lock behavior and handoff validation"
```

Validate a handoff:

```bash
arn collab validate-handoff "$ARN_DATA_DIR/collab/handoffs/<file>.md"
```

## Operator Commands

### Dashboard

```bash
arn collab dashboard --once        # single-shot status board
arn collab dashboard --refresh 30  # auto-refresh every 30s
```

### Agent health

```bash
arn collab agents
```

Reports binary presence and (for Kimi) OAuth expiry.

### Handoff history

```bash
arn collab history                        # last 10 handoffs as JSON
arn collab history --limit 5              # limit to 5
arn collab history --file <path>.md       # cat a specific handoff
```

### Human feed

```bash
arn collab feed --message "..." --agent claude
arn collab feed --message "..." --agent all
```

Messages are stored as JSON lines in `$ARN_DATA_DIR/collab/feeds/YYYY-MM-DD.jsonl`:

```json
{"feed_version": "1.0", "timestamp": "...", "from": "human", "to": "claude", "message": "..."}
```

The runner reads the last 5 feed entries relevant to the current agent and injects
them into the prompt under a **Human Context** section. Feeds targeting `all` are
visible to every agent.

### Manual cycle run

```bash
arn collab run --task-id ARN-001
arn collab run --task-id ARN-001 --dry-run
arn collab run --task-id ARN-001 --force --review-chain kimi,claude,codex
```

Exit code 0 means DONE; exit code 2 means blocked or incomplete.

## Runner

Dry-run the next cycle without launching agents:

```bash
python3 -m arn_v9.collab_runner \
  --repo-dir /Users/hustle/arn-v9-repo \
  --data-dir /Users/hustle/.arn_data \
  --task-id ARN-001 \
  --force
```

Launch the real serial loop:

```bash
python3 -m arn_v9.collab_runner \
  --repo-dir /Users/hustle/arn-v9-repo \
  --data-dir /Users/hustle/.arn_data \
  --task-id ARN-001 \
  --review-chain codex,claude,kimi \
  --force \
  --execute
```

The runner stops when:

- the final agent writes a handoff and state becomes `DONE`
- an agent command exits non-zero
- an agent exits without writing a handoff
- dry-run mode has shown the next command

## Morning/Night Schedule

Launchd templates live in:

```text
launchd/com.arn.collab.morning.plist
launchd/com.arn.collab.night.plist
```

They are templates, not installed automatically. Morning uses:

```text
codex -> claude -> kimi
```

Night uses:

```text
claude -> kimi -> codex
```

Install them manually with:

```bash
mkdir -p ~/Library/LaunchAgents
mkdir -p ~/.arn_data/collab/logs
cp launchd/com.arn.collab.morning.plist ~/Library/LaunchAgents/
cp launchd/com.arn.collab.night.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.arn.collab.morning.plist
launchctl load ~/Library/LaunchAgents/com.arn.collab.night.plist
```

## Agent Instructions

Before editing:

1. Run `arn collab status`.
2. Confirm you are the next agent.
3. Claim the task.
4. Read the previous handoff if one exists.

When choosing work:

1. Use the ARN vision in `COLLAB.md` as the north star.
2. Prefer reliability and data integrity before new intelligence features.
3. Turn vague goals into specific tasks with file paths and success criteria.
4. Use `research/*.md`, failing tests, and code inspection as evidence.
5. Record newly discovered tasks in the handoff instead of silently expanding
   scope.

When reviewing:

1. Inspect the prior handoff.
2. Inspect changed files and related tests.
3. Fix concrete issues if found.
4. If no issues are found, do not edit code.
5. Write a handoff either way.

## ARN Roadmap Priorities

Agents should improve ARN in this order unless the user gives a newer priority:

1. Data integrity: atomic vector expansion, corrupted `.npy` recovery, schema
   migration safety, disk-space checks, larger vector pre-allocation, narrower
   storage locks.
2. Embedding drift: `model_version` tracking and a re-embedding migration tool.
3. API/resource safety: request limits, timing-safe API key comparison, global
   rate limiting.
4. Integration reliability: OpenClaw plugin retry logic, plugin tests, nightly
   stress tests.
5. Memory usability: local dashboard for inspecting memories and manually
   connecting related memories such as identity, tool calls, responses, and time.

## Proposed Task Format

Use this block in handoffs when discovering necessary follow-up work:

```text
Proposed task:
- Problem:
- Evidence:
- Files:
- Success criteria:
- Verification:
- Suggested owner:
```

## Handoff Status Values

- `complete`: implementation or review completed
- `blocked`: work cannot continue without human input
- `needs_review`: work completed but risk remains
- `no_issues`: review found no actionable issue

## Safety Rules

- Do not run multiple editing agents at once.
- Do not overwrite another agent's handoff.
- Do not use ARN's SQLite/memmap storage for collaboration locks.
- Use a fresh test data directory for tests that touch ARN memory.
- Treat locks older than `stale_after_minutes` as stale only when explicitly
  using `--steal-stale`.
