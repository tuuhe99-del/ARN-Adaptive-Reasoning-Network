# ARN Collaboration Log

## Vision

ARN should support serial collaboration between local coding agents without
parallel edits or hidden state. One agent claims a task, writes or reviews code,
records a handoff, and the next agent continues from that handoff.

The larger ARN goal is a local-first portable memory layer for AI agents:
episodic memory for what happened, semantic memory for what it means, working
memory for what is active now, contradiction handling when facts change, and
portable import/export so memory can move across agents, models, and devices.
ARN should stay small, inspectable, local-first, and reliable enough to trust.

Agents are expected to improve ARN toward that goal. They should not wait for
every task to be prewritten. When research files, tests, code inspection, or a
handoff reveal a concrete gap, the agent should propose or create the next
specific task with file paths, success criteria, and verification steps.

## Rules

- One active agent edits at a time.
- Every run starts with `arn collab status`.
- An agent may only work after `arn collab claim --agent <agent>`.
- Every completed run ends with `arn collab handoff`.
- If no issues are found during review, write a `no_issues` handoff and do not
  edit code.
- Runtime state lives under `$ARN_DATA_DIR/collab`.
- Do not store secrets, API keys, or credentials in handoffs.

## Default Review Chain

```text
codex -> claude -> kimi
```

The chain can be changed per cycle with:

```bash
arn collab init --task-id ARN-001 --review-chain kimi,codex,claude --force
```

## Current State

Use the CLI as the source of truth:

```bash
arn collab status
```

## Operator Dashboard

Check all agents, recent handoffs, and lock status at a glance:

```bash
arn collab dashboard --once
arn collab dashboard --refresh 30   # auto-refresh every 30s
arn collab agents                   # show binary/auth health for codex, claude, kimi
arn collab history                  # list recent handoffs
arn collab history --file /path/to/handoff.md  # cat a specific handoff
```

Feed a message to one or all agents before the next cycle:

```bash
arn collab feed -m "Focus on edge cases in the link schema" --agent claude
arn collab feed -m "Prioritize data-integrity tests" --agent all
```

Feeds are stored in `$ARN_DATA_DIR/collab/feeds/YYYY-MM-DD.jsonl` and injected into
the agent's prompt under a "Human Context" section on the next cycle.

Trigger a full collaboration cycle manually:

```bash
arn collab run --task-id ARN-001
arn collab run --task-id ARN-001 --dry-run   # show prompts without launching agents
arn collab run --task-id ARN-001 --force     # restart from scratch
```

## Runner

Dry-run the next agent command:

```bash
python3 -m arn_v9.collab_runner --repo-dir /Users/hustle/arn-v9-repo --task-id ARN-001 --force
```

Execute the real serial loop:

```bash
python3 -m arn_v9.collab_runner --repo-dir /Users/hustle/arn-v9-repo --task-id ARN-001 --force --execute
```

## Initial Improvement Backlog

These are seed tasks, not the full roadmap. Agents should split, refine, and
reprioritize them based on code inspection and research evidence.

- Atomic vector expansion with temp-file and rename
- Corrupted `.npy` detection and recovery
- Schema migration error handling
- `model_version` tracking for embedding drift
- Re-embedding migration tool
- Disk-space pre-flight checks
- Request size limits
- Nightly CI stress tests
- OpenClaw plugin Jest test suite
- OpenClaw plugin retry logic
- Timing-safe API key comparison
- Global rate limiter
- Pre-allocate larger vector stores
- Narrow lock scope in `store_episode`
- Future local dashboard for browsing memories and manually wiring related
  identity, tool-call, response, and time memories together

## Task Creation Rule

When an agent finds work needed to move ARN toward the vision, it should record
the proposed task in its handoff using this format:

```text
Proposed task:
- Problem:
- Evidence:
- Files:
- Success criteria:
- Verification:
- Suggested owner:
```

## Handoff Standard

Every handoff must include:

- task summary
- changed files or review result
- verification performed
- concerns
- next-agent focus

See `docs/collab-protocol.md` for command examples.
