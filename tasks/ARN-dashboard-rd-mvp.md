# ARN Dashboard R&D MVP

## Goal

Build the first local ARN dashboard so the user can see memory features as they
come in. This is an R&D dashboard, not a polished product shell.

## User Need

The user needs a browser-visible control surface for ARN:

- see stored memories
- filter/search memories by agent/type/source
- inspect memory details
- see basic stats
- prepare for future manual wiring between related identity, tool-call,
  response, and time memories

The long-term dashboard direction is "neurons that fire together wire together":
the user should eventually be able to explicitly connect memories that belong
together. For this run, do not overbuild the graph system. If manual links are
small and safe, implement them; otherwise create a concrete follow-up task with
schema/API/UI success criteria.

## Current OpenClaw Redteam Context

Checked before this task:

```text
openclaw --profile redteam gateway health
Gateway Health: OK
Telegram: configured

openclaw --profile redteam channels status
Gateway reachable.
Telegram default (website-redteam): enabled, configured, running, connected, mode: polling
Warning: plugins.entries.arn-memory-bridge is configured but disabled
```

Agents should use the redteam OpenClaw profile only for validation/configuration
checks, not as a replacement for local tests.

## Initial Scope

Implement a minimal local dashboard in the existing FastAPI API server.

Expected direction:

- Add a dashboard route such as `GET /dashboard`.
- Serve a simple HTML/CSS/JS dashboard from repo files or an inline response.
- Reuse existing API capabilities where possible:
  - `/v1/memory/list`
  - `/v1/memory/stats/{agent_id}`
  - `/v1/memory/recall`
- Add only small API endpoints if the dashboard cannot function without them.
- Keep it local-first and dependency-light.
- Do not require a cloud service, build step, npm app, or React/Vue.

## Dashboard MVP Features

Required:

- Agent ID input, default `default`
- memory list with id, type, source, importance, created time, short content
- search/recall box
- memory detail panel
- stats panel
- clear loading/error states
- usable at desktop browser size

Optional if small:

- memory type/source filters
- simple "related memories" panel using recall for the selected memory
- first-pass manual link API/UI for memory-to-memory relationships

## Out Of Scope

- marketing landing page
- auth redesign
- heavy frontend framework
- graph visualization library
- dashboard write access beyond the minimal safe controls needed for MVP
- changing OpenClaw config unless needed for validation and explicitly recorded

## Agent Split

Codex:

- Implement the dashboard MVP and any minimal server support.
- Add focused tests where practical.
- Verify the API/server imports cleanly.

Claude:

- Review UX/API safety and fix concrete issues.
- Check request/auth behavior and whether dashboard leaks secrets.
- Check OpenClaw redteam health/status and record warnings.

Kimi:

- Final integration review.
- Run focused verification.
- Confirm the task is complete or write a blocked handoff with exact failures.

## Success Criteria

- `GET /dashboard` returns the dashboard.
- Dashboard can list memories for an agent using existing data.
- Dashboard can recall/search memories.
- Dashboard can show stats for the selected agent.
- No secrets or API keys are exposed in HTML/JS/logs.
- Existing collaboration tests pass.
- OpenClaw redteam health/status is checked and results are recorded.

## Suggested Verification

```bash
python3 -m py_compile arn_v9/api/server.py
python3 -m pytest arn_v9/tests/test_collab.py arn_v9/tests/test_collab_runner.py
openclaw --profile redteam gateway health
openclaw --profile redteam channels status
```

If a FastAPI test client is already available, add or run a focused test that
asserts `/dashboard` returns HTML.

## Follow-Up Task Rule

If manual memory linking is not implemented in this run, the handoff must include
a proposed task:

```text
Proposed task:
- Problem: User needs to explicitly connect related memories.
- Evidence:
- Files:
- Success criteria:
- Verification:
- Suggested owner:
```
