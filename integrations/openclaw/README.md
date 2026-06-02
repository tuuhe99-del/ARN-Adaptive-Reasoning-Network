# ARN Memory — OpenClaw Plugin

Replaces OpenClaw's built-in memory with persistent cross-session recall powered by hybrid semantic search (vector KNN + FTS5 + entity matching).

## What it does

- **Auto-injects** relevant memories before every LLM call via the `before_prompt_build` hook (priority 40)
- **Captures** all messages, assistant outputs, tool calls, and tool results
- **Manages sessions** — records when conversations start and end, triggers post-session reflection
- **Gives the agent tools** to pin facts, forget stale info, and review flagged contradictions

The core feature is `before_prompt_build`. Before every call to the LLM, the plugin takes the latest user message, searches ARN's memory, and appends relevant results to the system prompt. The agent never calls a tool — memories just appear naturally, like how a human remembers relevant context during conversation.

## Prerequisites

ARN daemon must be running:

```bash
arn server --daemon --port 7900
# or
arn status
```

## Install

```bash
arn connect
```

That's it. `arn connect`:
1. Copies this plugin to `~/.openclaw/plugins/arn-memory/`
2. Installs npm dependencies
3. Registers the plugin with OpenClaw
4. Disables OpenClaw's built-in memory (memory-core)
5. Copies SKILL.md to your OpenClaw workspace skills directory
6. Starts the ARN daemon on port 7900

## Configuration (openclaw.json)

```json
{
  "plugins": {
    "entries": {
      "arn-memory": {
        "config": {
          "arnApiUrl": "http://localhost:7900",
          "maxInjectedMemories": 8,
          "captureToolCalls": true,
          "captureAssistant": true
        }
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `arnApiUrl` | `http://localhost:7900` | ARN server URL |
| `maxInjectedMemories` | `8` | Max memories injected per turn |
| `captureToolCalls` | `true` | Store tool calls and results |
| `captureAssistant` | `true` | Store assistant responses |

## Hook event flow

```
user sends message
  → message_received: store user message (fire-and-forget)
  → before_prompt_build: recall relevant memories → append to system prompt
  → LLM call with injected memories
  → llm_output: store assistant response (fire-and-forget)
  → agent calls tool
    → before_tool_call: store tool call (fire-and-forget)
    → after_tool_call: store tool result (fire-and-forget)
session ends
  → session_end: await session_end API → triggers reflect()
```

## Agent tools

| Tool | When to use |
|------|-------------|
| `arn_recall` | Targeted search by role or session (auto-inject covers most cases) |
| `arn_pin` | Pin a permanent fact (name, preference, critical decision) |
| `arn_forget` | Remove an outdated or incorrect memory |
| `arn_sessions` | List past sessions |
| `arn_review` | Check for flagged contradictions or ambiguities |

## Troubleshooting

**Plugin not loading:** Check `openclaw plugins list`. If not shown, run `arn connect` again.

**ARN not running:** Run `arn status`. If stopped, run `arn server --daemon`.

**No memories injecting:** Memories only inject after the first session stores content. After storing a few messages, recall should work.

**Memories injecting from wrong agent:** The server uses `ARN_AGENT_ID` (default: "default"). Set this env var if you use multiple agents.

## Disconnect

```bash
arn disconnect
```

Restores OpenClaw's built-in memory. Your ARN data is preserved.
