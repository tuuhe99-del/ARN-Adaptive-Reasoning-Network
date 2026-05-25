# OpenClaw ARN Memory Plugin

Inject [ARN v9](https://github.com/mtrt/arn) semantic memory into OpenClaw agent prompts.

## What it does

- **Session startup**: Loads the agent's identity, preferences, and procedures from ARN into a cacheable system-context block.
- **Per-turn**: Queries ARN with the user's current prompt and injects the most relevant memories as dynamic context.
- **Zero markdown files**: Agents no longer need `SOUL.md`, `IDENTITY.md`, or `MEMORY.md` — everything lives in ARN's brain.

## Installation

```bash
# 1. Ensure ARN API server is running
uvicorn arn_v9.api.server:app --host 0.0.0.0 --port 8742

# 2. Copy plugin into OpenClaw extensions
mkdir -p ~/.openclaw/extensions/arn-memory-plugin
cp -r openclaw-arn-plugin/* ~/.openclaw/extensions/arn-memory-plugin/

# 3. Enable in ~/.openclaw/openclaw.json
```

## Configuration

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "arn-memory": {
        "enabled": true,
        "config": {
          "arnEndpoint": "http://localhost:8742",
          "apiKey": "",
          "topK": 5,
          "minScore": 0.3,
          "tokenBudget": 1500
        },
        "hooks": {
          "allowPromptInjection": true
        }
      }
    }
  }
}
```

## How it works

| Hook | What it injects | When |
|------|----------------|------|
| `session_start` | `prependSystemContext` — identity, preferences, procedures | Once per session |
| `before_prompt_build` | `prependContext` — memories relevant to the current user prompt | Every turn |

## Troubleshooting

- **"ARN API failed"**: Make sure `arn_v9.api.server` is running on the configured endpoint.
- **"Plugin blocked"**: Ensure `hooks.allowPromptInjection: true` in the plugin config.
- **Too much token usage**: Lower `tokenBudget` (default 1500 chars ≈ ~375 tokens).
