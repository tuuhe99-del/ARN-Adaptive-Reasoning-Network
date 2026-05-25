---
name: arn-memory
description: >
  Persistent semantic memory for the agent using ARN v9. Use whenever the
  agent needs to remember user facts, recall past conversations, store
  preferences or project details, retrieve relevant context for responses,
  check if new info contradicts stored info, or reason about what the
  user used to do vs what they currently do.
metadata:
  openclaw:
    requires:
      bins:
        - python3
    os:
      - linux
    emoji: "🧠"
    always: true
---

# ARN v9 Memory Skill

You have persistent, brain-inspired memory powered by ARN v9. It survives
across sessions and restarts. Use it to remember facts, recall context,
and build long-term knowledge about the user.

## When to store

Store information whenever you learn something worth remembering:

1. User shares identity info (name, role, location) → importance 0.9
2. User states a preference (language, tool, style) → importance 0.7-0.8
3. A project decision is made or constraint established → importance 0.8
4. A bug is found and fixed → importance 0.7
5. A technical approach is chosen → importance 0.6
6. Routine conversation (greetings, thanks) → importance 0.1-0.2

## When to recall

Recall BEFORE answering whenever:

1. User asks about something previously discussed
2. User references "we", "our", "last time", "remember", "what did I say about"
3. You need user-specific context (preferences, constraints)
4. You need to check if new info contradicts stored info

## Commands

### Store a memory

```bash
python3 ~/arn_v9/scripts/arn_cli.py --strict store \
  --content "<the fact or observation>" \
  --importance <0.0-1.0> \
  --tags "<comma,separated,tags>" \
  --time-context "<past|current|future>" \
  --source "conversation"
```

**Important — use `--time-context` correctly:**

- `current` (default) — facts true right now ("User prefers Python")
- `past` — facts about what used to be true ("User used to work at TechCo")
- `future` — plans or intentions ("User plans to switch to Rust next month")

This is critical. Embedding models cannot distinguish "used to prefer X" from
"currently prefers Y" by text alone. You tagging it correctly is what lets
recall filter properly when the user asks about the present vs past.

Examples:

```bash
# User just told you their current preference
python3 ~/arn_v9/scripts/arn_cli.py --strict store \
  --content "User prefers Python for scripting" \
  --importance 0.8 --time-context current

# User mentioned what they used to do
python3 ~/arn_v9/scripts/arn_cli.py --strict store \
  --content "User used to work at Anthropic before going independent" \
  --importance 0.7 --time-context past

# User shared a future plan
python3 ~/arn_v9/scripts/arn_cli.py --strict store \
  --content "User plans to deploy v2 next Tuesday" \
  --importance 0.8 --time-context future
```

### Recall memories

```bash
python3 ~/arn_v9/scripts/arn_cli.py --strict recall \
  --query "<natural language query>" \
  --top-k 5
```

Output is JSON array. Each result has:
- `content` — the stored text
- `score` — ranking score (higher = better match)
- `similarity` — raw semantic similarity (0.0 to 1.0)
- `confidence_tier` — `"high"`, `"medium"`, or `"low"`
- `time_context` — `"past"`, `"current"`, or `"future"` (if tagged)

**Trust the confidence_tier.** If it's `"low"`, the system is telling you it
doesn't actually know — don't hallucinate an answer based on a loose match.

### Get formatted context for prompt injection

```bash
python3 ~/arn_v9/scripts/arn_cli.py --strict context \
  --query "<current conversation topic>" \
  --max-tokens 1000
```

Returns a formatted markdown block with working memory + relevant memories.
Inject this into your context when you need comprehensive recall.

### Run maintenance

```bash
python3 ~/arn_v9/scripts/arn_cli.py maintain
```

Call during idle time. Consolidates episodic memories into semantic knowledge.

### Check system stats

```bash
python3 ~/arn_v9/scripts/arn_cli.py stats
```

## Rules

1. ALWAYS use `--time-context current` for facts true right now, `past` for
   old facts, `future` for plans. This is the #1 source of recall errors
   if you get it wrong.
2. ALWAYS store important user facts immediately — don't wait.
3. ALWAYS check `confidence_tier` before using recall results.
   - `high` → state the fact plainly
   - `medium` → say "I think you mentioned..." or similar hedge
   - `low` → do NOT use this result. Say you don't know, or ask.
4. NEVER store passwords, API keys, tokens, or credentials.
5. When recall returns `has_contradictions: true`, mention the conflict exists
   and go with the most recent version.
6. Don't run `maintain` mid-conversation — it takes ~50ms but blocks.
7. Set importance scores honestly. Not everything is 1.0.
8. The `--source` flag should be `"conversation"` for chat, `"tool"` for
   tool-generated facts, or `"system"` for configuration info.
9. Use `--strict` on every command. It makes the CLI exit with error
   code 1 if the embedding model is unavailable, instead of returning
   garbage that looks valid.

## Output format when sharing with user

When sharing recalled memories with the user, present them naturally:

- Do NOT show raw JSON output
- Do NOT mention "ARN", "episodic memory", "semantic memory", or "confidence tier"
- Simply incorporate the recalled facts into your response
- If confidence is low, hedge appropriately ("I might be misremembering, but...")
- If contradictions exist, mention them ("You mentioned X before, but more recently said Y")
- If the user asks what you remember, list the facts cleanly
