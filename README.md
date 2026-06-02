# ARN — Adaptive Reasoning Network

> **Beta v0.10.0** — current development line. Previous stable release preserved on the `beta-v9` branch.

AI agents forget everything between sessions. ARN fixes that — locally, with no cloud and no monthly bill.

A lightweight server runs on your machine. Every time your agent talks to a user, ARN stores what happened. Next session, it pulls back what's relevant using three signals at once: vector similarity, full-text search, and entity matching — fused together, then ranked for diversity. Your agent picks up where it left off.

Runs on a Raspberry Pi 5. Costs $0/month. One command to set up.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Agent                           │
│  (OpenClaw / LangChain / Python script / CLI)               │
└───────────────┬─────────────────┬───────────────────────────┘
                │ perceive()      │ recall()
                ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      ARN Core (Python)                      │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  EmbedEngine │   │  Retrieval   │   │   Reflection   │  │
│  │ MiniLM-L6-v2 │   │  Pipeline    │   │  scan/recalib  │  │
│  │   384-dim    │   │  RRF + MMR   │   │  review queue  │  │
│  └──────┬───────┘   └──────┬───────┘   └────────────────┘  │
│         │                  │                                 │
└─────────┼──────────────────┼─────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  SQLite Database (single file)              │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  episodes   │  │episode_embed │  │  episodes_fts    │   │
│  │  (rows)     │  │(vec0 KNN)    │  │  (FTS5/porter)   │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  sessions   │  │  entities    │  │ memory_review_   │   │
│  │  (CRUD)     │  │(proper nouns)│  │ queue            │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Retrieval Pipeline

Every `recall()` call runs through this chain:

```
query text
    │
    ├──► Vector KNN (sqlite-vec)       top-20 by cosine sim
    │         ┌─────────────┐
    ├──► FTS5 full-text      │         top-20 by BM25 rank
    │         │              │
    └──► Entity matching     │         top-20 by entity overlap
                │            │
                ▼            ▼
        ┌──────────────────────┐
        │  Reciprocal Rank     │       fuse all 3 ranked lists
        │  Fusion (RRF)        │       without hand-tuning weights
        └──────────┬───────────┘
                   │
                   ▼ composite score
        rrf_score
          + recency_decay × 0.3        (14-day half-life; pinned = 1.0)
          + importance × 0.15
          + log(1 + access_count) × 0.05
          + pin_boost (0.15 if pinned)
                   │
                   ▼
        ┌──────────────────────┐
        │  MMR Reranking       │       eliminate near-duplicate results
        │  (λ = 0.7)           │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Score-Gap Cutoff    │       adaptive threshold — cuts at the
        │                      │       largest relative gap in score dist
        └──────────┬───────────┘
                   │
                   ▼
             final results
```

---

## Memory Types

```
┌─────────────────────────────────────────────────────────────────┐
│                       Memory Architecture                       │
│                                                                 │
│  Working Memory (7-slot ring buffer)                            │
│  ┌───┬───┬───┬───┬───┬───┬───┐                                 │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  ← current session context     │
│  └───┴───┴───┴───┴───┴───┴───┘    always surfaces in recall    │
│                                                                 │
│  Long-term Memory (SQLite)                                      │
│  ┌──────────────────┬────────────────────────────────────────┐  │
│  │ Episodic         │ specific events, conversations, facts   │  │
│  │ Semantic         │ consolidated patterns (post-reflect)    │  │
│  │ Pinned           │ bypass decay + consolidation forever    │  │
│  └──────────────────┴────────────────────────────────────────┘  │
│                                                                 │
│  Bi-temporal columns on every episode                           │
│  valid_from ──────────────────► valid_until (NULL = still valid)│
│                                                                 │
│  Supersedes chain                                               │
│  [old fact] ──superseded_by──► [new fact]                       │
│  old: valid_until = NOW        new: valid_from = NOW            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Prerequisites:** Python 3.10+, Mac or Linux (including Raspberry Pi)

```bash
git clone https://github.com/tuuhe99-del/ARN-Adaptive-Reasoning-Network.git
cd ARN-Adaptive-Reasoning-Network
./install.sh
```

`install.sh` installs dependencies, downloads `all-MiniLM-L6-v2` (22MB), and sets up the `arn` command.

Verify it's running:

```bash
arn server &
curl http://localhost:7900/v1/health
# → {"status": "ok", "episodes": 0, "sessions": 0, "db_size_mb": 0.1}
```

Store and recall something:

```bash
arn store -c "User prefers Python for scripting" -i 0.8
arn recall -q "what language does the user code in?"
# → returns the Python fact, even though "language" and "code" weren't in the stored text
```

---

## How It Works

### Storing a memory

```bash
arn store -c "content here" -i 0.7
```

Internally:
1. Text → 384-dim embedding via `all-MiniLM-L6-v2`
2. Episode row inserted (content, importance, valid_from, role, session_id)
3. Vector inserted into `episode_embeddings` (vec0 virtual table)
4. FTS5 index updated via trigger
5. Entities extracted and stored in `entities` table
6. Working memory slot updated

### Recalling memories

```bash
arn recall -q "query here" -k 5
```

Runs the full pipeline shown in the diagram above. Returns diverse, relevant results — not just keyword matches.

### Pinning a fact

```bash
arn pin <episode_id>
```

Pinned episodes:
- Bypass recency decay (always score as if just created)
- Are never removed by consolidation
- Are never superseded by new facts
- Survive `arn reflect`

### Session lifecycle

```
arn server --daemon              # start daemon (background)

# Your agent runs:
POST /session/start              # records session start
POST /perceive  (role=user)      # every user message
POST /perceive  (role=assistant) # every agent reply
POST /perceive  (role=tool_call) # every tool invocation
POST /perceive  (role=tool_result)

POST /session/end                # triggers reflect(), closes session
                                 # episode_count updated automatically
```

### Post-session reflection

```bash
arn reflect
```

Three passes:
1. **Contradiction scan** — finds episode pairs with sim > 0.85 + word overlap < 40%. Queues them for review.
2. **Importance recalibration** — episodes accessed ≥ 5 times get an importance boost (capped at 0.95).
3. **Ambiguity detection** — low-importance episodes accessed frequently are flagged as likely undervalued.

Then runs consolidation (only merges episodes with sim > 0.90 — near-identical, not just similar).

Review what was flagged:

```bash
arn review
# Lists pending items with episode content and reason

arn resolve <review_id> keep_both
arn resolve <review_id> delete
arn resolve <review_id> pin
arn resolve <review_id> update --content "corrected text" --importance 0.9
arn resolve <review_id> defer
```

---

## CLI Reference

Single entry point: `arn`

```bash
# Memory operations
arn store -c "..." -i 0.8            # store a memory (importance 0–1)
arn recall -q "..."                  # retrieve by meaning
arn context -q "..."                 # formatted block for prompt injection
arn forget <id>                      # soft-delete
arn pin <id>                         # pin (survives decay + consolidation)
arn unpin <id>                       # unpin
arn history <id>                     # supersession chain for an episode

# Post-session workflow
arn reflect                          # run reflection + populate review queue
arn review                           # list pending review items
arn resolve <id> <action>            # action: update / delete / pin / keep_both / defer
arn consolidate                      # explicit consolidation run

# Data management
arn stats                            # episode counts, tier sizes, queue depth
arn export                           # export all memories to JSON
arn import                           # import from JSON

# Server
arn server                           # start HTTP API server (foreground)
arn server --daemon --port 7900      # start as background daemon
arn server --stop                    # stop daemon
arn status                           # daemon status and stats

# OpenClaw integration
arn connect                          # wire up OpenClaw integration
arn disconnect                       # remove OpenClaw integration
```

`arn store` options: `-c/--content`, `-i/--importance` (default 0.5), `-a/--agent`

`arn recall` options: `-q/--query`, `-k/--top-k` (default 5), `-a/--agent`

`arn resolve` options: `--content "new text"`, `--importance 0.9` (for `update` action)

---

## OpenClaw Integration

ARN replaces OpenClaw's built-in memory with automatic cross-session recall.

### Setup

```bash
arn server --daemon --port 7900   # start daemon first
arn connect                       # wire up the plugin
```

`arn connect`:
1. Copies the plugin from `integrations/openclaw/` to `~/.openclaw/plugins/arn-memory/`
2. Installs npm dependencies
3. Registers the plugin with OpenClaw
4. Disables OpenClaw's built-in memory (memory-core)
5. Copies SKILL.md to your OpenClaw workspace

### How the plugin works

```
user sends message
    │
    ├──► message_received hook
    │         store(content, role="user")  ← fire-and-forget
    │
    ├──► before_prompt_build hook (priority 40)
    │         recall(last_user_message, top_k=8)
    │         append relevant results to system prompt
    │         ← agent sees memories naturally, no tool call needed
    │
    ├──► LLM call with injected memories
    │
    ├──► llm_output hook
    │         store(response, role="assistant")  ← fire-and-forget
    │
    └──► tool lifecycle (if tools called)
              before_tool_call → store(call, role="tool_call")
              after_tool_call  → store(result, role="tool_result")

session ends
    └──► session_end hook → POST /session/end → triggers reflect()
```

### Agent tools

The plugin registers 5 tools the agent can call directly:

| Tool | When to use |
|------|-------------|
| `arn_recall` | Targeted search by role or session |
| `arn_pin` | Pin a permanent fact (name, preference, decision) |
| `arn_forget` | Remove an outdated or incorrect memory |
| `arn_sessions` | List past sessions |
| `arn_review` | Check flagged contradictions or ambiguities |

Auto-inject via `before_prompt_build` covers most cases — these tools are for explicit control.

### Plugin configuration (`openclaw.json`)

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

### Disconnect

```bash
arn disconnect
```

Restores OpenClaw's built-in memory. ARN data is preserved.

---

## Plugin API (port 7900)

The OpenClaw plugin communicates with ARN on port 7900. No auth required. No `agent_id` — uses `ARN_AGENT_ID` (default: `"default"`).

| Method | Path | What it does |
|--------|------|--------------|
| `POST` | `/perceive` | Store a memory with role + session context |
| `POST` | `/recall` | Recall memories with optional role_filter |
| `POST` | `/session/start` | Start a session record |
| `POST` | `/session/end` | End session, trigger reflect() |
| `GET` | `/sessions/recent` | List recent sessions |
| `GET` | `/session/{id}` | Session detail with role breakdown |
| `POST` | `/pin` | Pin an episode |
| `POST` | `/unpin` | Unpin |
| `POST` | `/forget` | Soft-delete |
| `GET` | `/reviews/pending` | Pending review queue |
| `POST` | `/reviews/resolve` | Resolve a review item |
| `GET` | `/v1/health` | Health + episode/session counts |

### Role values

| Role | What it represents |
|------|-------------------|
| `user` | User message |
| `assistant` | Agent/LLM response |
| `tool_call` | Tool invocation |
| `tool_result` | Tool output |
| `compaction_marker` | Context compaction event |
| `user_identity` | Stable facts about the user (highest importance: 0.9) |
| `semantic` | Consolidated semantic knowledge |
| `episodic` | General episodic memory |

---

## REST API

Server runs on `http://localhost:8742` by default (OpenClaw plugin uses port 7900). Auth is optional — set `ARN_API_KEY` to require `X-Api-Key` on all writes.

| Method | Path | Auth | What it does |
|--------|------|------|--------------|
| `POST` | `/v1/memory/store` | optional | Store a memory episode |
| `POST` | `/v1/memory/recall` | optional | Retrieve relevant memories |
| `POST` | `/v1/memory/context` | optional | Formatted context block for prompt injection |
| `POST` | `/v1/memory/exchange` | required | Store a full user + agent exchange |
| `POST` | `/v1/memory/workflow` | required | Store a multi-step tool workflow |
| `POST` | `/v1/memory/inject` | required | Inject memories into a prompt string |
| `POST` | `/v1/memory/feedback` | required | Reinforcement signal on a recalled memory |
| `POST` | `/v1/memory/embed_similarity` | required | Semantic similarity between two texts |
| `POST` | `/v1/memory/link` / `unlink` / `links` | required | Explicit memory graph — link episodes |
| `POST` | `/v1/memory/consolidate` | required | Trigger consolidation |
| `POST` | `/v1/memory/edit` | required | Edit an existing episode |
| `POST` | `/v1/memory/delete` | required | Soft-delete an episode |
| `POST` | `/v1/memory/list` | required | List all episodes for an agent |
| `GET` | `/v1/memory/stats/{agent_id}` | optional | Episode counts, memory tier sizes |
| `GET` | `/v1/health` | none | Health check |
| `DELETE` | `/v1/memory/agent` | required | Wipe all data for an agent |
| `GET` | `/dashboard` | none | Browser dashboard (HTML) |

Each `agent_id` gets fully isolated storage. No cross-agent data leakage.

---

## Python API

High-level (recommended for most use cases):

```python
from arn_v9.plugin import ARNPlugin

with ARNPlugin(agent_id="my_agent", data_root="./memory") as p:
    p.store("User prefers dark mode", importance=0.7)
    p.store("User's main project is ARN", importance=0.9)

    results = p.recall("what project is the user working on?")
    for r in results:
        print(r['content'], r['score'])
```

Low-level (full control):

```python
from arn_v9 import ARNv9

arn = ARNv9(data_dir="./my_agent_memory")

# Store
ep_id = arn.perceive("Deployed on Raspberry Pi 5 with 8GB RAM", importance=0.7)['episode_id']

# Retrieve
results = arn.recall("what hardware does the user run?", top_k=3)

# Pin a ground-truth fact
arn.pin(ep_id)

# Update (re-embeds automatically)
arn.update(ep_id, new_content="Deployed on Raspberry Pi 5 with 16GB RAM")

# Walk the supersession chain
chain = arn.get_history(ep_id)

# Soft-delete
arn.forget(ep_id)

# Post-session reflection
stats = arn.reflect()
reviews = arn.get_pending_reviews()
for item in reviews:
    print(item['review_type'], item['content'])

# Resolve a review
arn.resolve_review(item['id'], 'pin')

arn.close()
```

### Role-aware storage (sessions)

```python
import numpy as np

# Start a session
arn.storage.create_session("sess-001", reason_start="user opened chat")

# Store with role tagging
vec = arn.embedder.encode("What's the weather like?")
arn.storage.store_episode(
    content="What's the weather like?",
    vector=vec,
    role="user",
    session_id="sess-001",
    importance=0.5,
)

# End session (triggers episode_count update)
arn.storage.end_session("sess-001", reason_end="user closed chat")

# Get session history
episodes = arn.storage.get_session_episodes("sess-001")
```

---

## Configuration

| Variable | Default | What it does |
|----------|---------|--------------|
| `ARN_DATA_DIR` | `~/.arn_data` | Where episode databases are stored |
| `ARN_AGENT_ID` | `default` | Default agent ID for CLI commands |
| `ARN_API_KEY` | *(none)* | If set, all write endpoints require `X-Api-Key` header |
| `ARN_RATE_LIMIT_RPS` | `60` | Max requests per second per IP |
| `ARN_DECAY_INTERVAL_SECONDS` | `3600` | How often the decay loop runs |

Files written per agent:
- `~/.arn_data/{agent_id}/arn_metadata.db` — SQLite database (all data in one file)

That's it. No `.npy` files, no separate vector store, no fingerprint files. Everything is in the database.

**Schema version:** 7

Tables: `episodes` · `episodes_fts` · `episode_embeddings` · `semantic_embeddings` · `entities` · `sessions` · `memory_review_queue` · `memory_links` · `semantic_nodes` · `system_state` · `schema_version`

---

## Project Structure

```
ARN-Adaptive-Reasoning-Network/
├── install.sh                      # Install script
├── arn_v9/
│   ├── core/
│   │   ├── cognitive.py            # ARNv9 class — perceive, recall, reflect, pin, forget, update
│   │   ├── embeddings.py           # EmbeddingEngine (all-MiniLM-L6-v2, 384-dim)
│   │   ├── retrieval.py            # fuse_rrf, recency_score, mmr_rerank, score_gap_cutoff
│   │   ├── entities.py             # extract_entities (proper nouns, code, URLs, paths, numbers)
│   │   └── reflect.py              # scan_contradictions, recalibrate_importance, detect_ambiguity
│   ├── storage/
│   │   └── persistence.py          # SQLite + sqlite-vec + FTS5, all storage ops
│   ├── api/
│   │   └── server.py               # FastAPI REST server + plugin endpoints + daemon
│   ├── plugin.py                   # ARNPlugin — high-level Python API
│   ├── scripts/
│   │   └── arn_cli.py              # arn CLI entry point
│   ├── tests/
│   │   └── test_all.py             # Unit + integration test suite
│   └── benchmarks/
│       └── stress_test.py          # Adversarial recall scenarios
├── integrations/
│   └── openclaw/                   # OpenClaw TypeScript plugin
│       ├── index.ts                # Plugin entry point (hooks + tools)
│       ├── package.json
│       ├── openclaw.plugin.json    # Plugin manifest
│       ├── SKILL.md                # Agent guidance (auto-copied by arn connect)
│       └── README.md               # Integration docs
├── tests/
│   └── test_openclaw_integration.py  # End-to-end pipeline tests
├── deploy/
│   └── Dockerfile                  # Docker deployment
└── contrib/                        # Experimental / community
```

---

## Reflect Workflow

```
arn reflect
     │
     ├─► scan_contradictions()
     │       get top-200 active unpinned episodes
     │       pairwise cosine similarity
     │       flag pairs: sim > 0.85 AND word-overlap < 40%
     │       → enqueue_review(type='contradiction', priority=sim_score)
     │
     ├─► recalibrate_importance()
     │       find episodes with access_count ≥ 5
     │       boost importance by (access_count // 5) × 0.05, cap 0.95
     │       apply updates immediately
     │
     ├─► detect_ambiguity()
     │       flag: access_count > 3 AND importance < 0.2 AND not invalidated
     │       → enqueue_review(type='ambiguous', priority=0.3)
     │
     └─► consolidate()
             cluster episodes by sim > 0.90 (near-identical only)
             skip: pinned, in review queue, created < 7 days ago
             merge survivors → semantic node

arn review
     ├── contradiction: old_episode ↔ new_episode
     ├── ambiguous: undervalued high-access episode
     └── (resolve each with: update / delete / pin / keep_both / defer)
```

---

## Known Limitations

- **No inter-agent memory sharing** — each `agent_id` is isolated. Sharing knowledge between agents requires a sync layer on top.
- **Contradiction detection is structural, not semantic** — cosine sim > 0.85 + word overlap < 50% flags supersessions. Semantically opposite but similarly-worded facts won't be caught.
- **Text only** — no images, audio, or structured data.
- **English-tuned by default** — `all-MiniLM-L6-v2` is English-optimized. Multilingual support means swapping to `paraphrase-multilingual-MiniLM-L12-v2` and passing a custom `embedding_fn`.
- **workers=1 recommended** — sentence-transformers loads ~500MB of PyTorch per process. Scale horizontally with separate containers + a reverse proxy.

---

## Contributing

Areas where contributions add real value:

1. **NLI-based contradiction detection** — a small cross-encoder would replace the cosine+overlap heuristic with actual entailment checking
2. **Async consolidation** — runs synchronously when called; a priority queue with background batching would help high-throughput setups
3. **Cross-agent shared semantic layer** — read-only organizational knowledge multiple agents can draw on
4. **Multilingual embedding support** — swap the default model, ensure the test suite covers non-English recall
5. **LangChain / CrewAI adapters** — thin wrappers adapting ARN's perceive/recall interface to other agent frameworks
6. **Mem0/Zep comparison benchmark** — head-to-head on published benchmarks

PRs welcome. Open an issue first if you're unsure whether something fits.

---

## License

**PolyForm Small Business 1.0.0** — see [LICENSE.md](./LICENSE.md) and [COMMERCIAL.md](./COMMERCIAL.md).

- **Free** if you're an individual, researcher, hobbyist, or at a company with fewer than 100 people and under $1M revenue
- **Paid license required** if you're at a larger company using this commercially

If you fit the free tier, use it — keep the license file in your fork and you're done. Commercial inquiries: open an issue titled "Commercial licensing inquiry."
