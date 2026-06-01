# ARN — Adaptive Reasoning Network

> **Beta v0.10.0** — this branch is the current development line. The previous stable release is preserved on the `beta-v9` branch.

AI agents forget everything between sessions. ARN fixes that, locally, with no cloud and no monthly bill.

It runs a small server on your machine. Every time your agent talks to a user, ARN stores what happened. Next session, it pulls back what's relevant using three signals at once — vector similarity, full-text search, and entity matching — fused together, then ranked for diversity. Your agent picks up where it left off.

Runs on a Raspberry Pi 5. Costs $0/month. One command to set up.

Hi, I'm Mohamed (MrKali). I built this because I was tired of re-explaining context to my agents every session. It started as a side project on my Pi 5 and turned into something that actually works.

---

## Quick start

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
curl http://localhost:8742/v1/health
# → {"status": "ok", "agent_count": 0}
```

Store and recall something:

```bash
arn store -c "Mohamed prefers Python for scripting" -i 0.8
arn recall -q "what language does the user code in?"
# → returns the Python fact, even though "language" and "code" weren't in the stored text
```

---

## What it does

ARN is a memory server. Your agent stores facts and events, and retrieves them by meaning — not keywords.

**Retrieval stack (v0.10.0):**

- **Vector KNN** via sqlite-vec — semantic similarity using `all-MiniLM-L6-v2` (384-dim embeddings stored in a vec0 virtual table for indexed KNN, not a flat memmap)
- **FTS5 full-text search** — BM25-ranked keyword matching with Porter stemming, catches things vector search misses ("JWT", "Redis", version strings)
- **Entity matching** — extracts proper nouns, quoted strings, code identifiers, file paths, and numbers+units; lets named entities boost recall scores
- **Reciprocal Rank Fusion** — fuses all three ranked lists into a single score without hand-tuning weights
- **Recency decay** — 14-day half-life, applied after fusion; pinned memories bypass decay
- **MMR reranking** — Maximal Marginal Relevance eliminates near-duplicate results so you get diverse answers
- **Score-gap cutoff** — instead of a fixed similarity threshold, finds the largest relative gap in the score distribution and cuts there

**Memory architecture:**

- **Three memory types** — episodic (specific events), semantic (consolidated patterns over time), working (current session context, 7-slot ring buffer that always surfaces in recall)
- **Bi-temporal facts** — every episode has `valid_from` and `valid_until` columns. Superseded facts are kept in history; `recall()` only returns currently valid ones by default
- **Supersedes chains** — when a new fact contradicts a stored one (cosine sim > 0.85, word overlap < 50%), the old episode is soft-invalidated and linked to the new one. You can walk the chain with `arn history <id>`
- **Pinned memories** — pin ground-truth facts that should never decay, be superseded, or removed during consolidation
- **Explicit consolidation** — clustering runs when you call it, not automatically mid-session. `arn consolidate` or `arn.consolidate()` in Python

**Post-session reflection:**

`arn reflect` (or `arn.reflect()`) runs three analysis passes after a session:
1. Scans for near-duplicate episodes with divergent content (contradiction candidates) → queues them for review
2. Recalibrates importance scores based on access frequency (often-accessed facts get a boost)
3. Flags low-importance facts that are accessed often (likely undervalued)

Then runs consolidation. All proposed changes appear in the review queue — you decide what to apply.

---

## CLI

Single entry point: `arn`

```bash
arn setup                             # first-time install + model download
arn store -c "..." -i 0.8            # store a memory (importance 0–1)
arn recall -q "..."                  # retrieve by meaning
arn context -q "..."                 # get a formatted block ready to inject into a prompt
arn pin <id>                         # pin an episode (survives decay + consolidation)
arn unpin <id>                       # unpin
arn history <id>                     # show the supersession chain for an episode
arn forget <id>                      # soft-delete
arn reflect                          # run post-session reflection + populate review queue
arn review                           # list pending review items
arn resolve <review_id> <action>     # action: update / delete / pin / keep_both / defer
arn consolidate                      # explicit consolidation run
arn stats                            # episode counts, tier sizes, queue depth
arn export                           # export all memories to JSON
arn import                           # import from JSON
arn server                           # start the HTTP API server
```

`arn store` options: `-c/--content`, `-i/--importance` (default 0.5), `-a/--agent` (default from `ARN_AGENT_ID`)

`arn recall` options: `-q/--query`, `-k/--top-k` (default 5), `-a/--agent`

`arn resolve` options: `--content "new text"`, `--importance 0.9` (for `update` action)

---

## REST API

Server runs on `http://localhost:8742`. Auth is optional — set `ARN_API_KEY` to require `X-Api-Key` on all writes.

| Method | Path | Auth | What it does |
|--------|------|------|--------------|
| `POST` | `/v1/memory/store` | optional | Store a memory episode |
| `POST` | `/v1/memory/recall` | optional | Retrieve relevant memories |
| `POST` | `/v1/memory/context` | optional | Get a formatted context block for prompt injection |
| `POST` | `/v1/memory/exchange` | required | Store a full user + agent exchange in one call |
| `POST` | `/v1/memory/workflow` | required | Store a multi-step tool workflow with results |
| `POST` | `/v1/memory/inject` | required | Inject relevant memories directly into a prompt string |
| `POST` | `/v1/memory/feedback` | required | Reinforcement signal on a recalled memory |
| `POST` | `/v1/memory/embed_similarity` | required | Semantic similarity between two texts |
| `POST` | `/v1/memory/link` / `unlink` / `links` | required | Explicit memory graph — link episodes together |
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

## OpenClaw plugin

The main integration path for OpenClaw users is in `contrib/openclaw/`. This replaces OpenClaw's markdown memory files (USER.md, MEMORY.md, IDENTITY.md) with live semantic memory that learns from every interaction.

**What it does automatically:**
- Before every agent turn: retrieves relevant memories and injects them into the prompt
- After every turn: stores user messages, agent replies, tool calls, and tool results
- Labels everything by source: `user`, `agent`, `tool:{name}`, `compaction`
- Deduplicates: won't inject the same memory twice in a session
- Detects topic shifts: when the conversation changes subject, triggers a fresh recall pass

**Install:**

```bash
./install.sh --client openclaw --profile redteam  # adjust profile to match yours
```

Or add manually to your `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "arn-memory": {
        "path": "/path/to/ARN-Adaptive-Reasoning-Network/contrib/openclaw",
        "config": {
          "arnEndpoint": "http://localhost:8742",
          "apiKey": "your-api-key",
          "storeMessages": true,
          "storeTools": true,
          "topK": 5,
          "tokenBudget": 1500
        }
      }
    }
  }
}
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
- `~/.arn_data/{agent_id}/arn_metadata.db` — SQLite database (episodes, FTS5 index, vec0 vectors, entities, review queue)

That's it. No `.npy` files, no separate vector store, no fingerprint files. Everything is in the database.

**Schema version:** 6 (`episodes` + `episodes_fts` + `episode_embeddings` + `semantic_embeddings` + `entities` + `memory_review_queue` + `memory_links` + `semantic_nodes` + `system_state` + `schema_version`)

---

## Python API

```python
from arn_v9.plugin import ARNPlugin

with ARNPlugin(agent_id="my_agent", data_root="./memory") as p:
    p.store("User prefers dark mode", importance=0.7)
    p.store("User's main project is ARN", importance=0.9)

    results = p.recall("what project is the user working on?")
    for r in results:
        print(r['content'], r['score'])
```

Or via the lower-level class for full control:

```python
from arn_v9 import ARNv9

arn = ARNv9(data_dir="./my_agent_memory")

# Store
ep_id = arn.perceive("Deployed on Raspberry Pi 5 with 8GB RAM", importance=0.7)['episode_id']

# Retrieve
results = arn.recall("what hardware does the user run?", top_k=3)

# Pin a ground-truth fact
arn.pin(ep_id)

# Update a fact (re-embeds automatically)
arn.update(ep_id, new_content="Deployed on Raspberry Pi 5 with 16GB RAM")

# Walk history after a supersession
chain = arn.get_history(ep_id)

# Soft-delete
arn.forget(ep_id)

# Post-session reflection
stats = arn.reflect()
reviews = arn.get_pending_reviews()
for item in reviews:
    print(item['review_type'], item['content'])

# Resolve a review (update / delete / pin / keep_both / defer)
arn.resolve_review(item['id'], 'pin')

arn.close()
```

---

## Project structure

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
│   │   └── server.py               # FastAPI REST server
│   ├── plugin.py                   # ARNPlugin — high-level Python API
│   ├── scripts/
│   │   └── arn_cli.py              # arn CLI (store, recall, pin, reflect, review, resolve, ...)
│   ├── tests/
│   │   └── test_all.py             # Unit + integration test suite (15 tests)
│   └── benchmarks/
│       └── stress_test.py          # Adversarial recall scenarios
├── contrib/
│   └── openclaw/                   # OpenClaw JS plugin (was openclaw_skill/)
│       └── SKILL.md
└── deploy/
    └── Dockerfile                  # Docker deployment
```

---

## Known limitations

- **No inter-agent memory sharing** — each `agent_id` is isolated. Two agents sharing knowledge requires a sync layer on top.
- **Contradiction detection is structural, not semantic** — when cosine sim > 0.85 and word overlap < 50%, the system flags a supersession. It won't catch contradictions that are semantically opposite but similarly worded.
- **Text only** — no images, audio, or structured data.
- **English-tuned by default** — `all-MiniLM-L6-v2` is English-optimized. Multilingual support means swapping to `paraphrase-multilingual-MiniLM-L12-v2` and passing a custom `embedding_fn`.
- **workers=1 recommended** — the embedding model is ~22MB per process but sentence-transformers loads ~500MB of PyTorch. Multiple workers multiply that. Scale horizontally with separate containers + a reverse proxy.

---

## Contributing

If you're looking for somewhere to add real value:

1. **NLI-based contradiction detection** — a small cross-encoder would replace the cosine+overlap heuristic with actual entailment checking
2. **Async consolidation** — runs synchronously when called; a priority queue with background batching would help high-throughput setups
3. **Cross-agent shared semantic layer** — read-only organizational knowledge multiple agents can draw on
4. **Multilingual embedding support** — swap the default model, ensure test suite covers non-English recall
5. **LangChain / CrewAI adapters** — I built the OpenClaw plugin because that's what I use; other frameworks need thin wrappers
6. **Mem0/Zep comparison benchmark** — head-to-head on published benchmarks would make this more credible

PRs welcome. If you're unsure whether something fits, open an issue first.

---

## License

**PolyForm Small Business 1.0.0** — see [LICENSE.md](./LICENSE.md) and [COMMERCIAL.md](./COMMERCIAL.md).

Short version:

- **Free** if you're an individual, researcher, hobbyist, or at a company with fewer than 100 people and under $1M revenue
- **Paid license required** if you're at a larger company using this commercially

If you fit the free tier, use it — keep the license file in your fork and you're done. If your company is over the threshold and you want to build on this, open an issue titled "Commercial licensing inquiry."

---

## About

My name is Mohamed Mohamed (MrKali). I built this on a Raspberry Pi 5, using OpenClaw as my agent framework.

If you want to reach out, open an issue or reach me through the contacts on my GitHub profile.

— Mohamed
