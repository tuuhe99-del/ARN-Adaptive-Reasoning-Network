# ARN — Adaptive Reasoning Network

AI agents forget everything between sessions. ARN fixes that, locally, with no cloud and no monthly bill.

It runs a small server on your machine. Every time your agent talks to a user, ARN stores what happened. Next session, it pulls back what's relevant — not by keyword match but by meaning. Your agent picks up where it left off.

Runs on a Raspberry Pi 5. Costs $0/month. One command to set up.

Hi, I'm Mohamed (MrKali). I built this because I was tired of re-explaining context to my agents every session. It started as a side project on my Pi 5 and turned into something that actually works.

---

## Quick start

**Prerequisites:** Python 3.10+, Mac or Linux (including Raspberry Pi)

```bash
git clone https://github.com/tuuhe99-del/ARN-Adaptive-Reasoning-Network.git
cd ARN-Adaptive-Reasoning-Network
./arn-setup
```

That's it. `arn-setup` installs dependencies, starts the server, installs a launchd service so it auto-starts on login (Mac), and wires the OpenClaw plugin if you're using it. No manual config.

Verify it's running:

```bash
curl http://localhost:8742/v1/health
# → {"status": "ok", "agent_count": 0}
```

Store and recall something:

```bash
curl -X POST http://localhost:8742/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "me", "content": "Mohamed prefers Python for scripting", "importance": 0.8}'

curl -X POST http://localhost:8742/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "me", "query": "what language does the user code in?", "top_k": 3}'
# → returns the Python fact, even though "language" and "code" weren't in the stored text
```

---

## What it does

ARN is a memory server. Your agent stores facts and events, and retrieves them by semantic similarity — meaning, not keyword.

Under the hood:

- **Three memory tiers** — episodic (recent specific events), semantic (repeated patterns consolidated over time), working (current session context). Loosely modeled on how human memory is structured.
- **8 domain-specialized cortical columns** — code, conversation, facts, procedures, preferences, temporal, errors, general. Each column evaluates incoming memories independently, so the system knows the difference between a code snippet and a personal preference.
- **Calibrated surprise scoring** — each domain tracks its own baseline of what's "normal" using Welford's algorithm. Genuinely novel information gets prioritized.
- **Consolidation** — runs as a background task. Clusters similar episodes into semantic memories over time, the way sleep-based consolidation works in humans.
- **Contradiction detection** — when new info conflicts with stored info, it flags the conflict, keeps both, and timestamps them. Doesn't silently overwrite.
- **Temporal tagging** — tag episodes with `time_context='past'|'current'|'future'`. Queries with temporal keywords ("currently", "used to") filter automatically.
- **Protected memories** — episodes stored with `source='api'` are never superseded, decayed, or evicted. Use this for ground-truth facts about a user.

Scoring formula:
```
score = 0.58 × similarity + 0.13 × recency + 0.19 × importance + surprise_bonus − supersession_penalty
```

---

## REST API

Server runs on `http://localhost:8742`. Auth is optional — set `ARN_API_KEY` to require `X-Api-Key` on all writes.

| Method | Path | Auth | What it does |
|--------|------|------|--------------|
| `POST` | `/v1/memory/store` | optional | Store a memory episode |
| `POST` | `/v1/memory/recall` | optional | Retrieve relevant memories by semantic similarity |
| `POST` | `/v1/memory/context` | optional | Get a formatted context block ready to inject into a prompt |
| `POST` | `/v1/memory/exchange` | required | Store a full user + agent exchange in one call |
| `POST` | `/v1/memory/workflow` | required | Store a multi-step tool workflow with results |
| `POST` | `/v1/memory/inject` | required | Inject relevant memories directly into a prompt string |
| `POST` | `/v1/memory/feedback` | required | Send reinforcement signal (thumbs up/down) on a recalled memory |
| `POST` | `/v1/memory/embed_similarity` | required | Compute semantic similarity between two texts |
| `POST` | `/v1/memory/link` / `unlink` / `links` | required | Explicit memory graph — link episodes together |
| `POST` | `/v1/memory/maintain` | required | Manually trigger consolidation |
| `POST` | `/v1/memory/edit` | required | Edit an existing episode |
| `POST` | `/v1/memory/delete` | required | Soft-delete an episode |
| `POST` | `/v1/memory/list` | required | List all episodes for an agent |
| `GET` | `/v1/memory/stats/{agent_id}` | optional | Episode counts, memory tier sizes, scoring stats |
| `GET` | `/v1/health` | none | Health check |
| `DELETE` | `/v1/memory/agent` | required | Wipe all data for an agent |
| `GET` | `/dashboard` | none | Browser dashboard (HTML) |

Each `agent_id` gets fully isolated storage. No cross-agent data leakage.

Rate limiting: token bucket, 60 req/s per IP by default.

---

## OpenClaw plugin

The main integration path for OpenClaw users is the JavaScript plugin at `openclaw-arn-plugin/`. This replaces OpenClaw's markdown memory files (USER.md, MEMORY.md, IDENTITY.md, etc.) with live semantic memory that learns from every interaction.

**What it does automatically:**
- Before every agent turn: retrieves relevant memories and injects them into the prompt
- After every turn: stores user messages, agent replies, tool calls, and tool results
- Labels everything by source: `user`, `agent`, `tool:{name}`, `compaction`
- Deduplicates: won't inject the same memory twice in a session
- Detects topic shifts: when the conversation changes subject, triggers a fresh recall pass
- Persists session state across gateway restarts

**Install:**

```bash
./arn-setup --client openclaw --profile redteam  # adjust profile to match yours
```

Or add manually to your `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "arn-memory": {
        "path": "/path/to/ARN-Adaptive-Reasoning-Network/openclaw-arn-plugin",
        "config": {
          "arnEndpoint": "http://localhost:8742",
          "apiKey": "your-api-key",
          "storeMessages": true,
          "storeTools": true,
          "topK": 5,
          "minScore": 0.35,
          "tokenBudget": 1500,
          "topicShiftThreshold": 0.45
        }
      }
    }
  }
}
```

---

## Model tiers

| Tier | Model | Disk | Speed | Quality |
|------|-------|------|-------|---------|
| `nano` (default) | all-MiniLM-L6-v2 | 22MB | ~30ms | Good |
| `small` | all-mpnet-base-v2 | 420MB | ~60ms | Better |
| `base` | bge-base-en-v1.5 | 440MB | ~80ms | Best retrieval |
| `base-e5` | e5-base-v2 | 440MB | ~80ms | Alternative |

Switch tiers at any time without losing memories:

```bash
./arn-switch-model base   # migrates all stored vectors, zero data loss
```

Set tier at startup:

```bash
export ARN_EMBEDDING_TIER=base
python3 -m uvicorn arn_v9.api.server:app --host 0.0.0.0 --port 8742
```

In stress tests, nano and bge-base both scored 7/7. The bigger model didn't win on any scenario. I'd use nano unless recall quality is specifically a problem for you.

---

## Configuration

| Variable | Default | What it does |
|----------|---------|--------------|
| `ARN_EMBEDDING_TIER` | `nano` | Embedding model tier |
| `ARN_DATA_DIR` | `~/.arn_data` | Where episode databases and vectors are stored |
| `ARN_API_KEY` | *(none)* | If set, all write endpoints require `X-Api-Key` header |
| `ARN_RATE_LIMIT_RPS` | `60` | Max requests per second per IP |
| `ARN_DECAY_INTERVAL_SECONDS` | `3600` | How often the decay loop runs |
| `ARN_CONSOLIDATE_THRESHOLD` | `10` | Episodes needed before consolidation triggers |

Files written:
- `~/.arn_data/{agent_id}/arn_metadata.db` — SQLite episode metadata
- `~/.arn_data/{agent_id}/vectors.npy` — memmap vector store
- `~/.arn_data/.model_fingerprint` — detects silent model swaps between restarts
- `~/.arn_data/session_state.json` — OpenClaw plugin session persistence

---

## Test results

**10/10 on the OpenClaw recall battery** — sequential tests across a real running agent session:

| Test | Scenario | Result |
|------|----------|--------|
| T1 | Identity recall (name, project) | PASS |
| T2 | Tool recall (Ollama, DeepSeek, Gemini) | PASS |
| T3 | ARN description recall | PASS |
| T4 | Language preference (Python) | PASS |
| T5 | Privacy — refuses to hallucinate SSN/bank info | PASS |
| T6 | Hardware recall (Mac, Pi 5, 8GB) | PASS |
| T7 | Cross-session conversation recall | PASS |
| T8 | Project recall from recent sessions | PASS |
| T9 | Workflow memory — store and recall tool steps | PASS |
| T10 | Dynamic recommendation from known setup | PASS |

**7/7 on adversarial stress tests** (`benchmarks/stress_test.py`):

| Test | Result |
|------|--------|
| Cross-session persistence (4 restarts + noise) | PASS |
| Distractor resistance (5 needles in 500 haystack) | PASS |
| Contradiction handling (most-recent-wins) | PASS |
| Temporal reasoning (with tagging) | PASS |
| Hallucination refusal | PASS |
| Paraphrase robustness | PASS |
| Scale (1K and 3K episodes, ~170ms latency) | PASS |

---

## Project structure

```
ARN-Adaptive-Reasoning-Network/
├── arn-setup                  # One-command install
├── arn-switch-model           # One-command model migration
├── install.sh                 # Alternative install script
├── arn_v9/
│   ├── core/
│   │   ├── embeddings.py      # Embedding engine, tier support
│   │   └── cognitive.py       # Memory scoring, cortical columns, consolidation
│   ├── storage/
│   │   └── persistence.py     # SQLite + memmap, protected sources, fingerprinting
│   ├── api/
│   │   └── server.py          # FastAPI REST server, rate limiting
│   ├── plugin.py              # Python API (ARNPlugin class)
│   ├── scripts/
│   │   ├── arn_cli.py         # CLI interface
│   │   └── migrate_to_base_tier.py  # Vector migration tool
│   ├── tests/
│   │   ├── check_env.py       # Pre-flight environment check
│   │   └── test_all.py        # Unit + semantic test suite
│   └── benchmarks/
│       ├── stress_test.py     # Adversarial scenarios
│       └── simulate_agent.py  # 5-day agent simulation
├── openclaw-arn-plugin/       # OpenClaw JS plugin
│   ├── index.js               # Plugin logic (store + inject hooks)
│   └── openclaw.plugin.json   # Plugin manifest
├── scripts/
│   ├── run_arn_battery.sh     # 10-test recall battery
│   └── arn_agent.sh           # OpenClaw agent runner for tests
└── launchd/
    └── com.arn.server.plist   # macOS auto-start service
```

---

## Python API (direct use)

```python
from arn_v9.plugin import ARNPlugin

with ARNPlugin(agent_id="my_agent", data_root="./memory") as p:
    # Store with temporal context
    p.store("User used to prefer Java",
            time_context='past', importance=0.6)
    p.store("User switched to Python last year",
            time_context='current', importance=0.8)

    # Temporal queries filter automatically
    results = p.recall("what does the user currently prefer?")
    # Returns Python as rank 0

    for r in results:
        if r['confidence_tier'] == 'low':
            print("Not enough matching info")
```

Or via the lower-level class:

```python
from arn_v9 import ARNv9

arn = ARNv9(data_dir="./my_agent_memory")
arn.perceive("Deployed on Raspberry Pi 5 with 8GB RAM", importance=0.7)
results = arn.recall("what hardware does the user run?", top_k=3)
arn.close()
```

---

## Known limitations

I'm being upfront because I'd rather you hit these on my docs page than mid-project:

- **No inter-agent memory sharing** — each `agent_id` is isolated. If you need two agents to share knowledge, you'd have to build a sync layer on top. I haven't.
- **Contradiction detection is a word-overlap heuristic** — real NLI would be better. It works for most cases but will miss semantic contradictions that don't share vocabulary.
- **Temporal reasoning requires explicit tagging** — the system can't automatically figure out that a stored fact is outdated. You have to tag it. Auto-inferring this from content is an open problem.
- **Text only** — no images, audio, or structured data.
- **English-tuned by default** — the default models are English-only. Multilingual support means swapping to `paraphrase-multilingual-MiniLM-L12-v2` or similar.
- **workers=1 recommended** — the embedding model is ~90MB per process. Running multiple workers multiplies RAM usage. For higher throughput, put a reverse proxy in front and scale horizontally with separate containers.
- **Scoring thresholds are empirically tuned** — the weights work well in testing but I'm not certain they're the right defaults for every use case. If you tune them, I'd be interested in what you find.

---

## Contributing

If you're looking for somewhere to add real value:

1. **NLI-based contradiction detection** — even a small cross-encoder would beat the word-overlap heuristic
2. **Async consolidation** — it already runs as a background asyncio task, but batching and priority queue improvements would help high-throughput setups
3. **Cross-agent shared semantic layer** — read-only organizational knowledge that multiple agents can draw on
4. **Multilingual embedding support** — swap the default model, ensure the test suite covers non-English recall
5. **LangChain / CrewAI adapters** — I built the OpenClaw plugin because that's what I use. Other frameworks need their own thin wrappers
6. **Mem0/Zep comparison benchmark** — head-to-head on published benchmarks would make this more credible

PRs welcome. If you're unsure whether something fits, open an issue first.

---

## License

**PolyForm Small Business 1.0.0** — see [LICENSE.md](./LICENSE.md) and [COMMERCIAL.md](./COMMERCIAL.md).

Short version:

- **Free** if you're an individual, researcher, hobbyist, or at a company with fewer than 100 people and under $1M revenue
- **Paid license required** if you're at a larger company using this commercially

If you fit the free tier, use it — keep the license file in your fork and you're done. If your company is over the threshold and you want to build on this, open an issue titled "Commercial licensing inquiry."

I picked this over MIT because this project took real work. If it's useful to you personally, I want you to have it free. If a corporation is making money off it, I'd like a share of that.

---

## About

My name is Mohamed Mohamed (MrKali). I built this on a Raspberry Pi 5 I recovered from a corrupted SD card, using OpenClaw as my agent framework.

If you want to reach out, open an issue or reach me through the contacts on my GitHub profile. If you find bugs or have ideas, say so.

Thanks for looking at this.

— Mohamed
