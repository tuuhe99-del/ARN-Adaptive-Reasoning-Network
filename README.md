# ARN v9 — Adaptive Reasoning Network

Brain-inspired persistent memory for AI agents. Runs locally, costs $0/month, fits in under 50MB on a Raspberry Pi 5.

Hi, I'm Mohamed. I'm a student at Columbus State Community College studying IT, and I built this because I wanted my AI agents to actually remember things between sessions without paying for a cloud service. It started as a side project on my Pi 5 and turned into something I think is worth sharing.

If you're building agents with OpenClaw, LangChain, or any framework where you keep re-explaining context to your agent every session, this might help.

## What it does

Your agent stores facts, the system remembers them across sessions, and when the agent needs context it gets back the relevant stuff — not by keyword match but by meaning. Ask "what does the user like to code in?" after storing "Mohamed prefers Python" and you get the Python answer back, even though "like" and "code" don't appear in the stored fact.

Under the hood it's a combination of things from neuroscience and ML:

- **Episodic + semantic split** — like hippocampus vs neocortex. Recent stuff is stored as specific episodes, repeated patterns get consolidated into general knowledge
- **Sentence embeddings** — using all-MiniLM-L6-v2 (22MB) by default, with optional upgrades to bge-base (440MB) if you want higher quality
- **Cortical column voting** — 8 domain-specialized columns (code, conversation, facts, errors, etc.) that each evaluate incoming info
- **Calibrated prediction error** — tracks what counts as "surprising" per domain using Welford's algorithm, so truly novel info gets prioritized
- **Consolidation** — periodically clusters similar episodes into semantic memories (like what happens during sleep)
- **Contradiction detection** — when new info conflicts with stored info, it flags the conflict and keeps both with timestamps
- **Explicit temporal tagging** — because embeddings alone can't tell "used to prefer X" from "currently prefers Y", you can tag episodes with `time_context='past'|'current'|'future'`

## What it passes

Full results from the adversarial stress test (`benchmarks/stress_test.py`):

| Test | Result |
|------|--------|
| Cross-session persistence (4 restarts + noise) | ✅ 3/3 facts recalled |
| Distractor resistance (5 needles in 500 haystack) | ✅ 5/5 found |
| Contradiction handling (most-recent-wins) | ✅ latest version wins |
| Temporal reasoning (with tagging) | ✅ current > past |
| Hallucination refusal | ✅ 3/3 flagged low-confidence |
| Paraphrase robustness | ✅ 6/6 reworded queries hit |
| Scale (1K and 3K episodes) | ✅ 100% accuracy, ~170ms latency |

Plus 40/40 on the main test suite.

## Quick start

```bash
pip install sentence-transformers numpy
git clone https://github.com/tuuhe99-del/ARN-Adaptive-Reasoning-Network.git
cd ARN-Adaptive-Reasoning-Network
python3 -m arn_v9.tests.check_env   # verify environment
python3 -m arn_v9.tests.test_all    # run tests
```

### Basic usage

```python
from arn_v9 import ARNv9

arn = ARNv9(data_dir="./my_agent_memory")

# Store facts
arn.perceive("User prefers Python for scripting", importance=0.8)
arn.perceive("Deployed on Raspberry Pi 5 with 8GB RAM", importance=0.7)

# Recall by meaning
results = arn.recall("what does the user code in?", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['content']}")

arn.close()
```

### With the plugin API (temporal tagging + confidence tiers)

```python
from arn_v9.plugin import ARNPlugin

with ARNPlugin(agent_id="my_agent", data_root="./memory") as p:
    # Tag facts with when they're true
    p.store("User used to prefer Python",
            time_context='past', importance=0.6)
    p.store("User now exclusively uses Rust",
            time_context='current', importance=0.8)

    # Queries with temporal keywords get filtered automatically
    results = p.recall("what does the user currently prefer?")
    # Returns Rust as rank 0, not Python

    # Check if recall is confident
    for r in results:
        if r['confidence_tier'] == 'low':
            print("I don't know — not enough matching info")
```

### REST API

```bash
# Run the server
python3 -m uvicorn arn_v9.api.server:app --host 0.0.0.0 --port 8742

# Or via Docker
docker build -t arn-v9 .
docker run -p 8742:8742 -v arn_data:/data arn-v9
```

Then:

```bash
curl -X POST http://localhost:8742/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "user1", "content": "User likes dark mode", "importance": 0.5}'

curl -X POST http://localhost:8742/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "user1", "query": "UI preferences", "top_k": 3}'
```

### Model tiers

You can pick the embedding model based on your constraints:

| Tier | Model | Size | Speed | Quality |
|------|-------|------|-------|---------|
| `nano` (default) | all-MiniLM-L6-v2 | 22MB | ~30ms | Good |
| `small` | all-mpnet-base-v2 | 420MB | ~60ms | Better |
| `base` | bge-base-en-v1.5 | 440MB | ~80ms | Best retrieval |
| `base-e5` | e5-base-v2 | 440MB | ~80ms | Alt. retrieval |

```python
# Set via env
export ARN_EMBEDDING_TIER=base

# Or at init
arn = ARNv9(embedding_tier='base')
```

In my stress tests, nano and bge-base both scored 7/7. The bigger model didn't win on any scenario. I'd stick with nano unless you have a specific reason.

## Project structure

```
arn_v9/
├── core/
│   ├── embeddings.py      # Embedding engine with tier support
│   └── cognitive.py       # Main ARN, domain columns, working memory
├── storage/
│   └── persistence.py     # SQLite + memmap vectors
├── api/
│   └── server.py          # FastAPI REST wrapper
├── plugin.py              # OpenClaw-compatible plugin interface
├── scripts/
│   └── arn_cli.py         # CLI for shell/SKILL.md integration
├── openclaw_skill/
│   └── SKILL.md           # Ready-to-install OpenClaw skill
├── tests/
│   ├── check_env.py       # Pre-flight check
│   └── test_all.py        # 40 tests, tier 1 (plumbing) + tier 2 (semantic)
└── benchmarks/
    ├── stress_test.py     # 7 adversarial scenarios
    └── simulate_agent.py  # 5-day agent simulation
```

## OpenClaw integration

Drop `openclaw_skill/SKILL.md` into `~/.openclaw/skills/arn-memory/` and the skill is auto-discovered. Your agent gets store/recall/context commands that invoke `scripts/arn_cli.py` under the hood.

See `ARN_V9_OPENCLAW_INTEGRATION_GUIDE.md` in the repo root for the full setup.

## What I know is rough

I'm being upfront about this because I want people to build on it, not waste time on dead ends:

- **No inter-agent memory sharing** — each agent_id gets isolated storage. If you want two agents to share knowledge, you'd need to build a sync layer. I haven't.
- **Consolidation is synchronous** — it runs on the main thread. Fine for small workloads, but if you're processing hundreds of stores per second you'd want to move it to a background thread or queue.
- **Contradiction detection is a word-overlap heuristic** — real NLI (natural language inference) would be better but too heavy for Pi 5. If you're deploying on beefier hardware, swapping in an NLI model would probably improve this a lot.
- **Temporal reasoning needs the agent to tag episodes** — the system can't automatically figure out "this fact is now outdated" without explicit `time_context`. Auto-inferring this from content is an open problem.
- **No multi-modal support** — text only. Adding image/audio embeddings is doable but not started.
- **Embedding model picks aren't great for non-English** — the defaults are English-tuned. Multilingual support would need a different model like `paraphrase-multilingual-MiniLM-L12-v2`.
- **Tests have `@requires_embeddings` skips** — if you run without sentence-transformers installed, 7 of 40 tests skip. That's correct behavior but means the CI matrix needs both modes.
- **I'm not a seasoned ML engineer** — I'm a student. Some of the scoring weights, similarity thresholds, and consolidation parameters are empirically tuned but probably not optimal. If you have a better intuition for these, PRs welcome.

## Things I think would be valuable next

If you're looking for somewhere to contribute:

1. **Mem0/Zep comparison benchmark** — I started one but ran out of runway. Running ARN head-to-head on their published benchmarks would make this more credible.
2. **Async consolidation** — move it off the main thread so high-throughput agents don't stall.
3. **Cross-agent shared semantic layer** — read-only "organizational knowledge" that multiple agents can draw on.
4. **Better contradiction detection** — even a small NLI model would help.
5. **Multilingual embeddings** — swapping the default model for a multilingual one.
6. **ONNX quantized version** — smaller and faster for edge deployment.
7. **LangChain/CrewAI adapters** — I built the OpenClaw one because that's what I use. Other frameworks need their own thin wrappers.

## License

This is licensed under **PolyForm Small Business 1.0.0** — which basically means:

- **Free for you** if you're an individual, student, researcher, hobbyist, or working at a small company (<100 people and <$1M revenue)
- **Paid license required** if you're at a bigger company that wants to use this commercially

If you fit the free tier, just use it. Keep the LICENSE file in your fork and you're good.

If your company is over the threshold and you want to use this in a product, open an issue titled "Commercial licensing inquiry" or reach me through my GitHub profile. See [COMMERCIAL.md](./COMMERCIAL.md) for details.

I picked this instead of MIT because I'm a student and this project took a lot of work. If it's useful to you personally, I want you to have it free. If a corporation is making money off it, I'd like a share of that. The PolyForm license is written by actual lawyers and is used by other projects for this same reason.

## About

My name is Mohamed Mohamed. I'm an IT student at Columbus State Community College (currently on academic probation, not gonna hide that). I built this on a Raspberry Pi 5 I recovered from a corrupted SD card, using OpenClaw as my agent framework.

If you want to reach out, open an issue or reach me through the contacts in my GitHub profile. I'm not claiming this is production-grade enterprise software — I'm a student who built something that works and passes real adversarial tests. If you find bugs or have ideas, please say so.

Thanks for looking at this.

— Mohamed
