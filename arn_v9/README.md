# ARN v9: Adaptive Reasoning Network

**Brain-inspired cognitive architecture for AI agents on Raspberry Pi 5.**

## What ARN v9 Does

ARN gives AI agents persistent, intelligent memory that works like a brain:

- **Learns from every interaction** — stores experiences with semantic understanding
- **Consolidates knowledge** — automatically extracts patterns from episodes (like sleep)
- **Detects contradictions** — flags when new info conflicts with existing knowledge
- **Recalls by meaning** — finds relevant memories by semantic similarity, not keyword match
- **Runs locally for $0/month** — no cloud APIs, fits under 50MB on a Pi 5

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ARN v9 Core                       │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Embedding   │  │   Domain     │  │  Working   │ │
│  │  Engine      │  │   Columns    │  │  Memory    │ │
│  │ (MiniLM-L6)  │  │  (8 domains) │  │  (7 slots) │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                │        │
│  ┌──────▼─────────────────▼────────────────▼──────┐ │
│  │              Perceive / Recall                  │ │
│  └──────┬─────────────────────────────────┬───────┘ │
│         │                                 │         │
│  ┌──────▼───────┐              ┌──────────▼───────┐ │
│  │   Episodic   │  consolidate │    Semantic      │ │
│  │   Memory     │ ──────────►  │    Memory        │ │
│  │  (fast learn) │  clustering  │  (slow learn)    │ │
│  └──────┬───────┘              └──────────┬───────┘ │
│         │                                 │         │
│  ┌──────▼─────────────────────────────────▼───────┐ │
│  │           Persistence Layer                     │ │
│  │     SQLite (metadata) + memmap (vectors)        │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Neuroscience Foundations

| Brain System | ARN Component | Function |
|---|---|---|
| Hippocampus | Episodic Memory | Fast, one-shot learning of specific events |
| Neocortex | Semantic Memory | Slow, generalized knowledge from patterns |
| Sleep Replay | Consolidation Engine | Cluster episodes → extract semantic knowledge |
| Prefrontal Cortex | Working Memory | Active context (7 slots, decay, rehearsal) |
| Cortical Columns | Domain Columns | Parallel domain-specific processing + voting |
| Prediction Error | Calibrated Surprise | Welford's running stats per domain |

## Quick Start

```python
from arn_v9 import ARNv9

# Initialize (creates persistent storage)
arn = ARNv9(data_dir="./my_agent_memory")

# Store experiences
arn.perceive("User prefers Python for scripting", importance=0.8)
arn.perceive("Raspberry Pi 5 is the deployment target", importance=0.7)

# Recall by meaning
results = arn.recall("What programming language?", top_k=3)
for r in results:
    print(f"[{r['score']:.3f}] {r['content']}")

# Run consolidation (during idle time)
stats = arn.consolidate()

# Clean shutdown
arn.close()
```

## OpenClaw Plugin Usage

```python
from arn_v9.plugin import ARNPlugin

with ARNPlugin(agent_id="my_agent", data_root="./memory") as plugin:
    # Store
    plugin.store("User likes dark mode", importance=0.5, tags=["preference"])
    
    # Recall
    results = plugin.recall("user interface preferences")
    
    # Get formatted context for LLM prompt injection
    context = plugin.get_context_window(
        query="help the user with settings",
        max_tokens=1000
    )
    
    # Maintenance (call during idle)
    plugin.maintain()
```

## Performance (Container Benchmarks)

| Metric | Value |
|---|---|
| Perceive (avg) | ~34 ms |
| Perceive (p95) | ~79 ms |
| Recall (avg) | ~41 ms |
| Recall (p95) | ~82 ms |
| Episode write | ~0.2 ms (excl. embedding) |
| Batch encode (100) | ~300 ms |
| Storage (45 episodes) | 9.0 MB |
| Recall accuracy | 100% (10/10 factual queries) |

## Deployment on Raspberry Pi 5

```bash
# Install dependencies
pip install sentence-transformers numpy

# Clone/copy ARN v9
cp -r arn_v9/ /home/mokali/arn_v9/

# Test
python3 -c "from arn_v9 import ARNv9; print('OK')"

# Run tests
python3 arn_v9/tests/test_all.py
```

### SD Card Wear Considerations

- SQLite uses WAL mode (sequential writes, no random I/O)
- Vector writes are memory-mapped (OS batches flushes)
- Consolidation batches writes (not per-episode)
- Consider mounting `/tmp` as tmpfs for working memory

## File Structure

```
arn_v9/
├── __init__.py              # Package entry point
├── plugin.py                # OpenClaw plugin interface
├── requirements.txt         # Dependencies
├── core/
│   ├── __init__.py
│   ├── embeddings.py        # Semantic embedding engine (MiniLM-L6-v2)
│   └── cognitive.py         # Main cognitive architecture
├── storage/
│   ├── __init__.py
│   └── persistence.py       # SQLite + memmap persistence
├── tests/
│   └── test_all.py          # 44-test comprehensive suite
└── benchmarks/
    └── simulate_agent.py    # Full agent simulation + stats
```

## Known Limitations & Future Work

1. **Embedding model size**: MiniLM-L6-v2 uses ~90MB RAM at runtime. For tighter
   budgets, explore ONNX quantized models or distilled alternatives.

2. **Contradiction detection**: Uses a word-overlap heuristic. Full NLI
   (natural language inference) would be more accurate but too heavy for Pi 5.

3. **Consolidation is synchronous**: Currently blocks the main thread. For
   production, run in a background thread or during idle periods.

4. **No inter-agent memory sharing**: Each agent has isolated memory. Cross-agent
   knowledge transfer would need a shared semantic layer.

5. **Domain column prototypes are seed-phrase based**: They adapt over time via
   slow learning, but initial domain boundaries depend on seed quality.

## License

MIT — use freely for your projects.
