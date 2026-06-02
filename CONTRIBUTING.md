# Contributing to ARN

## Setup

```bash
git clone https://github.com/tuuhe99-del/ARN-Adaptive-Reasoning-Network.git
cd ARN-Adaptive-Reasoning-Network
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Running tests

```bash
python -m pytest tests/ -v                              # all integration tests
python -m pytest tests/test_procedural.py -v            # procedural memory
python -m pytest tests/test_openclaw_integration.py -v  # OpenClaw pipeline
python -m pytest arn_v9/tests/test_all.py -v            # unit tests
```

Tests run in degraded mode (no embedding model) in offline environments.
Tests requiring real embeddings are automatically skipped via `@pytest.mark.skipif`.

## Architecture

See `docs/build-session-report.md` for a full explanation of every system and
the reasoning behind each design decision.

Core pipeline:
```
perceive(text)
  → encode (all-MiniLM-L6-v2, 384-dim)
  → store (episodes + vec0 + FTS5 + entities)
  → working memory update

recall(query)
  → encode query
  → KNN (sqlite-vec) + FTS5 (BM25) + entity matching
  → Reciprocal Rank Fusion
  → composite score (recency + importance + frequency + pin boost)
  → MMR reranking
  → score-gap cutoff

reflect(session_id)
  → contradiction scan
  → importance recalibration
  → procedure extraction (complexity >= 8.0)
  → consolidation
```

## What needs help

1. **LoCoMo / LongMemEval benchmarks** — run ARN against published long-term
   memory benchmarks. This is the most credible way to demonstrate recall quality.

2. **NLI-based contradiction detection** — replace the cosine+overlap heuristic
   with a small cross-encoder. The current approach generates false positives on
   paraphrases.

3. **Multilingual support** — swap `all-MiniLM-L6-v2` for
   `paraphrase-multilingual-MiniLM-L12-v2`, add non-English recall tests.

4. **LangChain / CrewAI adapters** — thin wrappers mapping ARN's
   `perceive()`/`recall()` interface to other agent framework abstractions.

5. **Async consolidation** — runs synchronously today. A background priority
   queue would help high-throughput setups without blocking agent responses.

## Code style

- Type hints on all public signatures
- One-line docstrings where the name isn't self-explanatory
- Comments explain WHY, not WHAT
- Tests must pass in degraded mode (offline, no model download)

## Before opening a PR

- All tests pass: `python -m pytest tests/ -v`
- No new dependencies without discussion (each one multiplies the Pi 5 footprint)
- Schema changes need a migration in `_migrate_schema()` with idempotent
  `ALTER TABLE ... ADD COLUMN` guards

Open an issue first for anything touching the schema, embedding pipeline, or
retrieval ranking — those have non-obvious interactions.
