# Test Results

## Summary

```
59 passed, 3 skipped in 5.9s
```

The 3 skipped tests require the real `all-MiniLM-L6-v2` embedding model
(internet access needed to download). All other tests pass in offline/degraded
mode using the lexical hash fallback.

---

## Test Suite Breakdown

### `tests/test_openclaw_integration.py` — OpenClaw pipeline (35 tests)

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestSchemaV7` | 3 | sessions table exists, role/metadata/session_id columns, SCHEMA_VERSION=7 |
| `TestSessionManagement` | 7 | create, get, idempotent duplicate, end, recent list, count, get_episodes |
| `TestRoleAwarePerceive` | 7 | user/assistant/tool_call/tool_result roles, metadata, session_id roundtrip |
| `TestRoleAwareRecall` | 4 | basic recall, role filter SQL, session filter SQL |
| `TestPinAndForget` | 5 | pin, DB-level verification, unpin, invalidate, excluded from recall |
| `TestReflectAndReview` | 4 | reflect() stats, pending reviews, enqueue+resolve, session end count |
| `TestAgeLabel` | 5 | just now (<90s), minutes, hours, days, weeks |
| `TestThresholdValidation` | 3 | topic-appropriate recall — **skipped** (requires real embeddings) |

### `tests/test_procedural.py` — Procedural memory (24 tests)

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestComplexityScoring` | 5 | trivial=0, debug session ≥8.0, multi-tool ≥8.0, no tools=0, error correction signal |
| `TestProceduralExtraction` | 5 | below-threshold skip, episode produced, GOAL/STEPS structure, metadata, recall |
| `TestProcedureSupersedesChain` | 2 | reflect() supersedes similar procedure, restore_procedure reversal |
| `TestEffectivenessTracking` | 6 | boost/reduce/no-change, cap at 2.0, floor at 0.1, review queue flag |
| `TestDeepReflect` | 4 | stats dict, stale→importance 0.1, archive valid_until, dup merge |
| `TestRoleFilter` | 2 | procedural role stored correctly, SQL role filter |

---

## Running locally

```bash
# Full suite
python -m pytest tests/ -v

# With real embeddings (requires network access to download model)
python -m pytest tests/ -v -k "ThresholdValidation or not ThresholdValidation"

# Individual suites
python -m pytest tests/test_procedural.py -v
python -m pytest tests/test_openclaw_integration.py -v
```

---

## Degraded mode

When `sentence-transformers/all-MiniLM-L6-v2` is not cached, ARN
automatically falls back to a deterministic lexical hash encoder. All tests
except `TestThresholdValidation` pass in this mode because they test
structural correctness (schema, storage, session lifecycle, role tagging,
review queue) rather than semantic quality.

The degraded-mode encoder uses `blake2b` hashes of token n-grams to produce
384-dim vectors. Recall works for exact/near keyword matches but not
cross-lingual or paraphrase queries.
