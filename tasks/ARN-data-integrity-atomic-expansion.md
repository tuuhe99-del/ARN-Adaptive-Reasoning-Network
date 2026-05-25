# ARN Data Integrity: Atomic Vector Expansion

## Goal

Make ARN's vector-store expansion crash-safe. Expanding `.npy` files must not
truncate or corrupt the active vector store if the process crashes mid-write.

## Why This Matters

ARN's north star is a trustworthy local-first portable memory layer for AI
agents. A memory system cannot be trusted if vector files can be corrupted during
normal growth.

## Evidence

- `research/research_01_data_integrity.md`
- `research/research_07_memmap_concurrency.md`
- Claude handoff `/Users/hustle/.arn_data/collab/handoffs/2026-05-18_102500-claude.md`
- Kimi handoff `/Users/hustle/.arn_data/collab/handoffs/2026-05-18_102807-kimi.md`

## Initial Scope

Implement the smallest correct fix for:

- `_expand_episodic_vectors`
- `_expand_semantic_vectors`

Expected direction:

- write the expanded array to a temp file in the same directory
- flush/sync where practical
- atomically replace the target `.npy` file with `os.replace`
- reopen the memmap after replacement
- preserve existing vectors exactly

## Out Of Scope

- dashboard work
- portable memory export/import
- re-embedding migration CLI
- broad storage refactors
- unrelated API security work

## Success Criteria

- Existing vectors survive expansion.
- Expansion uses temp-file plus atomic rename/replace.
- A failed temp write does not replace the active vector file.
- Focused tests cover episodic and semantic expansion.
- Existing collaboration tests still pass.

## Suggested Verification

```bash
python3 -m py_compile arn_v9/storage/persistence.py arn_v9/tests/test_all.py
python3 -m pytest arn_v9/tests/test_collab.py arn_v9/tests/test_collab_runner.py
python3 arn_v9/tests/test_all.py
```

If the full `test_all.py` is too slow or environment-limited, run the focused
persistence section and record the limitation in the handoff.
