# Research Report 2 -- Embedding Model Drift

**Agent:** Embedding Model Drift Analyst
**Area:** Embedding model stability across upgrades
**Risk Level:** MEDIUM
**Effort to Fix:** 16-24 hours

---

## Executive Summary

ARN v9 is pinned to `sentence-transformers/all-MiniLM-L6-v2` (384-dim). The code detects dimension mismatches on startup and warns, but there is NO model version tracking in the database. The schema has no field indicating which model generated each vector. If the model is ever changed, all existing vectors become semantically incompatible with new queries, silently destroying recall quality. The `content_hash` field provides a partial migration path but is not currently used for re-embedding.

---

## Drift Scenarios

| Scenario | Trigger | Impact | Detectable? |
|----------|---------|--------|-------------|
| Operator upgrades model tier | Config change | All existing vectors incompatible | Yes (dimension mismatch warning) |
| Operator switches model family | e.g., MiniLM -> E5 | Same dimension, different semantics | No (dimension matches, quality silently degrades) |
| Model weights update upstream | HuggingFace hub update | Subtle semantic shift | No |
| Hash fallback activates | Model load failure | Random recall quality | Yes (degraded flag) |

---

## Detailed Findings

### 1. No Model Version in Schema (persistence.py:151-180)

The episodes table has 18 columns including content_hash, source, memory_type -- but NO column tracking which embedding model generated the vector. The vec_index points to a vector slot, but there is no metadata about the model that produced it.

### 2. Dimension Mismatch Warning Only (embeddings.py:127-148)

```python
self._config = MODEL_CONFIGS[self._tier]
self.embedding_dim = self._config['dim']
```

If the tier changes, the new model loads with a potentially different dimension. The code does NOT compare against existing vectors. It simply loads the new model and starts producing vectors that may be incompatible.

The `_load_model()` method has no check for "existing vectors were produced by a different model."

### 3. Hash Fallback is Dangerous (embeddings.py:393-425)

```python
def _hash_encode(self, text: str) -> np.ndarray:
    """Deterministic lexical hash encoding used when the transformer model cannot load."""
```

If the model fails to load (missing dependencies, network issues, corrupted cache), the system falls back to `_hash_encode()`. This produces vectors based on token hashing, NOT semantic meaning. The system continues to operate but recall becomes essentially random for semantic queries.

The degraded flag is set (`self._degraded_warned = False` is checked), but there is no alerting mechanism.

### 4. SimilarityCalibrator is Not Model-Aware (embeddings.py:471-520)

```python
class SimilarityCalibrator:
    def __init__(self):
        self._observations = []
        self._mean = 0.5
        self._m2 = 0.0
        self._count = 0
```

The calibrator learns thresholds from observed similarity scores. If the model changes, the score distribution changes, but the calibrator does not reset. Thresholds learned on MiniLM may be completely wrong for E5.

---

## Migration Strategy Design

### Approach: Lazy Re-embedding with content_hash

The `content_hash` column (SHA256 of normalized content, 16 chars) can be used as a stable identifier. A migration would:

1. Add `model_version` column to episodes table (schema v4)
2. On startup, compare current model with episodes lacking `model_version`
3. For episodes with missing or mismatched `model_version`:
   - Re-embed using `content` field
   - Update vector in memmap
   - Update `model_version` and `content_hash`
4. Run in background during idle periods

### Code Sketch
```python
def migrate_model_version(self, target_model: str):
    """Re-embed episodes that were encoded with a different model."""
    conn = self._get_conn()
    rows = conn.execute(
        "SELECT id, content, vec_index FROM episodes WHERE model_version != ? OR model_version IS NULL",
        (target_model,)
    ).fetchall()
    
    for row in rows:
        new_vector = embedder.encode(row['content'])
        self._episodic_vectors[row['vec_index']] = new_vector
        conn.execute(
            "UPDATE episodes SET model_version = ? WHERE id = ?",
            (target_model, row['id'])
        )
    conn.commit()
```

---

## Recommended Fixes

### Fix 1: Add model_version Column (High Priority)
Add `model_version TEXT` to episodes table. Store the model name/tier on every insert.

### Fix 2: Model Change Detection (High Priority)
On `StorageEngine` initialization, check if existing episodes have a different `model_version`. If so, log a warning and optionally trigger re-embedding.

### Fix 3: Calibrator Reset on Model Change (Medium Priority)
Reset `SimilarityCalibrator` when the model tier changes. Store calibration state per model.

### Fix 4: Hash Fallback Alerting (Medium Priority)
When `_hash_encode()` is used, raise a clear error or warning that is visible to the operator, not just a boolean flag.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Silent model mismatch | High | Medium | Fix 1 + Fix 2 |
| Degraded recall after upgrade | High | Low | Fix 1 + migration tool |
| Hash fallback goes unnoticed | Medium | Low | Fix 4 |
| Wrong calibration thresholds | Medium | Low | Fix 3 |
