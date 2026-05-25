"""
ARN sqlite-vec Accelerator
===========================
Optional ANN (approximate nearest neighbor) search using sqlite-vec + apsw.

When available, replaces the O(N) numpy dot-product scan in cognitive.py with
a sqlite-vec KNN query — faster for large episode stores (1k+ episodes).

Falls back gracefully to None (caller uses numpy) if:
  - apsw is not installed
  - sqlite_vec is not installed
  - The database file can't be opened via apsw
  - Any other error during setup or search

Usage:
    from arn_v9.storage.vec_accelerator import VecAccelerator

    acc = VecAccelerator(data_dir, embedding_dim)
    results = acc.search(query_vector, top_k=20, active_ids=set([1,2,3]))
    # results = [(episode_id, similarity_score), ...] sorted desc, or None if unavailable
    acc.upsert(episode_id, vector)
    acc.delete(episode_id)

The accelerator maintains its own `episodes_vec` vec0 virtual table that
mirrors the episodic_vectors.npy data. It is advisory — if out of sync
(e.g., after a crash or migration), cognitive.py falls back to numpy.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

_APSW_AVAILABLE = False
_SQLITE_VEC_AVAILABLE = False

try:
    import apsw
    _APSW_AVAILABLE = True
except ImportError:
    pass

try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    pass


class VecAccelerator:
    """
    sqlite-vec powered ANN search accelerator for ARN episode recall.
    Maintains a shadow vec0 virtual table alongside the main SQLite DB.
    """

    def __init__(self, data_dir: Path, embedding_dim: int):
        self._data_dir = Path(data_dir)
        self._dim = embedding_dim
        self._conn = None
        self._available = False

        if not _APSW_AVAILABLE or not _SQLITE_VEC_AVAILABLE:
            logger.debug(
                "[VecAccelerator] sqlite-vec acceleration unavailable "
                f"(apsw={_APSW_AVAILABLE}, sqlite_vec={_SQLITE_VEC_AVAILABLE})"
            )
            return

        try:
            self._init()
            self._available = True
            logger.info(
                f"[VecAccelerator] sqlite-vec acceleration enabled "
                f"(dim={self._dim}, db={self._data_dir/'arn_metadata.db'})"
            )
        except Exception as e:
            logger.warning(f"[VecAccelerator] setup failed, falling back to numpy: {e}")

    def _init(self):
        """Open apsw connection and create the vec0 virtual table."""
        db_path = str(self._data_dir / "arn_metadata.db")
        self._conn = apsw.Connection(db_path)
        self._conn.enableloadextension(True)
        sqlite_vec.load(self._conn)
        self._conn.enableloadextension(False)  # disable after loading for security

        # Create the vec0 virtual table if it doesn't exist.
        # The table stores episode_id (rowid) → float32[dim] embedding.
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec "
            f"USING vec0(embedding float[{self._dim}])"
        )

    @property
    def available(self) -> bool:
        return self._available

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        active_ids: Optional[Set[int]] = None,
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Find top_k most similar episodes by cosine similarity.

        Args:
            query_vector: normalized (D,) float32 query embedding
            top_k: number of candidates to return (before filtering)
            active_ids: if provided, only return IDs in this set

        Returns:
            List of (episode_id, similarity) sorted by similarity desc, or None on error.
        """
        if not self._available:
            return None

        try:
            blob = sqlite_vec.serialize_float32(query_vector.astype(np.float32).tolist())
            # Request more candidates than top_k since active_ids may filter some out
            fetch_k = top_k * 4 if active_ids else top_k
            rows = self._conn.execute(
                "SELECT rowid, distance FROM episodes_vec "
                "WHERE embedding MATCH ? AND k=? "
                "ORDER BY distance",
                (blob, fetch_k),
            ).fetchall()

            results = []
            for row_id, distance in rows:
                if active_ids is not None and row_id not in active_ids:
                    continue
                # sqlite-vec distance is L2 by default for float32.
                # Convert to cosine similarity for normalized vectors:
                # For normalized vectors: cosine_sim = 1 - (L2_dist² / 2)
                sim = max(0.0, 1.0 - (distance ** 2) / 2.0)
                results.append((int(row_id), float(sim)))
                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.warning(f"[VecAccelerator] search failed: {e}")
            return None

    def upsert(self, episode_id: int, vector: np.ndarray) -> bool:
        """
        Insert or update an episode's vector in the vec0 table.
        Call this whenever a new episode is stored or updated.
        """
        if not self._available:
            return False
        try:
            blob = sqlite_vec.serialize_float32(vector.astype(np.float32).tolist())
            # vec0 uses INSERT OR REPLACE semantics via DELETE + INSERT
            self._conn.execute(
                "DELETE FROM episodes_vec WHERE rowid = ?", (episode_id,)
            )
            self._conn.execute(
                "INSERT INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                (episode_id, blob),
            )
            return True
        except Exception as e:
            logger.warning(f"[VecAccelerator] upsert failed for ep {episode_id}: {e}")
            return False

    def delete(self, episode_id: int) -> bool:
        """Remove an episode's vector from the vec0 table."""
        if not self._available:
            return False
        try:
            self._conn.execute(
                "DELETE FROM episodes_vec WHERE rowid = ?", (episode_id,)
            )
            return True
        except Exception as e:
            logger.warning(f"[VecAccelerator] delete failed for ep {episode_id}: {e}")
            return False

    def rebuild(self, episode_id_vector_pairs: List[Tuple[int, np.ndarray]]) -> int:
        """
        Rebuild the entire vec0 table from scratch.
        Call this after a migration or if the table gets out of sync.
        Returns number of rows inserted.
        """
        if not self._available:
            return 0
        try:
            self._conn.execute("DELETE FROM episodes_vec")
            count = 0
            for ep_id, vec in episode_id_vector_pairs:
                blob = sqlite_vec.serialize_float32(vec.astype(np.float32).tolist())
                self._conn.execute(
                    "INSERT INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                    (ep_id, blob),
                )
                count += 1
            logger.info(f"[VecAccelerator] rebuilt {count} vectors in episodes_vec")
            return count
        except Exception as e:
            logger.warning(f"[VecAccelerator] rebuild failed: {e}")
            return 0

    def count(self) -> int:
        """Return number of vectors in the vec0 table."""
        if not self._available:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM episodes_vec").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def sync_from_storage(self, storage) -> int:
        """
        Sync vec0 table from an ARNStorage instance.
        Reads all active non-invalidated episodes and their vectors.
        Returns number of vectors synced.
        """
        if not self._available:
            return 0
        try:
            conn = storage._get_conn()
            rows = conn.execute(
                "SELECT id, vec_index FROM episodes WHERE invalidated_at IS NULL"
            ).fetchall()
            pairs = []
            for row in rows:
                ep_id = row['id']
                vi = row['vec_index']
                if vi < storage._episodic_vectors.shape[0]:
                    vec = storage._episodic_vectors[vi].copy()
                    # Skip zero vectors (empty slots)
                    if np.any(vec):
                        pairs.append((ep_id, vec))
            return self.rebuild(pairs)
        except Exception as e:
            logger.warning(f"[VecAccelerator] sync_from_storage failed: {e}")
            return 0

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._available = False
