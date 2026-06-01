"""
Tests for arn_v9.storage.vec_accelerator.VecAccelerator.

Two test classes:

1. TestVecAcceleratorFallback — always runs (no optional deps required).
   Patches the module-level availability flags to simulate missing apsw
   or sqlite_vec and verifies that every public method degrades cleanly.

2. TestL2ToCosineFormula — pure-math validation of the similarity
   conversion in VecAccelerator.search() (line ~138):

       sim = max(0.0, 1.0 - (distance ** 2) / 2.0)

   This formula converts sqlite-vec's L2 distance to cosine similarity
   for unit-normalized vectors. It must hold for random normalized inputs
   — any breakage silently corrupts every recall score.
"""

import numpy as np
import pytest

import arn_v9.storage.vec_accelerator as _mod
from arn_v9.storage.vec_accelerator import VecAccelerator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_unavailable(tmp_path):
    """Return a VecAccelerator that is guaranteed to be in unavailable mode."""
    orig_apsw = _mod._APSW_AVAILABLE
    orig_vec = _mod._SQLITE_VEC_AVAILABLE
    _mod._APSW_AVAILABLE = False
    _mod._SQLITE_VEC_AVAILABLE = False
    try:
        acc = VecAccelerator(tmp_path, embedding_dim=384)
    finally:
        _mod._APSW_AVAILABLE = orig_apsw
        _mod._SQLITE_VEC_AVAILABLE = orig_vec
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Graceful degradation
# ──────────────────────────────────────────────────────────────────────────────

class TestVecAcceleratorFallback:
    """VecAccelerator must degrade cleanly when optional deps are absent."""

    def test_available_is_false_without_deps(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        assert acc.available is False

    def test_search_returns_none_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        result = acc.search(np.zeros(384, dtype=np.float32), top_k=5)
        assert result is None

    def test_search_with_active_ids_returns_none_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        result = acc.search(
            np.zeros(384, dtype=np.float32), top_k=5, active_ids={1, 2, 3}
        )
        assert result is None

    def test_upsert_returns_false_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        assert acc.upsert(1, np.zeros(384, dtype=np.float32)) is False

    def test_delete_returns_false_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        assert acc.delete(99) is False

    def test_count_returns_zero_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        assert acc.count() == 0

    def test_rebuild_returns_zero_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        pairs = [(i, np.zeros(384, dtype=np.float32)) for i in range(5)]
        assert acc.rebuild(pairs) == 0

    def test_sync_from_storage_returns_zero_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        assert acc.sync_from_storage(object()) == 0

    def test_close_does_not_raise_when_unavailable(self, tmp_path):
        acc = _make_unavailable(tmp_path)
        acc.close()  # Must not raise


# ──────────────────────────────────────────────────────────────────────────────
# L2 → cosine similarity formula
# ──────────────────────────────────────────────────────────────────────────────

class TestL2ToCosineFormula:
    """
    Validate the conversion used inside VecAccelerator.search().

    For unit-normalized vectors, sqlite-vec returns an L2 distance and
    the code converts it via:
        sim = max(0.0, 1.0 - (distance ** 2) / 2.0)

    The identity  cosine_sim = 1 - L2² / 2  holds exactly for normalized
    vectors (derived from expanding ‖a − b‖²).
    """

    @staticmethod
    def _l2_to_cosine(l2_distance: float) -> float:
        return max(0.0, 1.0 - (l2_distance ** 2) / 2.0)

    def test_identical_vectors_yield_similarity_one(self):
        # L2 distance between identical unit vectors = 0
        assert self._l2_to_cosine(0.0) == pytest.approx(1.0)

    def test_orthogonal_unit_vectors_yield_similarity_zero(self):
        # Two orthogonal unit vectors have L2 distance = √2
        l2 = float(np.sqrt(2.0))
        assert self._l2_to_cosine(l2) == pytest.approx(0.0, abs=1e-6)

    def test_antiparallel_vectors_clipped_to_zero_not_negative(self):
        # Antiparallel unit vectors → L2 = 2 → raw formula = -1, clipped to 0
        assert self._l2_to_cosine(2.0) == 0.0

    def test_sixty_degree_angle_gives_half_similarity(self):
        # cos(60°) = 0.5, which means L2 = 1.0 for unit vectors
        assert self._l2_to_cosine(1.0) == pytest.approx(0.5)

    def test_formula_matches_numpy_dot_for_random_normalized_pairs(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.standard_normal(384).astype(np.float32)
            b = rng.standard_normal(384).astype(np.float32)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)

            l2 = float(np.linalg.norm(a - b))
            sim_formula = self._l2_to_cosine(l2)
            sim_dot = max(0.0, float(np.dot(a, b)))

            assert sim_formula == pytest.approx(sim_dot, abs=1e-5), (
                f"Formula mismatch: formula={sim_formula:.6f} dot={sim_dot:.6f} l2={l2:.6f}"
            )

    def test_output_is_always_non_negative(self):
        # No L2 distance (even > 2, which shouldn't occur for unit vectors)
        # should produce a negative similarity.
        for l2 in [0.0, 0.5, 1.0, np.sqrt(2), 1.9, 2.0, 2.5, 3.0]:
            assert self._l2_to_cosine(l2) >= 0.0

    def test_output_is_always_at_most_one(self):
        for l2 in [0.0, 0.1, 0.5, 1.0, np.sqrt(2)]:
            assert self._l2_to_cosine(l2) <= 1.0 + 1e-9
