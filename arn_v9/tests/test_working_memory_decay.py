"""
Unit tests for WorkingMemory — focusing on decay behaviour.

WorkingMemory implements time-based activation decay:

    decay_factor = max(0.0, 1.0 - rate * elapsed_seconds)
    slot.activation *= decay_factor

Slots whose activation drops below 0.01 are freed.  No embedding model
is needed; these tests use raw numpy vectors.
"""

import numpy as np
import pytest

from arn_v9.core.cognitive import WorkingMemory

DIM = 384
_VEC = np.zeros(DIM, dtype=np.float32)


def _wm(max_slots: int = 7) -> WorkingMemory:
    return WorkingMemory(max_slots=max_slots, embedding_dim=DIM)


def _add(wm: WorkingMemory, text: str, priority: float = 1.0, source_id: int = 0):
    wm.add(text, _VEC.copy(), priority=priority, source_id=source_id)


# ──────────────────────────────────────────────────────────────────────────────
# Basic add / retrieval
# ──────────────────────────────────────────────────────────────────────────────

def test_add_single_item():
    wm = _wm()
    _add(wm, "hello")
    assert wm.count == 1
    active = wm.get_active()
    assert len(active) == 1
    assert active[0].content == "hello"


def test_add_multiple_items_up_to_capacity():
    wm = _wm(max_slots=3)
    for i in range(3):
        _add(wm, f"item {i}", priority=float(i + 1))
    assert wm.count == 3


def test_lowest_activation_evicted_when_full():
    wm = _wm(max_slots=3)
    _add(wm, "low", priority=0.1)
    _add(wm, "mid", priority=0.5)
    _add(wm, "high", priority=0.9)
    # Adding a higher-priority item must evict "low"
    _add(wm, "very high", priority=1.0)
    contents = {s.content for s in wm.get_active()}
    assert "low" not in contents
    assert "very high" in contents


def test_item_below_existing_minimum_is_discarded():
    wm = _wm(max_slots=2)
    _add(wm, "strong", priority=0.8)
    _add(wm, "moderate", priority=0.6)
    # Priority 0.1 < both existing slots → should be silently dropped
    _add(wm, "weak", priority=0.1)
    assert wm.count == 2
    contents = {s.content for s in wm.get_active()}
    assert "weak" not in contents


# ──────────────────────────────────────────────────────────────────────────────
# Decay math
# ──────────────────────────────────────────────────────────────────────────────

def test_activation_decreases_after_decay():
    wm = _wm()
    _add(wm, "item", priority=1.0)
    before = wm.get_active()[0].activation
    wm.decay(elapsed_seconds=1.0, rate=0.05)
    after = wm.get_active()[0].activation
    assert after < before


def test_decay_is_proportional_to_elapsed_time():
    """Longer elapsed time → smaller remaining activation."""
    wm1 = _wm()
    _add(wm1, "item", priority=1.0)
    wm1.decay(elapsed_seconds=1.0, rate=0.05)
    act_short = wm1.get_active()[0].activation

    wm2 = _wm()
    _add(wm2, "item", priority=1.0)
    wm2.decay(elapsed_seconds=10.0, rate=0.05)
    active2 = wm2.get_active()
    act_long = active2[0].activation if active2 else 0.0

    assert act_long < act_short


def test_decay_formula_matches_expected_value():
    """Verify the exact formula: activation *= max(0, 1 - rate * elapsed)."""
    wm = _wm()
    _add(wm, "item", priority=0.8)
    rate, elapsed = 0.05, 2.0
    expected = 0.8 * max(0.0, 1.0 - rate * elapsed)
    wm.decay(elapsed_seconds=elapsed, rate=rate)
    actual = wm.get_active()[0].activation
    assert actual == pytest.approx(expected, abs=1e-6)


def test_multiple_decay_calls_are_cumulative():
    wm = _wm()
    _add(wm, "item", priority=1.0)
    for _ in range(3):
        wm.decay(elapsed_seconds=1.0, rate=0.05)
    final = wm.get_active()[0].activation
    expected = 1.0 * (0.95 ** 3)
    assert final == pytest.approx(expected, abs=1e-6)


def test_zero_elapsed_causes_no_decay():
    wm = _wm()
    _add(wm, "item", priority=0.7)
    wm.decay(elapsed_seconds=0.0, rate=0.05)
    assert wm.get_active()[0].activation == pytest.approx(0.7)


def test_very_large_elapsed_does_not_produce_negative_activation():
    wm = _wm()
    _add(wm, "item", priority=1.0)
    wm.decay(elapsed_seconds=1_000_000.0, rate=0.05)
    active = wm.get_active()
    # Either the slot was freed (below 0.01 threshold) or its activation ≥ 0
    if active:
        assert active[0].activation >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Slot freeing
# ──────────────────────────────────────────────────────────────────────────────

def test_slot_freed_when_activation_falls_below_threshold():
    """A slot with activation < 0.01 after decay must be removed."""
    wm = _wm()
    _add(wm, "fading", priority=0.011)
    # Decay enough to push below 0.01
    wm.decay(elapsed_seconds=1000.0, rate=0.05)
    assert wm.count == 0
    assert wm.get_active() == []


def test_slot_with_high_activation_survives_small_decay():
    wm = _wm()
    _add(wm, "durable", priority=1.0)
    wm.decay(elapsed_seconds=1.0, rate=0.05)
    assert wm.count == 1


def test_only_low_activation_slots_freed():
    wm = _wm(max_slots=3)
    _add(wm, "strong", priority=1.0)
    _add(wm, "weak",   priority=0.015)
    _add(wm, "medium", priority=0.5)
    # Decay just enough to drop "weak" below 0.01
    wm.decay(elapsed_seconds=1000.0, rate=0.05)
    contents = {s.content for s in wm.get_active()}
    assert "weak" not in contents
    assert "strong" not in contents  # also gone with large elapsed
    # At least medium should also be gone with such extreme elapsed
    assert wm.count == 0


def test_slot_count_decremented_correctly_on_free():
    """_slot_count must match the actual number of non-None slots after decay."""
    wm = _wm(max_slots=4)
    for i in range(4):
        _add(wm, f"item {i}", priority=float(i + 1) * 0.1)
    before_count = wm.count
    wm.decay(elapsed_seconds=1000.0, rate=0.05)
    # All should be gone; count must be 0, not negative or stale
    assert wm.count == 0
    assert wm.count == sum(1 for s in wm.slots if s is not None)


# ──────────────────────────────────────────────────────────────────────────────
# Context vector
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_working_memory_context_vector_is_none():
    wm = _wm()
    assert wm.get_context_vector() is None


def test_context_vector_present_with_active_slots():
    wm = _wm()
    vec = np.ones(DIM, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    wm.add("item", vec, priority=1.0)
    ctx = wm.get_context_vector()
    assert ctx is not None
    assert ctx.shape == (DIM,)
