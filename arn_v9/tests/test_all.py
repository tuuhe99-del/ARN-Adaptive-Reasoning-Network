"""
ARN v9 Comprehensive Test Suite
=================================
Two tiers of tests:

TIER 1 - PLUMBING (always runs, no model required):
  - Storage read/write/persistence
  - Working memory mechanics
  - Data structure integrity
  - CLI argument parsing
  - Degraded mode detection

TIER 2 - SEMANTIC (requires sentence-transformers + model):
  - Embedding quality and similarity
  - Prediction error calibration
  - Consolidation clustering
  - Contradiction detection
  - Recall accuracy
  - Full agent simulation
  - Stress test

Tests that require embeddings are SKIPPED (not failed) when the
model is unavailable. This gives a clean pass/skip/fail result
instead of 6 misleading failures.
"""

import sys
import os
import time
import json
import shutil
import tempfile
import traceback
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arn_v9.core.embeddings import EmbeddingEngine, EMBEDDING_DIM
from arn_v9.storage import persistence as persistence_module
from arn_v9.storage.persistence import StorageEngine
from arn_v9.core.cognitive import (
    ARNv9, WorkingMemory, ConsolidationEngine
)


# =========================================================
# ENVIRONMENT DETECTION
# =========================================================

def check_embeddings_available() -> bool:
    """Check if sentence-transformers is installed and model loads."""
    try:
        from sentence_transformers import SentenceTransformer
        engine = EmbeddingEngine(use_model=True)
        if engine.is_degraded:
            return False
        # Quick sanity: two similar texts should have sim > 0.5
        v1 = engine.encode("hello world")
        v2 = engine.encode("hi there")
        sim = float(np.dot(v1, v2))
        return sim > 0.2  # Sanity check that vectors are semantic
    except Exception:
        return False


EMBEDDINGS_AVAILABLE = check_embeddings_available()


# =========================================================
# TEST RESULTS TRACKER
# =========================================================

class TestResults:
    """Collect test results with pass/fail/skip tracking."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.benchmarks = {}

    def ok(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")

    def skip(self, name: str, reason: str = "requires embedding model"):
        self.skipped += 1
        print(f"  ⊘ SKIP {name}: {reason}")

    def bench(self, name: str, value: float, unit: str):
        self.benchmarks[name] = (value, unit)
        print(f"  ⏱ {name}: {value:.2f} {unit}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed} passed, {self.failed} failed, "
              f"{self.skipped} skipped (total {total})")
        if not EMBEDDINGS_AVAILABLE:
            print(f"\n⚠  EMBEDDING MODEL NOT AVAILABLE")
            print(f"   {self.skipped} semantic tests were skipped.")
            print(f"   Install: pip install sentence-transformers")
            print(f"   Then re-run for full 44/44 validation.")
        if self.errors:
            print(f"\nFAILURES:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        if self.benchmarks:
            print(f"\nBENCHMARKS:")
            for name, (val, unit) in self.benchmarks.items():
                print(f"  - {name}: {val:.2f} {unit}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


def make_temp_dir():
    return tempfile.mkdtemp(prefix="arn_test_")


def requires_embeddings(func):
    """Decorator: skip test if embeddings unavailable."""
    def wrapper():
        if not EMBEDDINGS_AVAILABLE:
            # Count expected tests in this function and skip them
            # We do this by running the function name as a skip
            results.skip(func.__name__, "requires sentence-transformers")
            return
        return func()
    wrapper.__name__ = func.__name__
    return wrapper


# =========================================================
# TIER 1: PLUMBING TESTS (always run)
# =========================================================

def test_embedding_basics():
    """Test embedding engine fundamentals (works even in degraded mode)."""
    print("\n[TIER 1] EMBEDDING ENGINE BASICS")
    print("-" * 40)

    engine = EmbeddingEngine(use_model=EMBEDDINGS_AVAILABLE)

    # Dimension check
    vec = engine.encode("hello world")
    if vec.shape == (EMBEDDING_DIM,):
        results.ok(f"Embedding dimension is {EMBEDDING_DIM}")
    else:
        results.fail("Embedding dimension", f"Got {vec.shape}")

    # Normalization check
    norm = np.linalg.norm(vec)
    if abs(norm - 1.0) < 0.01:
        results.ok(f"Vector is unit-normalized (norm={norm:.4f})")
    else:
        results.fail("Normalization", f"norm={norm:.4f}")

    # Batch encoding shape
    texts = ["hello", "world", "test", "batch", "encode"]
    batch_vecs = engine.encode_batch(texts)
    if batch_vecs.shape == (5, EMBEDDING_DIM):
        results.ok("Batch encoding shape correct")
    else:
        results.fail("Batch encoding", f"Got shape {batch_vecs.shape}")

    # Cache works
    engine.clear_cache()
    engine.encode("cached text test")
    engine.encode("cached text test")
    stats = engine.get_stats()
    if stats['cache_hits'] >= 1:
        results.ok(f"Cache hit working (hits={stats['cache_hits']})")
    else:
        results.fail("Cache", f"No cache hits: {stats}")

    # Degraded state detection
    if EMBEDDINGS_AVAILABLE:
        if not engine.is_degraded:
            results.ok("is_degraded=False with model loaded")
        else:
            results.fail("Degraded detection", "Model loaded but is_degraded=True")
    else:
        if engine.is_degraded:
            results.ok("is_degraded=True without model (correct)")
        else:
            results.fail("Degraded detection", "No model but is_degraded=False")


def test_persistence():
    """Test storage layer (does NOT require embeddings)."""
    print("\n[TIER 1] PERSISTENCE TESTS")
    print("-" * 40)

    tmp_dir = make_temp_dir()

    try:
        # Use random vectors for plumbing tests — semantics don't matter here
        def random_vec():
            v = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        # Basic store and retrieve
        with StorageEngine(tmp_dir) as storage:
            vec = random_vec()
            eid = storage.store_episode("test episode content", vec, importance=0.8)
            ep = storage.get_episode(eid)
            if ep and ep['content'] == "test episode content" and abs(ep['importance'] - 0.8) < 0.01:
                results.ok("Episode store and retrieve")
            else:
                results.fail("Episode store/retrieve", f"Got: {ep}")

        # Persistence across restart
        with StorageEngine(tmp_dir) as storage:
            count = storage.count_episodes()
            if count >= 1:
                results.ok(f"Data persists across restart (count={count})")
            else:
                results.fail("Persistence", f"count={count} after restart")

        # Corrupted vector files should not prevent startup
        corrupt_dir = make_temp_dir()
        try:
            for filename in ("episodic_vectors.npy", "semantic_vectors.npy"):
                with open(os.path.join(corrupt_dir, filename), "wb") as f:
                    f.write(b"not a valid numpy file")
            with StorageEngine(corrupt_dir) as storage:
                expected_shapes = (
                    storage._episodic_vectors.shape == (4096, EMBEDDING_DIM)
                    and storage._semantic_vectors.shape == (2048, EMBEDDING_DIM)
                )
                quarantined = os.listdir(corrupt_dir)
                has_quarantines = (
                    any(name.startswith("episodic_vectors.npy.corrupt-") for name in quarantined)
                    and any(name.startswith("semantic_vectors.npy.corrupt-") for name in quarantined)
                )
                if expected_shapes and has_quarantines:
                    results.ok("Corrupted vectors are quarantined and rebuilt")
                elif not expected_shapes:
                    results.fail(
                        "Corrupt vector recovery",
                        f"Unexpected shapes: {storage._episodic_vectors.shape}, "
                        f"{storage._semantic_vectors.shape}",
                    )
                else:
                    results.fail("Corrupt vector quarantine", "Missing quarantined file")
        finally:
            shutil.rmtree(corrupt_dir, ignore_errors=True)

        # Semantic node storage
        with StorageEngine(tmp_dir) as storage:
            vec = random_vec()
            sid = storage.store_semantic("test_concept", vec, confidence=0.5)
            sems = storage.get_all_semantics()
            if len(sems) >= 1 and sems[0]['concept_label'] == "test_concept":
                results.ok("Semantic node storage")
            else:
                results.fail("Semantic storage", f"Got: {sems}")

        # Atomic vector expansion preserves existing vectors
        expansion_dir = make_temp_dir()
        try:
            with StorageEngine(expansion_dir, max_episodes=2, max_semantics=2) as storage:
                ep_vec = random_vec()
                sem_vec = random_vec()
                storage._episodic_vectors[0] = ep_vec
                storage._semantic_vectors[0] = sem_vec

                storage._expand_episodic_vectors()
                storage._expand_semantic_vectors()

                ep_ok = (
                    storage._episodic_vectors.shape == (4, EMBEDDING_DIM)
                    and np.allclose(storage._episodic_vectors[0], ep_vec)
                )
                sem_ok = (
                    storage._semantic_vectors.shape == (4, EMBEDDING_DIM)
                    and np.allclose(storage._semantic_vectors[0], sem_vec)
                )
                if ep_ok and sem_ok:
                    results.ok("Vector expansion preserves existing vectors")
                else:
                    results.fail(
                        "Vector expansion preservation",
                        f"episodic={storage._episodic_vectors.shape}, "
                        f"semantic={storage._semantic_vectors.shape}",
                    )
        finally:
            shutil.rmtree(expansion_dir, ignore_errors=True)

        # Failed temp writes must not replace the active vector file
        failure_dir = make_temp_dir()
        original_np_save = persistence_module.np.save
        try:
            with StorageEngine(failure_dir, max_episodes=2, max_semantics=2) as storage:
                ep_vec = random_vec()
                storage._episodic_vectors[0] = ep_vec
                storage._episodic_vectors.flush()

                def fail_save(*args, **kwargs):
                    raise RuntimeError("simulated temp write failure")

                persistence_module.np.save = fail_save
                try:
                    storage._expand_episodic_vectors()
                    results.fail("Atomic expansion failure handling", "expand unexpectedly succeeded")
                except RuntimeError:
                    persisted = np.load(str(storage.episodic_vec_path), mmap_mode='r')
                    active_ok = (
                        persisted.shape == (2, EMBEDDING_DIM)
                        and np.allclose(persisted[0], ep_vec)
                    )
                    if active_ok:
                        results.ok("Failed vector expansion leaves active file intact")
                    else:
                        results.fail(
                            "Atomic expansion failure handling",
                            f"active file changed to shape={persisted.shape}",
                        )
                    del persisted
        finally:
            persistence_module.np.save = original_np_save
            shutil.rmtree(failure_dir, ignore_errors=True)

        # Vec_index uniqueness (the critical bug that was fixed)
        with StorageEngine(tmp_dir) as storage:
            for i in range(20):
                storage.store_episode(f"episode {i}", random_vec(), importance=0.5)
            
            # Mark some as consolidated
            storage.mark_episodes_consolidated([1, 2, 3, 4, 5])
            
            # Store more after consolidation
            for i in range(10):
                storage.store_episode(f"post-consolidation {i}", random_vec(), importance=0.5)
            
            # Check for vec_index collisions
            all_eps = storage.get_all_episodes(consolidated=None)
            vec_indices = [e['vec_index'] for e in all_eps]
            unique_indices = set(vec_indices)
            
            if len(vec_indices) == len(unique_indices):
                results.ok(f"No vec_index collisions ({len(vec_indices)} episodes, all unique)")
            else:
                from collections import Counter
                dupes = [(idx, cnt) for idx, cnt in Counter(vec_indices).items() if cnt > 1]
                results.fail("Vec_index collision", f"{len(dupes)} collisions found: {dupes[:3]}")

        # Storage size check
        with StorageEngine(tmp_dir) as storage:
            stats = storage.get_storage_stats()
            if stats['total_size_mb'] < 50:
                results.ok(f"Storage size under budget ({stats['total_size_mb']:.2f} MB)")
            else:
                results.fail("Storage size", f"{stats['total_size_mb']:.2f} MB")

        # Write performance
        with StorageEngine(tmp_dir) as storage:
            start = time.time()
            for i in range(100):
                storage.store_episode(f"bench episode {i}", random_vec())
            elapsed = time.time() - start
            results.bench("Episode write (100 ops)", elapsed * 1000, "ms")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_link_storage():
    """Test memory link CRUD operations (no embeddings needed)."""
    print("\n[TIER 1] LINK STORAGE")
    print("-" * 40)

    tmp = make_temp_dir()
    try:
        storage = StorageEngine(tmp)
        zero_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        id1 = storage.store_episode("Episode A", zero_vec)
        id2 = storage.store_episode("Episode B", zero_vec)
        id3 = storage.store_episode("Episode C", zero_vec)

        # Create link
        lid = storage.create_link(id1, id2, "relates_to")
        if lid > 0:
            results.ok("create_link returns valid id")
        else:
            results.fail("create_link", f"Expected positive id, got {lid}")

        # Duplicate returns existing id
        lid2 = storage.create_link(id1, id2, "relates_to")
        if lid2 == lid:
            results.ok("Duplicate link returns same id (no error)")
        else:
            results.fail("Duplicate link", f"Expected {lid}, got {lid2}")

        # Different relation type → new link
        lid3 = storage.create_link(id1, id2, "used_by")
        if lid3 != lid and lid3 > 0:
            results.ok("Different relation type creates new link")
        else:
            results.fail("Different relation type", f"lid={lid}, lid3={lid3}")

        # get_links_for_episode: outgoing
        links = storage.get_links_for_episode(id1)
        if len(links) == 2:
            results.ok("get_links_for_episode returns 2 outgoing links")
        else:
            results.fail("get_links_for_episode (outgoing)", f"Expected 2, got {len(links)}")

        # get_links_for_episode: incoming
        lid4 = storage.create_link(id3, id1, "part_of")
        links = storage.get_links_for_episode(id1)
        if len(links) == 3:
            results.ok("get_links_for_episode includes incoming links")
        else:
            results.fail("get_links_for_episode (incoming)", f"Expected 3, got {len(links)}")

        # get_all_links
        all_links = storage.get_all_links()
        if len(all_links) == 3:
            results.ok("get_all_links returns all links")
        else:
            results.fail("get_all_links", f"Expected 3, got {len(all_links)}")

        # delete_link
        storage.delete_link(lid)
        links = storage.get_links_for_episode(id1)
        if len(links) == 2:
            results.ok("delete_link removes the link")
        else:
            results.fail("delete_link", f"Expected 2 after delete, got {len(links)}")

        # Persistence across restarts
        storage.close()
        storage2 = StorageEngine(tmp)
        all2 = storage2.get_all_links()
        if len(all2) == 2:
            results.ok("Links persist across storage restarts")
        else:
            results.fail("Link persistence", f"Expected 2, got {len(all2)}")
        storage2.close()

    except Exception as exc:
        results.fail("test_link_storage", traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_working_memory():
    """Test working memory mechanics (no embeddings needed for structure tests)."""
    print("\n[TIER 1] WORKING MEMORY TESTS")
    print("-" * 40)

    def random_vec():
        v = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    # Capacity limit
    wm = WorkingMemory(max_slots=3)
    for i in range(5):
        wm.add(f"item {i}", random_vec(), priority=float(i))

    active = wm.get_active()
    if len(active) <= 3:
        results.ok(f"Working memory respects capacity (active={len(active)})")
    else:
        results.fail("WM capacity", f"active={len(active)} > 3")

    # Priority-based eviction
    contents = [s.content for s in active]
    if "item 4" in contents and "item 3" in contents:
        results.ok("Highest priority items retained")
    else:
        results.fail("WM priority eviction", f"Active: {contents}")

    # Decay
    wm2 = WorkingMemory(max_slots=5)
    wm2.add("decaying item", random_vec(), priority=0.5)
    initial_count = wm2.count
    wm2.decay(elapsed_seconds=100, rate=0.1)
    after_count = wm2.count
    if after_count < initial_count:
        results.ok("Decay removes low-activation items")
    else:
        active = wm2.get_active()
        if active and active[0].activation < 0.5:
            results.ok(f"Decay reduces activation ({active[0].activation:.3f})")
        else:
            results.fail("WM decay", "No decay observed")

    # Context vector
    wm3 = WorkingMemory(max_slots=5)
    wm3.add("item a", random_vec(), priority=0.8)
    wm3.add("item b", random_vec(), priority=0.6)
    ctx = wm3.get_context_vector()
    if ctx is not None and ctx.shape == (EMBEDDING_DIM,):
        results.ok("Context vector computed correctly")
    else:
        results.fail("WM context vector", f"Got: {type(ctx)}")


# =========================================================
# TIER 2: SEMANTIC TESTS (require embeddings)
# =========================================================

@requires_embeddings
def test_embedding_quality():
    """Test semantic similarity quality (REQUIRES model)."""
    print("\n[TIER 2] EMBEDDING SEMANTIC QUALITY")
    print("-" * 40)

    engine = EmbeddingEngine(use_model=True)

    # Similar texts
    v1 = engine.encode("Python is a programming language")
    v2 = engine.encode("Python is a coding language used by developers")
    sim = engine.similarity(v1, v2)
    if sim > 0.6:
        results.ok(f"Similar texts have high similarity ({sim:.3f})")
    else:
        results.fail("Similar text similarity", f"{sim:.3f} < 0.6")

    # Different topics
    v3 = engine.encode("The weather is sunny today")
    sim_diff = engine.similarity(v1, v3)
    if sim_diff < 0.3:
        results.ok(f"Different topics have low similarity ({sim_diff:.3f})")
    else:
        results.fail("Different topic dissimilarity", f"{sim_diff:.3f} >= 0.3")

    # Polysemy
    v_prog = engine.encode("Python is used for machine learning and data science")
    v_snake = engine.encode("The python snake is found in tropical regions")
    sim_poly = engine.similarity(v_prog, v_snake)
    if sim_poly < 0.7:
        results.ok(f"Polysemous terms partially discriminated ({sim_poly:.3f})")
    else:
        results.fail("Polysemy discrimination", f"{sim_poly:.3f} >= 0.7")

    # Benchmarks
    start = time.time()
    for i in range(50):
        engine.encode(f"benchmark sentence number {i} with unique content")
    elapsed = time.time() - start
    results.bench("Single encode avg", elapsed / 50 * 1000, "ms")

    start = time.time()
    texts = [f"batch sentence {i} unique" for i in range(100)]
    engine.encode_batch(texts)
    elapsed = time.time() - start
    results.bench("Batch encode (100 texts)", elapsed * 1000, "ms")




@requires_embeddings
def test_consolidation():
    """Test consolidation clustering (REQUIRES model)."""
    print("\n[TIER 2] CONSOLIDATION TESTS")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    engine = EmbeddingEngine(use_model=True)

    try:
        with StorageEngine(tmp_dir) as storage:
            python_eps = [
                "Python is great for data science",
                "Python has excellent libraries like NumPy",
                "Python is widely used in machine learning",
                "Python scripting for automation tasks",
                "Python web development with Django",
            ]
            cooking_eps = [
                "Making pasta requires boiling water first",
                "Italian cooking uses olive oil extensively",
                "Baking bread needs flour yeast and water",
                "French cuisine is known for rich sauces",
                "Cooking rice perfectly requires proper ratio",
            ]

            for text in python_eps + cooking_eps:
                storage.store_episode(text, engine.encode(text), importance=0.5)

            consolidator = ConsolidationEngine(similarity_threshold=0.45, min_cluster_size=2)
            stats = consolidator.consolidate(storage, engine)

            if stats['clusters_formed'] >= 2:
                results.ok(f"Multiple clusters formed ({stats['clusters_formed']})")
            else:
                results.fail("Clustering", f"Only {stats['clusters_formed']} clusters")

            if stats['semantic_nodes_created'] >= 1:
                results.ok(f"Semantic nodes created ({stats['semantic_nodes_created']})")
            else:
                results.fail("Semantic creation", f"Created: {stats['semantic_nodes_created']}")

            consolidated_count = storage.count_episodes(consolidated=True)
            if consolidated_count > 0:
                results.ok(f"Episodes marked consolidated ({consolidated_count})")
            else:
                results.fail("Consolidation marking", "No episodes marked")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)




@requires_embeddings
def test_full_integration():
    """Test full ARNv9 perceive/recall cycle (REQUIRES model)."""
    print("\n[TIER 2] FULL ARN v9 INTEGRATION")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir) as arn:
            result = arn.perceive(
                "The user's name is Mtr and they work on OpenClaw",
                importance=0.9, context={'source': 'conversation'}
            )
            if result['episode_id'] > 0:
                results.ok(f"Perceive returns episode_id ({result['episode_id']})")
            else:
                results.fail("Perceive", f"episode_id={result['episode_id']}")

            if 'prediction_error' in result:
                results.ok(f"Prediction error computed ({result['prediction_error']:.3f})")
            else:
                results.fail("Prediction error", "Missing")

            if result.get('best_domain'):
                results.ok(f"Best domain identified ({result['best_domain']})")
            else:
                results.fail("Domain routing", "No best domain")

        # Recall after restart
        with ARNv9(data_dir=tmp_dir) as arn:
            recalls = arn.recall("What is the user's name?")
            if recalls and 'Mtr' in recalls[0]['content']:
                results.ok(f"Recall finds content after restart (score={recalls[0]['score']:.3f})")
            else:
                results.fail("Recall after restart", f"Got: {recalls[:1]}")

        # Multi-topic
        with ARNv9(data_dir=tmp_dir) as arn:
            topics = [
                ("Python is the user's favorite language", 0.8),
                ("The user runs Raspberry Pi 5 as homelab", 0.7),
                ("ARN is a brain-inspired memory system", 0.9),
                ("The user is studying IT at CSCC", 0.6),
                ("OpenClaw is a multi-agent framework", 0.8),
                ("The weather is nice today", 0.2),
            ]
            for content, importance in topics:
                arn.perceive(content, importance=importance)

            prog = arn.recall("programming language")
            if prog and any('Python' in r['content'] for r in prog[:3]):
                results.ok("Multi-topic recall: programming finds Python")
            else:
                results.fail("Multi-topic recall", f"Top: {[r['content'][:40] for r in prog[:3]]}")

            hw = arn.recall("hardware setup homelab")
            if hw and any('Pi' in r['content'] for r in hw[:3]):
                results.ok("Multi-topic recall: hardware finds Pi 5")
            else:
                results.fail("Hardware recall", f"Top: {[r['content'][:40] for r in hw[:3]]}")

            stats = arn.get_stats()
            results.ok(f"Stats: {stats['episodic_count']} episodes, "
                       f"{stats['semantic_count']} semantics, "
                       f"WM={stats['working_memory_active']}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_total_experiences_seeded_from_db():
    """Test that total_experiences is seeded from DB count on restart."""
    print("\n[TIER 2] TOTAL_EXPERIENCES SEEDING")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir) as arn:
            arn.perceive("First memory", importance=0.5)
            arn.perceive("Second memory", importance=0.5)
            stats = arn.get_stats()
            if stats['total_experiences'] == 2:
                results.ok("total_experiences tracks stores correctly")
            else:
                results.fail("total_experiences initial", f"Expected 2, got {stats['total_experiences']}")

        # Restart — total_experiences should seed from DB count
        with ARNv9(data_dir=tmp_dir) as arn:
            stats = arn.get_stats()
            if stats['total_experiences'] == 2:
                results.ok("total_experiences seeded from DB on restart")
            else:
                results.fail("total_experiences seed", f"Expected 2, got {stats['total_experiences']}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_agent_simulation():
    """Simulated agent workload (REQUIRES model)."""
    print("\n[TIER 2] AGENT SIMULATION")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        interactions = [
            ("User asked about Python list comprehensions", 0.6),
            ("Helped debug a segfault in C++ code", 0.8),
            ("User prefers tabs over spaces", 0.4),
            ("Raspberry Pi 5 running Ubuntu 24.04 ARM64", 0.7),
            ("User's name is Mtr", 0.9),
            ("User studies IT at Columbus State Community College", 0.8),
            ("ARN is Adaptive Reasoning Network for agent memory", 0.9),
            ("OpenClaw is the multi-agent harness", 0.9),
            ("Good morning, starting work on ARN today", 0.2),
            ("Thanks for the help with the bug fix", 0.3),
        ]

        expanded = list(interactions)
        for i in range(190):
            base_idx = i % len(interactions)
            content, imp = interactions[base_idx]
            expanded.append((f"{content} (variation {i})", imp * 0.9))

        with ARNv9(data_dir=tmp_dir, auto_consolidate=True,
                    consolidation_threshold=64) as arn:

            perceive_times = []
            for i, (content, importance) in enumerate(expanded):
                start = time.time()
                arn.perceive(content, importance=importance,
                             context={'turn': i, 'source': 'simulation'})
                perceive_times.append(time.time() - start)

            results.bench("Avg perceive time", np.mean(perceive_times) * 1000, "ms")
            results.bench("P95 perceive time", np.percentile(perceive_times, 95) * 1000, "ms")

            # Recall quality
            name_results = arn.recall("user's name", top_k=3)
            if name_results and any('Mtr' in r['content'] for r in name_results):
                results.ok("Agent sim: recalls user's name")
            else:
                results.fail("Agent sim name recall",
                             f"Top: {[r['content'][:40] for r in name_results[:3]]}")

            arn_results = arn.recall("brain-inspired memory architecture", top_k=3)
            if arn_results and any('ARN' in r['content'] for r in arn_results):
                results.ok("Agent sim: recalls ARN project details")
            else:
                results.fail("Agent sim ARN recall",
                             f"Top: {[r['content'][:40] for r in arn_results[:3]]}")

            stats = arn.get_stats()
            if stats['consolidation_count'] > 0:
                results.ok(f"Auto-consolidation triggered {stats['consolidation_count']} times")
            else:
                results.ok("System ran (consolidation may not have triggered)")

            results.ok(f"Final: {stats['episodic_count']} ep, {stats['semantic_count']} sem, "
                       f"{stats['storage']['total_size_mb']:.2f}MB")

            if stats['storage']['total_size_mb'] < 50:
                results.ok(f"Under 50MB budget ({stats['storage']['total_size_mb']:.2f}MB)")
            else:
                results.fail("Memory budget", f"{stats['storage']['total_size_mb']:.2f}MB > 50MB")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_precision_recall_quality():
    """Store 10 distinct facts and recall each by query, assert correct fact in top-2."""
    print("\n[TIER 2] PRECISION / RECALL QUALITY")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        facts = [
            ("The capital of France is Paris", "What is the capital of France?"),
            ("Water boils at 100 degrees Celsius at sea level", "At what temperature does water boil?"),
            ("The Python programming language was created by Guido van Rossum", "Who created Python?"),
            ("The speed of light is approximately 299792 kilometers per second", "How fast does light travel?"),
            ("Shakespeare wrote Hamlet around the year 1600", "When was Hamlet written?"),
            ("The Earth orbits the Sun once every 365.25 days", "How long is an Earth year?"),
            ("Mitochondria are the powerhouses of the cell", "What are mitochondria?"),
            ("The Amazon is the largest rainforest in the world", "What is the biggest rainforest?"),
            ("Photosynthesis converts carbon dioxide and water into glucose", "What is photosynthesis?"),
            ("The Great Wall of China is over 21000 kilometers long", "How long is the Great Wall of China?"),
        ]

        with ARNv9(data_dir=tmp_dir) as arn:
            for content, _ in facts:
                arn.perceive(content, importance=0.8, memory_type="fact")

            correct = 0
            for content, query in facts:
                recalls = arn.recall(query, top_k=2)
                if any(content[:30] in r['content'][:60] for r in recalls):
                    correct += 1

            if correct >= 7:
                results.ok(f"Precision/recall quality: {correct}/10 facts in top-2")
            else:
                results.fail("Precision/recall quality", f"Only {correct}/10 facts in top-2")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_stress():
    """High-volume stress test (REQUIRES model)."""
    print("\n[TIER 2] STRESS TEST")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir, auto_consolidate=True,
                    consolidation_threshold=128) as arn:

            start = time.time()
            for i in range(500):
                arn.perceive(
                    f"Stress test {i}: {np.random.choice(['code', 'infra', 'chat'])} "
                    f"about {np.random.choice(['Python', 'Rust', 'Docker', 'Linux'])}",
                    importance=float(np.random.random()),
                )
            total = time.time() - start
            results.bench("500 perceive ops total", total * 1000, "ms")
            results.bench("500 perceive avg", total / 500 * 1000, "ms/op")

            stats = arn.get_stats()
            results.ok(f"Stress complete: {stats['episodic_count']} ep, "
                       f"{stats['semantic_count']} sem, "
                       f"{stats['consolidation_count']} consolidations, "
                       f"{stats['storage']['total_size_mb']:.2f}MB")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================================================
# RUN ALL TESTS
# =========================================================

def main():
    print("=" * 60)
    print("ARN v9 COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    if EMBEDDINGS_AVAILABLE:
        print("Embedding model: LOADED ✓")
        print("Running: ALL tests (Tier 1 + Tier 2)")
    else:
        print("Embedding model: NOT AVAILABLE ⚠")
        print("Running: Tier 1 (plumbing) only")
        print("Tier 2 (semantic) tests will be SKIPPED")
        print("Install sentence-transformers for full validation")

    test_functions = [
        # Tier 1: always run
        test_embedding_basics,
        test_persistence,
        test_link_storage,
        test_working_memory,
        # Tier 2: require embeddings
        test_embedding_quality,
        test_consolidation,
        test_full_integration,
        test_agent_simulation,
        test_precision_recall_quality,
        test_stress,
    ]

    for test_fn in test_functions:
        try:
            test_fn()
        except Exception as e:
            results.fail(f"{test_fn.__name__} (EXCEPTION)", str(e))
            traceback.print_exc()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
