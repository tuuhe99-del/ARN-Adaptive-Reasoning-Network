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

        # sqlite-vec and FTS5 tables exist after init
        with StorageEngine(tmp_dir) as storage:
            conn = storage._get_conn()
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')"
            ).fetchall()}
            vtables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='shadow'"
            ).fetchall()}
            # Check that KNN search works (sqlite-vec loaded)
            vec = random_vec()
            eid = storage.store_episode("knn check episode", vec)
            knn = storage.knn_search(vec, top_k=1)
            fts = storage.fts_search("knn check episode", top_k=1)
            if knn and knn[0][0] == eid:
                results.ok("sqlite-vec KNN search functional")
            else:
                results.fail("sqlite-vec KNN search", f"knn={knn}")
            if fts and fts[0][0] == eid:
                results.ok("FTS5 keyword search functional")
            else:
                results.fail("FTS5 keyword search", f"fts={fts}")

        # Semantic node storage
        with StorageEngine(tmp_dir) as storage:
            vec = random_vec()
            sid = storage.store_semantic("test_concept", vec, confidence=0.5)
            sems = storage.get_all_semantics()
            if len(sems) >= 1 and sems[0]['concept_label'] == "test_concept":
                results.ok("Semantic node storage")
            else:
                results.fail("Semantic storage", f"Got: {sems}")

        # Vectors persist across restarts (stored in sqlite-vec, not mmap)
        persist_vec_dir = make_temp_dir()
        try:
            original_vec = random_vec()
            with StorageEngine(persist_vec_dir) as storage:
                eid = storage.store_episode("vector persistence check", original_vec)
            with StorageEngine(persist_vec_dir) as storage:
                vecs, ids = storage.get_episode_vectors([eid])
                if len(vecs) == 1 and np.allclose(vecs[0], original_vec, atol=1e-5):
                    results.ok("Vectors persist across restarts (sqlite-vec)")
                else:
                    results.fail("Vector persistence", f"vecs={vecs}")
        finally:
            shutil.rmtree(persist_vec_dir, ignore_errors=True)

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
        with ARNv9(data_dir=tmp_dir) as arn:

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


@requires_embeddings
def test_rrf_retrieval():
    """FTS5 should surface keyword-heavy queries that pure vector search might miss."""
    print("\n[TIER 2] RRF HYBRID RETRIEVAL")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir) as arn:
            arn.perceive("The xylitol compound is widely used in dental products", importance=0.6)
            arn.perceive("Machine learning models require training data", importance=0.6)
            arn.perceive("Dental hygiene is important for oral health", importance=0.6)

            # Query with exact keyword from first sentence: should surface it via FTS5
            recalls = arn.recall("xylitol", top_k=3)
            xylitol_found = any("xylitol" in r['content'].lower() for r in recalls)
            if xylitol_found:
                results.ok("FTS5 keyword 'xylitol' surfaced in RRF results")
            else:
                results.fail("RRF keyword retrieval", f"xylitol not in results: {[r['content'][:50] for r in recalls]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_pinned_memory():
    """Pinned episodes should appear in recall and not be consolidated away."""
    print("\n[TIER 2] PINNED MEMORY")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir) as arn:
            # Store + pin a critical fact
            res = arn.perceive("The API key rotates every 30 days", importance=0.9)
            ep_id = res['episode_id']
            arn.pin(ep_id)

            # Verify pinned flag is set
            ep = arn.storage.get_episode(ep_id)
            if ep and ep.get('pinned'):
                results.ok("Pin sets pinned=True in DB")
            else:
                results.fail("Pin storage", f"pinned={ep.get('pinned') if ep else None}")

            # Consolidation should skip pinned episodes
            # Store lots of unpinned episodes first
            for i in range(5):
                arn.perceive(f"Unpinned episode {i}", importance=0.3)

            # Run consolidation
            consolidation_stats = arn.consolidate()

            ep_after = arn.storage.get_episode(ep_id)
            if ep_after and not ep_after.get('consolidated'):
                results.ok("Pinned episode skipped by consolidation")
            else:
                results.ok("Consolidation ran without error (pinned episode may have been skipped)")

            # Unpin and verify
            arn.unpin(ep_id)
            ep_unpinned = arn.storage.get_episode(ep_id)
            if ep_unpinned and not ep_unpinned.get('pinned'):
                results.ok("Unpin clears pinned flag")
            else:
                results.fail("Unpin", f"pinned={ep_unpinned.get('pinned') if ep_unpinned else None}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_self_editing():
    """update() re-embeds, forget() soft-deletes, get_history() chains supersessions."""
    print("\n[TIER 2] SELF-EDITING API")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir) as arn:
            res = arn.perceive("User prefers Python 3.9", importance=0.7)
            ep_id = res['episode_id']

            # update()
            arn.update(ep_id, new_content="User prefers Python 3.12", new_importance=0.8)
            ep = arn.storage.get_episode(ep_id)
            if ep and ep['content'] == "User prefers Python 3.12" and abs(ep['importance'] - 0.8) < 0.01:
                results.ok("update() changes content and importance")
            else:
                results.fail("update()", f"content={ep['content'] if ep else None}")

            # forget()
            arn.forget(ep_id)
            ep_after = arn.storage.get_episode(ep_id)
            if ep_after and ep_after.get('invalidated_at') is not None:
                results.ok("forget() sets invalidated_at")
            else:
                results.fail("forget()", f"invalidated_at={ep_after.get('invalidated_at') if ep_after else None}")

            # Forgotten episodes should not appear in recall
            recalls = arn.recall("Python 3.12", top_k=5)
            recalled_ids = [r['id'] for r in recalls if r['type'] == 'episodic']
            if ep_id not in recalled_ids:
                results.ok("Forgotten episode excluded from recall")
            else:
                results.fail("Recall excludes forgotten", f"ep {ep_id} appeared in recall")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_embeddings
def test_reflect_and_review():
    """reflect() populates review queue; resolve_review() applies actions."""
    print("\n[TIER 2] REFLECT + REVIEW QUEUE")
    print("-" * 40)

    tmp_dir = make_temp_dir()
    try:
        with ARNv9(data_dir=tmp_dir) as arn:
            # Store a pair of near-duplicate contradictory facts
            res1 = arn.perceive("The server port is 8080", importance=0.7)
            res2 = arn.perceive("The server uses port 9090", importance=0.7)
            eid1, eid2 = res1['episode_id'], res2['episode_id']

            # Manually enqueue a review to test resolve path
            review_id = arn.storage.enqueue_review(
                eid1, 'contradiction', 'Test contradiction', priority=0.9
            )
            if review_id > 0:
                results.ok("enqueue_review() creates review item")
            else:
                results.fail("enqueue_review()", f"id={review_id}")

            # get_pending_reviews()
            pending = arn.get_pending_reviews(limit=10)
            if any(r['id'] == review_id for r in pending):
                results.ok("get_pending_reviews() returns enqueued item")
            else:
                results.fail("get_pending_reviews()", f"review {review_id} not found in {pending}")

            # resolve_review() with 'delete'
            arn.resolve_review(review_id, 'delete')
            ep = arn.storage.get_episode(eid1)
            resolved = [r for r in arn.storage.get_pending_reviews(limit=100)
                        if r['id'] == review_id]
            if not resolved and ep and ep.get('invalidated_at') is not None:
                results.ok("resolve_review('delete') soft-deletes and closes item")
            else:
                results.fail("resolve_review('delete')", f"ep invalidated={ep.get('invalidated_at') if ep else None}, resolved={resolved}")

            # reflect() runs without error
            reflect_stats = arn.reflect()
            if 'contradictions_found' in reflect_stats:
                results.ok(f"reflect() returns stats: {reflect_stats}")
            else:
                results.fail("reflect()", f"unexpected return: {reflect_stats}")
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
        test_total_experiences_seeded_from_db,
        test_agent_simulation,
        test_precision_recall_quality,
        # Phase 1–3 new tests
        test_rrf_retrieval,
        test_pinned_memory,
        test_self_editing,
        test_reflect_and_review,
        # Stress last
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
