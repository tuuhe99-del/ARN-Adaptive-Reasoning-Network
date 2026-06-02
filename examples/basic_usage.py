"""
Basic ARN usage — store and recall memories.

Run: python examples/basic_usage.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arn_v9.core.cognitive import ARNv9

# Use a temp dir so this example is self-contained
data_dir = tempfile.mkdtemp(prefix="arn_example_")
print(f"Using temp data dir: {data_dir}\n")

arn = ARNv9(data_dir=data_dir, use_embeddings=True)

# ── Store memories ──────────────────────────────────────────────────────────

result = arn.perceive("User prefers Python for scripting", importance=0.8)
python_id = result['episode_id']
print(f"Stored 'Python preference' as episode {python_id}")

result = arn.perceive("User's main project is ARN — a memory system for AI agents", importance=0.9)
project_id = result['episode_id']
print(f"Stored 'main project' as episode {project_id}")

arn.perceive("User deployed the app to a Raspberry Pi 5 with 8GB RAM", importance=0.7)
arn.perceive("User prefers dark mode in all tools", importance=0.6)
arn.perceive("Redis is used as the cache layer in the production stack", importance=0.75)

# ── Pin an identity fact so it survives consolidation and decay ──────────────

arn.pin(python_id)
print(f"\nPinned episode {python_id} (Python preference)")

# ── Recall ───────────────────────────────────────────────────────────────────

print("\n── Recall: 'what language does the user prefer?' ──")
results = arn.recall("what language does the user prefer?", top_k=3)
for r in results:
    print(f"  [{r['source']}] {r['content']}  (score: {r['score']:.3f})")

print("\n── Recall: 'what project is the user working on?' ──")
results = arn.recall("what project is the user working on?", top_k=3)
for r in results:
    print(f"  [{r['source']}] {r['content']}  (score: {r['score']:.3f})")

# ── Update a fact (re-embeds automatically) ──────────────────────────────────

print("\n── Updating Pi RAM to 16GB ──")
arn.update(
    episode_id=result['id'] if results else python_id,
    new_content="User deployed the app to a Raspberry Pi 5 with 16GB RAM",
)

# ── Session lifecycle ─────────────────────────────────────────────────────────

print("\n── Session lifecycle ──")
storage = arn.storage
storage.create_session("example-session-001", reason_start="demo run")
print("Session started: example-session-001")

import numpy as np
vec = arn.embedder.encode("User asked about the project architecture")
storage.store_episode(
    content="User asked about the project architecture",
    vector=vec,
    role="user",
    session_id="example-session-001",
    importance=0.5,
)
session = storage.end_session("example-session-001", reason_end="demo complete")
print(f"Session ended. Episodes captured: {session['episode_count']}")

# ── Post-session reflection ───────────────────────────────────────────────────

print("\n── Running reflect() ──")
stats = arn.reflect()
print(f"  Contradictions queued: {stats['contradictions_found']}")
print(f"  Importances recalibrated: {stats['importance_recalibrated']}")

reviews = arn.get_pending_reviews()
if reviews:
    print(f"\n── {len(reviews)} review item(s) pending ──")
    for r in reviews[:3]:
        print(f"  [{r['review_type']}] episode {r['episode_id']}: {r['reason'][:80]}")
else:
    print("\nNo review items pending.")

# ── Stats ─────────────────────────────────────────────────────────────────────

stats = arn.get_stats()
print(f"\n── Stats ──")
print(f"  Episodes: {stats['episodic_count']}")
print(f"  Semantic nodes: {stats['semantic_count']}")
print(f"  Model degraded (no embeddings): {stats['embeddings']['degraded']}")

arn.close()
print("\nDone.")
