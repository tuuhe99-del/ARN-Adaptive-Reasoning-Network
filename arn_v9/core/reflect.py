"""
ARN v9 Post-Session Reflection System.

Three analysis passes run after a session completes:
1. scan_contradictions — find near-duplicate episodes with divergent content
2. recalibrate_importance — boost frequently-accessed episodes
3. detect_ambiguity — flag low-importance episodes that are accessed often
"""

import time
import numpy as np
from typing import List, Dict, Any

from ..storage.persistence import StorageEngine
from .embeddings import EmbeddingEngine


def scan_contradictions(storage: StorageEngine,
                        embedder: EmbeddingEngine,
                        max_candidates: int = 200) -> List[Dict[str, Any]]:
    """
    Find pairs of active episodes that are semantically similar but
    have divergent word content — likely candidates for contradiction.

    Returns a list of contradiction dicts sorted by similarity descending.
    """
    now = time.time()
    episodes = [
        ep for ep in storage.get_all_episodes(consolidated=None)
        if not ep.get('pinned', False)
        and ep.get('invalidated_at') is None
        and ep.get('superseded_by') is None
    ]

    # Sort by importance × recency, take top N
    for ep in episodes:
        age_days = (now - ep['created_at']) / 86400.0
        ep['_score'] = ep['importance'] * max(0.01, 1.0 - age_days / 90.0)
    episodes.sort(key=lambda e: e['_score'], reverse=True)
    candidates = episodes[:max_candidates]

    if len(candidates) < 2:
        return []

    # Get their vectors
    ids = [ep['id'] for ep in candidates]
    vecs, vec_ids = storage.get_episode_vectors(ids)
    if len(vecs) < 2:
        return []

    id_to_idx = {eid: i for i, eid in enumerate(vec_ids)}
    id_to_ep = {ep['id']: ep for ep in candidates}

    # Pairwise cosine similarity (upper triangle only)
    contradictions = []
    n = len(vec_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim < 0.85:
                continue

            ep_i = id_to_ep.get(vec_ids[i])
            ep_j = id_to_ep.get(vec_ids[j])
            if not ep_i or not ep_j:
                continue

            # Word overlap (Jaccard on tokens)
            words_i = set(ep_i['content'].lower().split())
            words_j = set(ep_j['content'].lower().split())
            union = len(words_i | words_j)
            if union == 0:
                continue
            jaccard = len(words_i & words_j) / union
            if jaccard >= 0.4:
                continue

            # Older episode is the one to flag
            older = ep_i if ep_i['created_at'] < ep_j['created_at'] else ep_j
            newer = ep_j if older is ep_i else ep_i

            contradictions.append({
                'older_episode_id': older['id'],
                'newer_episode_id': newer['id'],
                'similarity': sim,
                'word_overlap': jaccard,
                'older_content': older['content'][:120],
                'newer_content': newer['content'][:120],
            })

    contradictions.sort(key=lambda c: c['similarity'], reverse=True)
    return contradictions


def recalibrate_importance(storage: StorageEngine) -> List[Dict[str, Any]]:
    """
    Identify episodes whose access_count warrants an importance boost.
    Returns suggested changes (does NOT apply them — caller decides).
    """
    episodes = storage.get_all_episodes(consolidated=None)
    suggestions = []
    for ep in episodes:
        if ep.get('invalidated_at') is not None:
            continue
        ac = ep.get('access_count', 0)
        if ac < 5:
            continue
        boost = (ac // 5) * 0.05
        suggested = min(0.95, ep['importance'] + boost)
        if suggested - ep['importance'] > 0.04:
            suggestions.append({
                'episode_id': ep['id'],
                'current': ep['importance'],
                'suggested': round(suggested, 3),
            })
    return suggestions


def detect_ambiguity(storage: StorageEngine,
                     max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Find episodes that are accessed often but have low importance —
    likely undervalued or ambiguous facts worth reviewing.
    """
    episodes = storage.get_all_episodes(consolidated=None)
    ambiguous = []
    for ep in episodes:
        if ep.get('invalidated_at') is not None:
            continue
        if ep.get('access_count', 0) > 3 and ep['importance'] < 0.2:
            ambiguous.append({
                'episode_id': ep['id'],
                'content': ep['content'][:120],
                'access_count': ep['access_count'],
                'importance': ep['importance'],
            })
    return ambiguous[:max_items]
