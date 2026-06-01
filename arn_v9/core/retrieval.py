"""
Retrieval utility functions for ARN v9.

Provides: RRF fusion, recency decay, MMR reranking, score-gap cutoff.
"""

import math
import time
import numpy as np
from typing import List, Tuple, Dict, Optional

RECENCY_HALF_LIFE_DAYS = 14.0


def fuse_rrf(vec_results: List[Tuple[int, float]],
             fts_results: List[Tuple[int, float]],
             entity_results: Optional[List[Tuple[int, float]]] = None,
             k: int = 60) -> Dict[int, float]:
    """Reciprocal Rank Fusion across up to 3 ranked lists."""
    vec_ranks = {eid: i + 1 for i, (eid, _) in enumerate(vec_results)}
    fts_ranks = {eid: i + 1 for i, (eid, _) in enumerate(fts_results)}
    ent_ranks = {eid: i + 1 for i, (eid, _) in enumerate(entity_results or [])}
    all_ids = set(vec_ranks) | set(fts_ranks) | set(ent_ranks)
    fallback = len(all_ids) + 1
    return {
        eid: sum(
            1.0 / (k + ranks.get(eid, fallback))
            for ranks in [vec_ranks, fts_ranks, ent_ranks]
        )
        for eid in all_ids
    }


def recency_score(created_at: float,
                  half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay score based on age."""
    days = (time.time() - created_at) / 86400.0
    return math.exp(-math.log(2) / half_life_days * days)


def mmr_rerank(query_emb: np.ndarray,
               results: List[dict],
               result_vecs: np.ndarray,
               lambda_param: float = 0.7,
               top_k: int = 5) -> List[dict]:
    """Maximal Marginal Relevance: balance relevance and diversity."""
    if not results or result_vecs.shape[0] == 0:
        return results[:top_k]
    top_k = min(top_k, len(results))
    query_sims = result_vecs @ query_emb
    selected: List[int] = []
    remaining = list(range(len(results)))
    while len(selected) < top_k and remaining:
        best, best_score = None, float('-inf')
        for i in remaining:
            if selected:
                redundancy = float(np.max(result_vecs[selected] @ result_vecs[i]))
            else:
                redundancy = 0.0
            score = lambda_param * float(query_sims[i]) - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score, best = score, i
        if best is not None:
            selected.append(best)
            remaining.remove(best)
    return [results[i] for i in selected]


def score_gap_cutoff(results: List[dict],
                     top_k: int = 5,
                     min_gap_ratio: float = 0.15) -> List[dict]:
    """Return items above the largest relative score gap in the top results."""
    if len(results) <= 1:
        return results[:top_k]
    sorted_r = sorted(results, key=lambda r: r['score'], reverse=True)
    scores = [r['score'] for r in sorted_r]
    score_range = scores[0] - scores[-1]
    if score_range < 1e-6:
        return sorted_r[:top_k]
    check_n = min(len(scores), top_k + 5)
    best_gap, best_cut = 0.0, top_k
    for i in range(1, check_n):
        gap = (scores[i - 1] - scores[i]) / score_range
        if gap > best_gap and gap >= min_gap_ratio:
            best_gap, best_cut = gap, i
    return sorted_r[:max(1, best_cut)]
