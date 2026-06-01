"""
ARN v9: Adaptive Reasoning Network
====================================

Hybrid retrieval memory system:
- Episodic storage with sqlite-vec (vector KNN) + FTS5 (keyword) + entity matching
- Reciprocal Rank Fusion across all retrieval signals
- Bi-temporal facts: valid_from / valid_until with supersedes chains
- Working memory ring buffer always surfaced at recall top
- Post-session reflection with user-in-the-loop reconciliation
"""

import math
import numpy as np
import time
import json
import os
import logging
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from .embeddings import EmbeddingEngine, EMBEDDING_DIM
from .retrieval import fuse_rrf, recency_score, mmr_rerank, score_gap_cutoff
from ..storage.persistence import StorageEngine

logger = logging.getLogger("arn.core")

AUTO_LINK_SIMILARITY_THRESHOLD = 0.38


# =========================================================
# WORKING MEMORY (Fixed from original)
# =========================================================

@dataclass
class WorkingMemorySlot:
    """A single slot in working memory."""
    content: str
    vector: np.ndarray
    activation: float
    timestamp: float
    source_id: int  # episode ID that produced this


class WorkingMemory:
    """
    Prefrontal cortex-inspired active maintenance.
    
    Fixes from original:
    - Uses a proper list instead of deque with mismatched indexing
    - Activation decay is time-based, not just per-tick
    - Rehearsal mechanism to keep important items active
    """
    
    def __init__(self, max_slots: int = 7, embedding_dim: int = None):
        self.max_slots = max_slots
        self.slots: List[Optional[WorkingMemorySlot]] = [None] * max_slots
        self._slot_count = 0
        # Default to legacy constant if not provided (backwards compat)
        self.embedding_dim = embedding_dim if embedding_dim is not None else EMBEDDING_DIM
    
    def add(self, content: str, vector: np.ndarray,
            priority: float = 1.0, source_id: int = -1):
        """Add item to working memory, evicting lowest activation if full."""
        new_slot = WorkingMemorySlot(
            content=content,
            vector=vector,
            activation=priority,
            timestamp=time.time(),
            source_id=source_id
        )
        
        # Find empty slot or lowest activation
        min_act = float('inf')
        min_idx = 0
        empty_idx = -1
        
        for i, slot in enumerate(self.slots):
            if slot is None:
                empty_idx = i
                break
            if slot.activation < min_act:
                min_act = slot.activation
                min_idx = i
        
        if empty_idx >= 0:
            self.slots[empty_idx] = new_slot
            self._slot_count += 1
        elif priority > min_act:
            self.slots[min_idx] = new_slot
        # else: new item not important enough; discard
    
    def decay(self, elapsed_seconds: float = 1.0, rate: float = 0.05):
        """Time-based decay of working memory activations."""
        decay_factor = max(0.0, 1.0 - rate * elapsed_seconds)
        for slot in self.slots:
            if slot is not None:
                slot.activation *= decay_factor
                if slot.activation < 0.01:
                    # Slot decayed below threshold; free it
                    idx = self.slots.index(slot)
                    self.slots[idx] = None
                    self._slot_count -= 1
    
    def get_active(self) -> List[WorkingMemorySlot]:
        """Get all active slots, sorted by activation."""
        active = [s for s in self.slots if s is not None]
        active.sort(key=lambda s: s.activation, reverse=True)
        return active
    
    def get_context_vector(self) -> Optional[np.ndarray]:
        """
        Compute a weighted average of working memory contents.
        This represents the current "context" for prediction.
        """
        active = self.get_active()
        if not active:
            return None
        
        total_activation = sum(s.activation for s in active)
        if total_activation < 1e-8:
            return None
        
        context = np.zeros(self.embedding_dim, dtype=np.float32)
        for slot in active:
            context += (slot.activation / total_activation) * slot.vector
        
        norm = np.linalg.norm(context)
        if norm > 0:
            context /= norm
        return context
    
    @property
    def count(self) -> int:
        return self._slot_count


# =========================================================
# CONSOLIDATION ENGINE
# =========================================================

class ConsolidationEngine:
    """
    Episodic → Semantic consolidation via clustering.
    
    Replaces the broken first-word-match approach with:
    1. Similarity-based clustering of episodes
    2. Multi-episode evidence requirement
    3. Contradiction detection and logging
    4. Incremental semantic node updates
    
    Neuroscience basis: During SWS (slow-wave sleep), the hippocampus
    replays episode sequences. Repeated co-activation of similar episodes
    strengthens neocortical traces (semantic memory). We simulate this
    with batch clustering during consolidation sweeps.
    """
    
    def __init__(self, similarity_threshold: float = 0.55,
                 min_cluster_size: int = 2,
                 contradiction_threshold: float = 0.3,
                 max_semantic_nodes: int = 2048):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.contradiction_threshold = contradiction_threshold
        self.max_semantic_nodes = max_semantic_nodes
    
    def consolidate(self, storage: StorageEngine, embedder: EmbeddingEngine,
                    batch_size: int = 64) -> dict:
        """
        Run a consolidation sweep.
        
        Returns stats about what was consolidated.
        """
        stats = {
            'episodes_processed': 0,
            'clusters_formed': 0,
            'semantic_nodes_created': 0,
            'semantic_nodes_updated': 0,
            'contradictions_found': 0,
            'episodes_pruned': 0,
        }
        
        # Get unconsolidated episodes sorted by replay priority
        episodes = storage.get_all_episodes(consolidated=False)
        if len(episodes) < self.min_cluster_size:
            return stats
        
        # Calculate replay priorities
        now = time.time()
        for ep in episodes:
            recency = 1.0 / (1.0 + (now - ep['created_at']) / 3600.0)
            surprise = ep['prediction_error']
            relevance = np.log1p(ep['access_count'])
            ep['_replay_priority'] = (
                ep['importance'] * 0.4 +
                recency * 0.2 +
                surprise * 0.3 +
                relevance * 0.1
            )
        
        # Sort by priority (highest first)
        episodes.sort(key=lambda e: e['_replay_priority'], reverse=True)
        
        # Take top batch
        batch = episodes[:batch_size]
        stats['episodes_processed'] = len(batch)
        
        # Get vectors for the batch
        batch_ids = [ep['id'] for ep in batch]
        vectors, vec_ids = storage.get_episode_vectors(batch_ids)
        
        if len(vectors) == 0:
            return stats
        
        # Build ID-to-vector map
        id_to_vec = {eid: vec for eid, vec in zip(vec_ids, vectors)}
        
        # Phase 1: Cluster episodes by semantic similarity
        clusters = self._cluster_episodes(batch, id_to_vec)
        stats['clusters_formed'] = len(clusters)
        
        # Phase 2: For each significant cluster, create/update semantic nodes
        existing_semantics = storage.get_all_semantics()
        sem_vectors, sem_ids = storage.get_semantic_vectors()
        
        consolidated_episode_ids = []
        
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
            
            # Compute cluster centroid
            cluster_vecs = np.array([id_to_vec[ep['id']] for ep in cluster])
            centroid = cluster_vecs.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid /= centroid_norm
            
            # Extract concept label (most common significant words)
            label = self._extract_label(cluster)
            
            # Check for contradictions within cluster
            contradictions = self._detect_contradictions(cluster, id_to_vec)
            if contradictions:
                stats['contradictions_found'] += len(contradictions)
            
            # Find matching existing semantic node
            best_match_id = None
            best_match_sim = 0.0
            
            if len(sem_vectors) > 0:
                sims = sem_vectors @ centroid
                best_idx = np.argmax(sims)
                if sims[best_idx] > self.similarity_threshold:
                    best_match_id = sem_ids[best_idx]
                    best_match_sim = sims[best_idx]
            
            if best_match_id is not None:
                # Update existing semantic node
                existing = None
                for s in existing_semantics:
                    if s['id'] == best_match_id:
                        existing = s
                        break
                
                if existing:
                    # Slow learning: blend centroid into existing
                    old_vec_idx = existing['vec_index']
                    old_vec = sem_vectors[sem_ids.index(best_match_id)]
                    lr = 0.05
                    new_vec = (1 - lr) * old_vec + lr * centroid
                    new_vec /= np.linalg.norm(new_vec)
                    
                    new_confidence = min(1.0, existing['confidence'] + 0.02 * len(cluster))
                    new_evidence = existing['evidence_count'] + len(cluster)
                    
                    existing_contradictions = existing.get('contradiction_log', [])
                    existing_contradictions.extend(contradictions)
                    # Keep only latest 20 contradictions
                    existing_contradictions = existing_contradictions[-20:]
                    
                    storage.update_semantic(
                        best_match_id,
                        vector=new_vec,
                        confidence=new_confidence,
                        evidence_count=new_evidence,
                        contradiction_log=existing_contradictions
                    )
                    stats['semantic_nodes_updated'] += 1
            else:
                # Create new semantic node - include representative content
                # Pick the highest-importance episode as the representative
                rep_episode = max(cluster, key=lambda e: e.get('importance', 0))
                storage.store_semantic(
                    concept_label=label,
                    vector=centroid,
                    confidence=0.1 + 0.02 * len(cluster),
                    evidence_count=len(cluster),
                    schema={
                        'contradictions': contradictions,
                        'representative_content': rep_episode['content'],
                        'episode_contents': [e['content'][:100] for e in cluster[:5]],
                    }
                )
                stats['semantic_nodes_created'] += 1
                
                # Refresh semantic data for next cluster
                sem_vectors, sem_ids = storage.get_semantic_vectors()
                existing_semantics = storage.get_all_semantics()
            
            # Mark cluster episodes as consolidated
            cluster_ids = [ep['id'] for ep in cluster]
            consolidated_episode_ids.extend(cluster_ids)
        
        # Mark all processed episodes
        if consolidated_episode_ids:
            storage.mark_episodes_consolidated(consolidated_episode_ids)
        
        # Prune old low-importance consolidated episodes
        pruned = self._prune_old_episodes(storage)
        stats['episodes_pruned'] = pruned
        
        # Prune semantic memory if over capacity
        self._prune_semantics(storage)
        
        return stats
    
    def _cluster_episodes(self, episodes: List[dict],
                          id_to_vec: dict) -> List[List[dict]]:
        """
        Simple online clustering by similarity threshold.
        
        We don't use sklearn KMeans because:
        - We don't know K in advance
        - Episodes arrive incrementally
        - Threshold-based clustering matches the neuroscience better
          (replay strengthens connections above a threshold)
        """
        clusters: List[List[dict]] = []
        cluster_centroids: List[np.ndarray] = []
        
        for ep in episodes:
            vec = id_to_vec.get(ep['id'])
            if vec is None:
                continue
            
            # Find best matching cluster
            best_cluster = -1
            best_sim = 0.0
            
            for i, centroid in enumerate(cluster_centroids):
                sim = float(np.dot(vec, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = i
            
            if best_sim >= self.similarity_threshold and best_cluster >= 0:
                # Add to existing cluster
                clusters[best_cluster].append(ep)
                # Update centroid incrementally
                n = len(clusters[best_cluster])
                cluster_centroids[best_cluster] = (
                    (n - 1) / n * cluster_centroids[best_cluster] +
                    1 / n * vec
                )
                norm = np.linalg.norm(cluster_centroids[best_cluster])
                if norm > 0:
                    cluster_centroids[best_cluster] /= norm
            else:
                # Create new cluster
                clusters.append([ep])
                cluster_centroids.append(vec.copy())
        
        return clusters
    
    def _extract_label(self, cluster: List[dict]) -> str:
        """
        Extract a descriptive label from a cluster of episodes.
        Uses TF-IDF-like word frequency across the cluster.
        """
        from collections import Counter
        
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so',
            'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
            'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they',
            'them', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
            'why', 'how', 'if', 'then', 'else', 'because', 'about', 'up',
            'out', 'off', 'over', 'under', 'again', 'once', 'here', 'there',
        }
        
        word_counts = Counter()
        for ep in cluster:
            words = ep['content'].lower().split()
            meaningful = [w.strip('.,!?;:()[]{}"\'-') for w in words
                         if w.lower().strip('.,!?;:()[]{}"\'-') not in stop_words
                         and len(w) > 2]
            word_counts.update(meaningful)
        
        # Top 3 most common words
        top_words = [w for w, _ in word_counts.most_common(5) if w][:3]
        if top_words:
            return '_'.join(top_words)
        return f"cluster_{int(time.time())}"
    
    def _detect_contradictions(self, cluster: List[dict],
                                id_to_vec: dict) -> List[dict]:
        """
        Detect contradictory episodes within a cluster.
        
        Two episodes contradict if they're semantically similar
        (same topic) but have low content overlap and different factual claims.
        
        This is a simplified heuristic — full contradiction detection
        would require NLI (natural language inference), which is too
        heavy for Pi 5.
        """
        contradictions = []
        
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                ep_i = cluster[i]
                ep_j = cluster[j]
                
                vec_i = id_to_vec.get(ep_i['id'])
                vec_j = id_to_vec.get(ep_j['id'])
                
                if vec_i is None or vec_j is None:
                    continue
                
                # High semantic similarity but different content
                sim = float(np.dot(vec_i, vec_j))
                
                # Simple content overlap check
                words_i = set(ep_i['content'].lower().split())
                words_j = set(ep_j['content'].lower().split())
                overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                
                # Potential contradiction: similar topic, different words
                if sim > 0.6 and overlap < self.contradiction_threshold:
                    contradictions.append({
                        'episode_a': ep_i['id'],
                        'episode_b': ep_j['id'],
                        'similarity': sim,
                        'word_overlap': overlap,
                        'content_a': ep_i['content'][:100],
                        'content_b': ep_j['content'][:100],
                        'timestamp_a': ep_i['created_at'],
                        'timestamp_b': ep_j['created_at'],
                    })
        
        return contradictions
    
    def _prune_old_episodes(self, storage: StorageEngine,
                             max_consolidated: int = 256) -> int:
        """Remove old consolidated episodes, keeping high-importance ones."""
        consolidated = storage.get_all_episodes(consolidated=True)
        if len(consolidated) <= max_consolidated:
            return 0
        
        # Sort by importance (keep highest)
        consolidated.sort(key=lambda e: e['importance'])
        
        # Remove lowest importance
        to_remove = consolidated[:len(consolidated) - max_consolidated]
        if to_remove:
            storage.delete_episodes([ep['id'] for ep in to_remove])
        
        return len(to_remove)
    
    def _prune_semantics(self, storage: StorageEngine):
        """Remove lowest-confidence semantic nodes if over capacity."""
        count = storage.count_semantics()
        if count <= self.max_semantic_nodes:
            return
        
        semantics = storage.get_all_semantics()
        semantics.sort(key=lambda s: s['confidence'])
        
        to_remove = semantics[:count - self.max_semantic_nodes]
        if to_remove:
            storage.delete_semantics([s['id'] for s in to_remove])


# =========================================================
# MAIN ARN v9 CLASS
# =========================================================

class ARNv9:
    """
    Adaptive Reasoning Network v9 — Production Implementation.
    
    Usage:
        arn = ARNv9(data_dir="/path/to/arn_data")
        
        # Store experience
        result = arn.perceive("User prefers Python for scripting", 
                              importance=0.7,
                              context={'source': 'conversation'})
        
        # Recall relevant memories
        results = arn.recall("What programming language does the user like?")
        
        # Run consolidation (during idle time)
        stats = arn.consolidate()
        
        # Shutdown cleanly
        arn.close()
    """
    
    def __init__(self, data_dir: str = "./arn_data",
                 use_embeddings: bool = True,
                 embedding_fn=None,
                 episodic_capacity: int = 4096,
                 semantic_capacity: int = 2048):

        # Core components
        self.embedder = EmbeddingEngine(use_model=use_embeddings, embedding_fn=embedding_fn)
        self.storage = StorageEngine(
            data_dir=data_dir,
            max_episodes=episodic_capacity,
            max_semantics=semantic_capacity,
            embedding_dim=self.embedder.embedding_dim,
        )
        self.working_memory = WorkingMemory(
            max_slots=7, embedding_dim=self.embedder.embedding_dim
        )
        self.consolidation_engine = ConsolidationEngine(
            max_semantic_nodes=semantic_capacity
        )

        # State
        self.total_experiences = 0
        self.consolidation_count = 0
        self._last_decay_time = time.time()
        
        # Load state
        self._load_state()
        
        # Seed total_experiences from DB if persisted state was zero but episodes exist
        if self.total_experiences == 0:
            db_count = self.storage.count_episodes()
            if db_count > 0:
                self.total_experiences = db_count
                logger.info(f"Seeded total_experiences from DB count: {db_count}")
        
        # Load working memory from disk
        self._load_working_memory()
        
        logger.info(f"ARN v9 initialized. Episodes: {self.storage.count_episodes()}, "
                     f"Semantics: {self.storage.count_semantics()}")
    
    def perceive(self, content: str, importance: float = 0.5,
                 context: dict = None, source: str = 'user',
                 memory_type: str = 'episode') -> dict:
        """
        Store new content as an episodic memory.

        Encodes content, computes prediction error vs. working memory context,
        stores the episode, auto-links to similar memories, updates working memory.
        Consolidation is opt-in — call arn.consolidate() explicitly.
        """
        now = time.time()
        elapsed = now - self._last_decay_time
        self.working_memory.decay(elapsed_seconds=elapsed)
        self._last_decay_time = now

        feature_vector = self.embedder.encode(content, mode='passage')

        wm_context = self.working_memory.get_context_vector()
        if wm_context is not None:
            prediction_error = 1.0 - float(np.dot(feature_vector, wm_context))
        else:
            prediction_error = 1.0

        episode_id = self.storage.store_episode(
            content=content,
            vector=feature_vector,
            context=context or {},
            importance=importance,
            prediction_error=prediction_error,
            source=source,
            memory_type=memory_type,
        )

        # Auto-link to top similar existing memories via KNN
        try:
            similar = self.storage.knn_search(feature_vector, top_k=6)
            links_added = 0
            for other_id, score in similar:
                if links_added >= 3:
                    break
                if other_id != episode_id and score >= 0.50:
                    self.storage.create_link(
                        episode_id, other_id, "relates_to", confidence=score
                    )
                    links_added += 1
        except Exception:
            pass

        self.working_memory.add(
            content=content,
            vector=feature_vector,
            priority=importance,
            source_id=episode_id
        )
        self.total_experiences += 1
        self._save_working_memory()

        return {
            'episode_id': episode_id,
            'prediction_error': prediction_error,
            'importance': importance,
            'memory_type': memory_type,
        }
    
    def recall(self, query: str, top_k: int = 5,
               include_semantic: bool = True,
               include_episodic: bool = True,
               memory_type: Optional[str] = None) -> List[dict]:
        """
        Retrieve relevant memories for a query.

        Uses hybrid retrieval: sqlite-vec KNN + FTS5 keyword search, fused
        via Reciprocal Rank Fusion, then scored with recency + importance,
        reranked with MMR for diversity, and trimmed at the largest score gap.
        """
        query_vector = self.embedder.encode(query, mode='query')
        results: List[dict] = []

        if include_episodic:
            # 1. Vector KNN + FTS5 search
            vec_results = self.storage.knn_search(query_vector, top_k=top_k * 4)
            fts_results = self.storage.fts_search(query, top_k=top_k * 4)

            # 2. RRF fusion
            rrf_scores = fuse_rrf(vec_results, fts_results)

            if rrf_scores:
                # 3. Fetch metadata for all candidates
                candidate_ids = list(rrf_scores.keys())
                episodes_map = {
                    ep['id']: ep
                    for ep in self.storage.get_episodes_by_ids(candidate_ids)
                }

                # 4. Filter + composite score
                scored: List[Tuple[int, float, dict]] = []
                for eid, rrf in rrf_scores.items():
                    ep = episodes_map.get(eid)
                    if ep is None:
                        continue
                    if ep.get('invalidated_at') is not None:
                        continue
                    if ep.get('superseded_by') is not None:
                        continue
                    if memory_type and ep.get('memory_type') != memory_type:
                        continue
                    rec = recency_score(ep['created_at'])
                    freq_boost = math.log1p(ep.get('access_count', 0)) * 0.05
                    score = rrf + rec * 0.3 + ep['importance'] * 0.15 + freq_boost
                    scored.append((eid, score, ep))

                # 5. Sort and take top_k * 2 candidates for MMR
                scored.sort(key=lambda x: x[1], reverse=True)
                top_scored = scored[:top_k * 2]

                if top_scored:
                    # 6. MMR reranking (needs vectors for the candidates)
                    top_ids = [eid for eid, _, _ in top_scored]
                    ep_vectors, ep_ids = self.storage.get_episode_vectors(top_ids)
                    id_to_vec_idx = {eid: i for i, eid in enumerate(ep_ids)}

                    ordered_vecs: List[np.ndarray] = []
                    ordered_items: List[dict] = []
                    for eid, score, ep in top_scored:
                        if eid not in id_to_vec_idx:
                            continue
                        vec = ep_vectors[id_to_vec_idx[eid]]
                        if vec.shape[0] != query_vector.shape[0]:
                            continue
                        ordered_vecs.append(vec)
                        ordered_items.append({
                            'type': 'episodic',
                            'id': eid,
                            'content': ep['content'],
                            'score': score,
                            'similarity': float(np.dot(vec, query_vector)),
                            'importance': ep['importance'],
                            'created_at': ep['created_at'],
                            'access_count': ep.get('access_count', 0),
                            'context': ep.get('context', {}),
                            'memory_type': ep.get('memory_type', 'episode'),
                            'source': ep.get('source', 'unknown'),
                        })

                    if ordered_items:
                        result_vecs = np.array(ordered_vecs, dtype=np.float32)
                        reranked = mmr_rerank(
                            query_vector, ordered_items, result_vecs,
                            top_k=min(top_k * 2, len(ordered_items))
                        )
                        # 7. Score gap cutoff
                        episodic_results = score_gap_cutoff(reranked, top_k=top_k)
                        results.extend(episodic_results)

                # 8. Update access counts
                for r in results:
                    if r['type'] == 'episodic':
                        self.storage.update_episode_access(r['id'])

        # Semantic memory (full vector scan — typically small corpus)
        if include_semantic:
            sem_vectors, sem_ids = self.storage.get_semantic_vectors()
            if len(sem_vectors) > 0 and sem_vectors.shape[1] == query_vector.shape[0]:
                similarities = sem_vectors @ query_vector
                semantics = self.storage.get_all_semantics()
                id_to_sem = {s['id']: s for s in semantics}
                for sid, sim in zip(sem_ids, similarities):
                    sem = id_to_sem.get(sid)
                    if sem:
                        score = float(sim) * (0.5 + 0.5 * sem['confidence'])
                        content = sem.get('schema', {}).get(
                            'representative_content', sem['concept_label']
                        )
                        results.append({
                            'type': 'semantic',
                            'id': sid,
                            'content': content,
                            'score': score,
                            'similarity': float(sim),
                            'confidence': sem['confidence'],
                            'evidence_count': sem['evidence_count'],
                            'contradictions': sem.get('contradiction_log', []),
                        })

        results.sort(key=lambda r: r['score'], reverse=True)
        top_results = results[:top_k]

        # Pull graph neighbors into the recall window
        if top_results:
            result_by_id = {r['id']: r for r in results if r['type'] == 'episodic'}
            top_ids = {r['id'] for r in top_results if r['type'] == 'episodic'}
            linked_results = []
            for r in list(top_results):
                if r['type'] != 'episodic':
                    continue
                for link in self.storage.get_links_for_episode(r['id']):
                    other_id = (
                        link['from_episode_id']
                        if link['to_episode_id'] == r['id']
                        else link['to_episode_id']
                    )
                    if other_id in top_ids or other_id == r['id']:
                        continue
                    other = result_by_id.get(other_id)
                    if other:
                        linked = dict(other)
                        linked['score'] = max(linked['score'], r['score'] * 0.95)
                        linked_results.append(linked)
                        top_ids.add(other_id)
            top_results.extend(linked_results)
            top_results.sort(key=lambda r: r['score'], reverse=True)
            top_results = top_results[:top_k]

        # Prepend active working memory items not already in results
        wm_active = self.working_memory.get_active()
        existing_ids = {r['id'] for r in top_results if r['type'] == 'episodic'}
        wm_additions = []
        for slot in wm_active:
            if slot.source_id >= 0 and slot.source_id not in existing_ids:
                wm_additions.append({
                    'type': 'episodic',
                    'id': slot.source_id,
                    'content': slot.content,
                    'score': 2.0 + slot.activation,
                    'similarity': 1.0,
                    'importance': slot.activation,
                    'created_at': slot.timestamp,
                    'access_count': 0,
                    'context': {},
                    'memory_type': 'episode',
                    'source': 'working_memory',
                    'from_working_memory': True,
                    'confidence_tier': 'high',
                    'calibrated_confidence': 1.0,
                })
                existing_ids.add(slot.source_id)
        top_results = wm_additions + top_results

        for r in top_results:
            if 'confidence_tier' not in r:
                r['confidence_tier'] = self.embedder.confidence_tier(r['similarity'])
                r['calibrated_confidence'] = round(
                    self.embedder.calibrate_similarity(r['similarity']), 3
                )

        return top_results[:top_k + len(wm_additions)]
    
    def consolidate(self) -> dict:
        """
        Run memory consolidation (the "sleep" phase).
        
        Transfers episodic memories to semantic memory through
        clustering and pattern extraction.
        """
        logger.info("Starting consolidation sweep...")
        stats = self.consolidation_engine.consolidate(
            self.storage, self.embedder
        )
        self.consolidation_count += 1
        self._save_state()
        logger.info(f"Consolidation complete: {stats}")
        return stats
    
    def get_stats(self) -> dict:
        """Return comprehensive system statistics."""
        storage_stats = self.storage.get_storage_stats()
        embed_stats = self.embedder.get_stats()
        
        return {
            'total_experiences': self.total_experiences,
            'consolidation_count': self.consolidation_count,
            'episodic_count': storage_stats['episode_count'],
            'semantic_count': storage_stats['semantic_count'],
            'working_memory_active': self.working_memory.count,
            'storage': storage_stats,
            'embeddings': embed_stats,
        }
    
    def _load_state(self):
        """Load persisted state."""
        state = self.storage.get_state('arn_state')
        if state:
            data = json.loads(state)
            self.total_experiences = data.get('total_experiences', 0)
            self.consolidation_count = data.get('consolidation_count', 0)
    
    def _save_state(self):
        """Persist current state."""
        state = {
            'total_experiences': self.total_experiences,
            'consolidation_count': self.consolidation_count,
        }
        self.storage.set_state('arn_state', json.dumps(state))
    
    def _save_working_memory(self):
        """Serialize working memory to disk (lightweight — just episode IDs)."""
        try:
            active = self.working_memory.get_active()
            wm_data = {
                # int() converts numpy int64 → Python int so json.dumps works
                "recent_input_ids": [int(s.source_id) for s in active[-10:]],
                "attention_weights": {},
                "updated_at": time.time(),
            }
            wm_path = Path(self.storage.data_dir) / "working_memory.json"
            # Atomic write: write to temp file then rename
            tmp_path = wm_path.with_suffix('.tmp')
            try:
                with open(tmp_path, 'w') as f:
                    json.dump(wm_data, f)
                os.replace(tmp_path, wm_path)  # atomic on POSIX and Windows
            except Exception as e:
                logger.warning(f"[ARN] working memory save failed: {e}")
                try:
                    tmp_path.unlink(missing_ok=True)
                except:
                    pass
        except Exception as e:
            logger.debug("working memory save skipped: %s", e)
    
    def _load_working_memory(self):
        """Load working memory from disk and restore episode references."""
        try:
            wm_path = Path(self.storage.data_dir) / "working_memory.json"
            if not wm_path.exists():
                return
            wm_data = json.loads(wm_path.read_text())
            recent_ids = wm_data.get("recent_input_ids", [])
            for eid in recent_ids:
                if eid < 0:
                    continue
                ep = self.storage.get_episode(eid)
                if ep is not None:
                    # Reconstruct vector from storage
                    vectors, vec_ids = self.storage.get_episode_vectors([eid])
                    vec = vectors[0] if len(vectors) > 0 else None
                    if vec is not None:
                        self.working_memory.add(
                            content=ep['content'],
                            vector=vec,
                            priority=ep.get('importance', 0.5),
                            source_id=eid
                        )
        except Exception:
            pass  # Working memory restoration is best-effort
    
    def close(self):
        """Clean shutdown."""
        self._save_state()
        self._save_working_memory()
        self.storage.close()
        logger.info("ARN v9 shut down cleanly.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
