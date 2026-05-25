"""
ARN v9: Adaptive Reasoning Network - Brain-Inspired Cognitive Architecture
=============================================================================

Production implementation combining:
1. Complementary Learning Systems (fast episodic + slow semantic)
2. Clustering-based consolidation with contradiction detection
3. Calibrated prediction error (running statistics baseline)
4. Domain-specialized columns (repurposed from Thousand Brains)
5. Semantic embeddings (all-MiniLM-L6-v2, 384-dim)
6. Persistent storage (SQLite + memmap)
7. Working memory with decay and rehearsal

Key design decisions:
- Cortical columns repurposed as DOMAIN PROCESSORS, not spatial grid cells.
  Text agents don't have spatial reference frames. Each column specializes
  in a domain (code, conversation, facts, etc.) providing parallel expertise.
- Consolidation uses similarity-based clustering, not first-word matching.
- Prediction error is calibrated against running baselines per domain.
- All vectors are 384-dim normalized (from sentence-transformers).
"""

import numpy as np
import time
import json
import os
import tempfile
import logging
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .embeddings import EmbeddingEngine, EMBEDDING_DIM
from ..storage.persistence import StorageEngine
from .contradictions import ContradictionDetector

logger = logging.getLogger("arn.core")

AUTO_LINK_SIMILARITY_THRESHOLD = 0.38


# =========================================================
# DOMAIN COLUMNS (Repurposed Cortical Columns)
# =========================================================

class DomainType(Enum):
    """Domain specializations for columns."""
    CODE = "code"
    CONVERSATION = "conversation"
    FACTS = "facts"
    PROCEDURES = "procedures"
    PREFERENCES = "preferences"
    TEMPORAL = "temporal"
    ERRORS = "errors"
    GENERAL = "general"


@dataclass
class DomainColumn:
    """
    A domain-specialized processing column.
    
    Instead of spatial reference frames (useless for text agents),
    each column maintains:
    - A domain prototype (what this domain "looks like")
    - Running prediction statistics (for calibrated surprise)
    - Domain-specific attention weight
    - Expertise confidence
    
    This is the Thousand Brains insight applied correctly to text:
    parallel domain-specific processors that vote on classification
    and flag anomalies within their expertise.
    """
    domain: DomainType
    prototype: np.ndarray  # 384-dim domain centroid
    
    # Running statistics for prediction error calibration
    error_mean: float = 0.0
    error_var: float = 1.0
    error_count: int = 0
    
    # Domain expertise
    attention: float = 1.0
    expertise: float = 0.0  # How much this column has learned
    sample_count: int = 0
    
    def update_error_stats(self, error: float):
        """
        Online update of running mean/variance (Welford's algorithm).
        This is crucial for calibrated prediction error — we need to
        know what's "normal" surprise for this domain.
        """
        self.error_count += 1
        delta = error - self.error_mean
        self.error_mean += delta / self.error_count
        delta2 = error - self.error_mean
        self.error_var += (delta * delta2 - self.error_var) / self.error_count
    
    def is_surprising(self, error: float, threshold_sigma: float = 2.0) -> bool:
        """
        Is this prediction error genuinely surprising for this domain?
        Uses calibrated threshold: error > mean + threshold * std.
        """
        if self.error_count < 10:
            return error > 0.5  # Not enough data; use fixed threshold
        std = max(np.sqrt(self.error_var), 1e-6)
        return error > self.error_mean + threshold_sigma * std
    
    def compute_relevance(self, feature_vector: np.ndarray) -> float:
        """How relevant is this input to this domain?"""
        return float(np.dot(self.prototype, feature_vector))
    
    def update_prototype(self, feature_vector: np.ndarray, learning_rate: float = 0.01):
        """Slowly update domain prototype toward new evidence."""
        self.prototype = (1 - learning_rate) * self.prototype + learning_rate * feature_vector
        norm = np.linalg.norm(self.prototype)
        if norm > 0:
            self.prototype /= norm
        self.sample_count += 1
        self.expertise = min(1.0, self.sample_count / 100.0)
    
    def to_dict(self) -> dict:
        return {
            'domain': self.domain.value,
            'prototype': self.prototype.tolist() if self.prototype is not None else None,
            'error_mean': self.error_mean,
            'error_var': self.error_var,
            'error_count': self.error_count,
            'attention': self.attention,
            'expertise': self.expertise,
            'sample_count': self.sample_count,
        }


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
                 embedding_tier: Optional[str] = None,
                 episodic_capacity: int = 4096,
                 semantic_capacity: int = 2048,
                 consolidation_threshold: int = 256,
                 auto_consolidate: bool = True):
        
        # Core components
        self.embedder = EmbeddingEngine(use_model=use_embeddings, tier=embedding_tier)
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
        
        # Domain columns
        self.columns = self._init_columns()
        
        # Configuration
        self.consolidation_threshold = consolidation_threshold
        self.auto_consolidate = auto_consolidate
        
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
    
    def _init_columns(self) -> List[DomainColumn]:
        """Initialize domain-specialized columns."""
        columns = []
        
        # Define domain seed phrases for initial prototypes
        domain_seeds = {
            DomainType.CODE: "programming code function variable algorithm software development",
            DomainType.CONVERSATION: "hello thanks question answer discuss chat talk conversation",
            DomainType.FACTS: "fact definition information data knowledge truth reference",
            DomainType.PROCEDURES: "step process procedure workflow instruction guide how to",
            DomainType.PREFERENCES: "prefer like want favorite choice setting option",
            DomainType.TEMPORAL: "today yesterday tomorrow schedule date time event meeting",
            DomainType.ERRORS: "error bug fix problem issue crash failure wrong broken",
            DomainType.GENERAL: "general topic subject matter content information",
        }
        
        for domain, seed in domain_seeds.items():
            prototype = self.embedder.encode(seed, mode='passage')
            columns.append(DomainColumn(
                domain=domain,
                prototype=prototype
            ))
        
        return columns
    
    def _restore_column_prototypes(self, col_data: list):
        """Restore column prototypes from saved state if available.

        Skips any prototype whose dimension doesn't match the current
        embedder dimension — this prevents a ValueError when the embedding
        tier has been upgraded (e.g. nano 384-dim → base 768-dim) and the
        stale saved prototypes are still 384-dim.  In that case the freshly
        generated prototypes from _init_columns() are kept instead.
        """
        expected_dim = self.embedder.embedding_dim
        for cd in col_data:
            for col in self.columns:
                if col.domain.value == cd.get('domain'):
                    proto = cd.get('prototype')
                    if proto is not None:
                        arr = np.array(proto, dtype=np.float32)
                        if arr.shape[0] != expected_dim:
                            logger.warning(
                                f"[ARN] Skipping saved prototype for domain "
                                f"'{col.domain.value}': dim={arr.shape[0]} "
                                f"!= expected {expected_dim} "
                                f"(embedding tier was upgraded). "
                                f"Fresh prototype will be used."
                            )
                            continue
                        col.prototype = arr
                        # Re-normalize in case of drift
                        norm = np.linalg.norm(col.prototype)
                        if norm > 0:
                            col.prototype /= norm
    
    def perceive(self, content: str, importance: float = 0.5,
                 context: dict = None, source: str = 'user',
                 memory_type: str = 'episode') -> dict:
        """
        Process new input — the main "learning" entry point.
        
        Steps:
        1. Encode to semantic vector
        2. Compute prediction error against working memory context
        3. Route through domain columns for calibrated surprise
        4. Store as episodic memory
        5. Update working memory
        6. Maybe trigger consolidation
        
        Returns dict with episode_id, prediction_error, domain_signals, etc.
        """
        # Decay working memory based on elapsed time
        now = time.time()
        elapsed = now - self._last_decay_time
        self.working_memory.decay(elapsed_seconds=elapsed)
        self._last_decay_time = now
        
        # Step 1: Encode as a passage (stored content)
        feature_vector = self.embedder.encode(content, mode='passage')
        
        # Step 2: Prediction error against working memory context
        wm_context = self.working_memory.get_context_vector()
        if wm_context is not None:
            prediction_error = 1.0 - float(np.dot(feature_vector, wm_context))
        else:
            prediction_error = 1.0  # No context = maximum surprise
        
        # Step 3: Domain column processing
        domain_signals = []
        best_domain = None
        best_relevance = -1.0
        
        for column in self.columns:
            relevance = column.compute_relevance(feature_vector)
            is_surprising = column.is_surprising(prediction_error)
            
            domain_signals.append({
                'domain': column.domain.value,
                'relevance': relevance,
                'surprising': is_surprising,
                'expertise': column.expertise,
            })
            
            if relevance > best_relevance:
                best_relevance = relevance
                best_domain = column
        
        # Update best matching column
        if best_domain is not None:
            best_domain.update_prototype(feature_vector, learning_rate=0.01)
            best_domain.update_error_stats(prediction_error)
        
        # Boost importance if content is surprising across multiple domains
        surprise_count = sum(1 for s in domain_signals if s['surprising'])
        if surprise_count >= 3:
            importance = min(0.95, importance + 0.2)
        
        # Step 4: Store episodic memory
        # Check for contradictions with existing memories
        contradiction_hits = []
        try:
            detector = ContradictionDetector(self.storage, self.embedder)
            contradiction_hits = detector.check(content, top_k_candidates=15)
        except Exception:
            pass  # Contradiction detection is best-effort

        episode_id = self.storage.store_episode(
            content=content,
            vector=feature_vector,
            context=context or {},
            importance=importance,
            prediction_error=prediction_error,
            source=source,
            memory_type=memory_type,
        )

        # If contradictions found, supersede the old episodes (never supersede api-seeded facts)
        if contradiction_hits:
            try:
                for hit in contradiction_hits:
                    old_ep = self.storage.get_episode(hit['old_episode_id'])
                    if old_ep and old_ep.get('source') == 'api':
                        continue
                    detector.supersede_old(hit['old_episode_id'], episode_id)
            except Exception:
                pass
        
        # Auto-link: connect new memory to top-3 similar existing memories
        try:
            all_episodes = self.storage.get_all_episodes(consolidated=None)
            existing = [ep for ep in all_episodes if ep['id'] != episode_id]
            if existing:
                ep_ids_list = [ep['id'] for ep in existing]
                ep_vectors, ep_ids = self.storage.get_episode_vectors(ep_ids_list)
                if len(ep_vectors) > 0:
                    similarities = ep_vectors @ feature_vector
                    scored = list(zip(ep_ids, similarities))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for other_id, sim in scored[:3]:
                        if float(sim) >= AUTO_LINK_SIMILARITY_THRESHOLD:
                            self.storage.create_link(
                                episode_id, other_id, "relates_to", confidence=float(sim)
                            )
        except Exception:
            pass  # Auto-linking is best-effort
        
        # Step 5: Update working memory
        self.working_memory.add(
            content=content,
            vector=feature_vector,
            priority=importance,
            source_id=episode_id
        )
        
        self.total_experiences += 1
        
        # Step 6: Auto-consolidation check
        unconsolidated = self.storage.count_episodes(consolidated=False)
        consolidation_triggered = False
        if self.auto_consolidate and unconsolidated >= self.consolidation_threshold:
            self.consolidate()
            consolidation_triggered = True
        
        # Save working memory to disk
        self._save_working_memory()
        
        return {
            'episode_id': episode_id,
            'prediction_error': prediction_error,
            'domain_signals': domain_signals,
            'best_domain': best_domain.domain.value if best_domain else None,
            'importance': importance,
            'surprise_count': surprise_count,
            'consolidation_triggered': consolidation_triggered,
            'contradictions_found': len(contradiction_hits),
            'memory_type': memory_type,
        }
    
    def recall(self, query: str, top_k: int = 5,
               include_semantic: bool = True,
               include_episodic: bool = True,
               memory_type: Optional[str] = None) -> List[dict]:
        """
        Retrieve relevant memories for a query.
        
        Searches ALL episodic memory (both consolidated and unconsolidated)
        and semantic memory, merges results, and returns top-k by relevance.
        
        Consolidated episodes are still searchable — consolidation marks them
        as processed for semantic extraction but does NOT remove them from
        recall. They just get a lower recency boost.
        """
        query_vector = self.embedder.encode(query, mode='query')
        results = []
        
        # Search episodic memory (consolidated + unconsolidated), with optional type filter
        if include_episodic:
            all_episodes = self.storage.get_all_episodes(
                consolidated=None, memory_type=memory_type
            )
            if all_episodes:
                # Filter out superseded episodes (they have a superseded_by pointer)
                active_episodes = [
                    ep for ep in all_episodes
                    if ep.get('superseded_by') is None and ep.get('invalidated_at') is None
                ]
                # If filtering removed everything, fall back to all (don't hide data)
                if not active_episodes:
                    active_episodes = all_episodes

                ep_ids_list = [ep['id'] for ep in active_episodes]
                ep_vectors, ep_ids = self.storage.get_episode_vectors(ep_ids_list)

                # Drop any stored vectors whose dimension doesn't match the current query.
                # This can happen when the embedding model changes between server restarts.
                if len(ep_vectors) > 0 and ep_vectors.shape[1] != query_vector.shape[0]:
                    expected_dim = query_vector.shape[0]
                    mask = [i for i, v in enumerate(ep_vectors) if v.shape[0] == expected_dim]
                    ep_vectors = ep_vectors[mask] if mask else ep_vectors[:0]
                    ep_ids = [ep_ids[i] for i in mask]

                if len(ep_vectors) > 0:
                    similarities = ep_vectors @ query_vector
                    id_to_ep = {ep['id']: ep for ep in active_episodes}
                    
                    # First pass: compute raw similarity scores
                    sim_scored = []
                    for eid, sim in zip(ep_ids, similarities):
                        ep = id_to_ep.get(eid)
                        if ep:
                            sim_scored.append((eid, ep, float(sim)))
                    
                    # Pre-compute content similarity matrix for supersession detection
                    # Only compute for top candidates to avoid O(n²) on large corpora
                    n_check = min(len(sim_scored), top_k * 4)
                    top_sim_scored = sorted(sim_scored, key=lambda x: x[2], reverse=True)[:n_check]
                    
                    # Build a set of superseded episode IDs efficiently
                    superseded_ids = set()
                    top_ep_ids = [eid for eid, _, _ in top_sim_scored]
                    top_ep_vecs = np.array([ep_vectors[ep_ids.index(eid)] for eid in top_ep_ids])
                    if len(top_ep_vecs) > 1:
                        # Pairwise similarities among top candidates
                        pair_sims = top_ep_vecs @ top_ep_vecs.T
                        sim_threshold = 0.75
                        for i in range(len(top_ep_ids)):
                            for j in range(i + 1, len(top_ep_ids)):
                                if pair_sims[i, j] > sim_threshold:
                                    ep_i = id_to_ep[top_ep_ids[i]]
                                    ep_j = id_to_ep[top_ep_ids[j]]
                                    # The older one is superseded
                                    if ep_i['created_at'] < ep_j['created_at']:
                                        superseded_ids.add(top_ep_ids[i])
                                    else:
                                        superseded_ids.add(top_ep_ids[j])
                    
                    for eid, ep, sim in top_sim_scored:
                        # Recency: exponential decay with 7-day half-life for creation
                        age_seconds = max(1.0, time.time() - ep['created_at'])
                        creation_recency = 0.5 ** (age_seconds / (86400.0 * 7))
                        
                        # Access recency: 1-day half-life
                        last_access = ep.get('last_accessed') or ep['created_at']
                        access_age = max(1.0, time.time() - last_access)
                        access_recency = 0.5 ** (access_age / 86400.0)
                        
                        # Access frequency bonus (logarithmic, capped)
                        freq_bonus = min(1.0, np.log1p(ep['access_count']) / 3.0)
                        
                        # Composite recency: creation + access blended
                        recency_blend = 0.4 * creation_recency + 0.6 * access_recency + 0.1 * freq_bonus
                        recency_blend = min(1.0, recency_blend)
                        
                        # Supersession penalty
                        supersession_penalty = 0.25 if eid in superseded_ids else 0.0
                        
                        # Surprise bonus: high-error episodes are more informative
                        surprise_bonus = 0.05 * ep['prediction_error']
                        
                        # Hybrid score: similarity + recency + importance + surprise
                        # Weights adjusted to sum to 1.0 at maximum:
                        #   sim=0.58, recency=0.13, importance=0.19,
                        #   non-superseded bonus=0.05, surprise=0.05
                        score = (
                            0.58 * sim +
                            0.13 * recency_blend +
                            0.19 * ep['importance'] +
                            surprise_bonus -
                            supersession_penalty
                        )
                        
                        # Boost current (non-superseded) episodes
                        if ep.get('superseded_by') is None and ep.get('invalidated_at') is None:
                            score += 0.05
                        
                        results.append({
                            'type': 'episodic',
                            'id': eid,
                            'content': ep['content'],
                            'score': score,
                            'similarity': float(sim),
                            'importance': ep['importance'],
                            'created_at': ep['created_at'],
                            'access_count': ep['access_count'],
                            'context': ep.get('context', {}),
                            'memory_type': ep.get('memory_type', 'episode'),
                            'source': ep.get('source', 'unknown'),
                        })
                        
                        # Update access count
                        self.storage.update_episode_access(eid)
        
        # Search semantic memory
        if include_semantic:
            sem_vectors, sem_ids = self.storage.get_semantic_vectors()
            if len(sem_vectors) > 0:
                similarities = sem_vectors @ query_vector
                
                semantics = self.storage.get_all_semantics()
                id_to_sem = {s['id']: s for s in semantics}
                
                for sid, sim in zip(sem_ids, similarities):
                    sem = id_to_sem.get(sid)
                    if sem:
                        # Confidence-weighted score  
                        score = float(sim) * (0.5 + 0.5 * sem['confidence'])
                        
                        # Use representative_content if available, else label
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
        
        # Sort by score and return top-k
        results.sort(key=lambda r: r['score'], reverse=True)
        top_results = results[:top_k]

        # Pull graph neighbors into the recall window. If a highly-ranked
        # episodic hit is linked to another candidate, that neighbor is useful
        # context even when its raw embedding score is just below top-k.
        if top_results:
            result_by_id = {
                r['id']: r for r in results
                if r['type'] == 'episodic'
            }
            top_ids = {r['id'] for r in top_results if r['type'] == 'episodic'}
            linked_results = []
            for r in list(top_results):
                if r['type'] != 'episodic':
                    continue
                links = self.storage.get_links_for_episode(r['id'])
                for link in links:
                    other_id = link['from_episode_id'] if link['to_episode_id'] == r['id'] else link['to_episode_id']
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
        
        # Tag each result with confidence level using calibrated thresholds.
        # The calibrator learns the score distribution of the currently-loaded
        # model, so thresholds adapt automatically when switching tiers.
        for r in top_results:
            r['confidence_tier'] = self.embedder.confidence_tier(r['similarity'])
            r['calibrated_confidence'] = round(
                self.embedder.calibrate_similarity(r['similarity']), 3
            )
        
        return top_results
    
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
            'columns': [c.to_dict() for c in self.columns],
        }
    
    def _load_state(self):
        """Load persisted state."""
        state = self.storage.get_state('arn_state')
        if state:
            data = json.loads(state)
            self.total_experiences = data.get('total_experiences', 0)
            self.consolidation_count = data.get('consolidation_count', 0)
            
            # Restore column stats and prototypes
            col_data = data.get('columns', [])
            for cd in col_data:
                for col in self.columns:
                    if col.domain.value == cd.get('domain'):
                        col.error_mean = cd.get('error_mean', 0.0)
                        col.error_var = cd.get('error_var', 1.0)
                        col.error_count = cd.get('error_count', 0)
                        col.expertise = cd.get('expertise', 0.0)
                        col.sample_count = cd.get('sample_count', 0)
            self._restore_column_prototypes(col_data)
    
    def _save_state(self):
        """Persist current state."""
        state = {
            'total_experiences': self.total_experiences,
            'consolidation_count': self.consolidation_count,
            'columns': [c.to_dict() for c in self.columns],
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
