"""
ARN: Adaptive Reasoning Network - Brain-Inspired Cognitive Architecture
=============================================================================

Production implementation combining:
1. Complementary Learning Systems (fast episodic + slow semantic)
2. Clustering-based consolidation with contradiction detection
3. Calibrated prediction error (running statistics baseline)
4. Domain-specialized columns (repurposed from Thousand Brains)
5. Semantic embeddings (all-MiniLM-L6-v2 by default; switchable tiers)
6. Persistent storage (SQLite + memmap)
7. Working memory with decay and rehearsal

Key design decisions:
- Cortical columns repurposed as DOMAIN PROCESSORS, not spatial grid cells.
  Text agents don't have spatial reference frames. Each column specializes
  in a domain (code, conversation, facts, etc.) providing parallel expertise.
- Consolidation uses similarity-based clustering, not first-word matching.
- Prediction error is calibrated against running baselines per domain.
- All vectors are normalized and stored per embedding tier to prevent dimension conflicts.
"""

import numpy as np
import time
import json
import logging
import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .embeddings import EmbeddingEngine, EMBEDDING_DIM
from ..storage.persistence import StorageEngine

logger = logging.getLogger("arn.core")


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
    prototype: np.ndarray  # embedding-dim domain centroid
    
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
        Online update of running mean/variance using Welford's algorithm.

        `error_var` is stored as the population variance so the state stays
        compact and backwards-compatible. The update reconstructs the running
        M2 value from the previous variance/count, applies Welford, then saves
        the new variance.
        """
        previous_count = self.error_count
        previous_mean = self.error_mean
        previous_m2 = self.error_var * max(previous_count, 0)

        self.error_count += 1
        delta = error - previous_mean
        self.error_mean = previous_mean + delta / self.error_count
        delta2 = error - self.error_mean
        new_m2 = previous_m2 + delta * delta2
        self.error_var = max(new_m2 / self.error_count, 1e-9)
    
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
        # similarity_threshold controls when two episodes are merged into the
        # same cluster during consolidation. It must be calibrated per embedding
        # model — each tier has different score distributions.
        # Pass the value from EmbeddingEngine._config['consolidation_similarity_threshold']
        # rather than using this hardcoded default (0.55 is tuned for nano/MiniLM only).
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
# MAIN ARN CLASS
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
        # Read per-model consolidation threshold from embedder config.
        # Falls back to 0.55 (MiniLM default) if key is missing for compatibility
        # with any third-partyly constructed EmbeddingEngine instances.
        consolidation_sim_threshold = self.embedder._config.get(
            'consolidation_similarity_threshold', 0.55
        )
        self.consolidation_engine = ConsolidationEngine(
            similarity_threshold=consolidation_sim_threshold,
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
        
        logger.info(f"ARN initialized. Episodes: {self.storage.count_episodes()}, "
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
    
    def perceive(self, content: str, importance: float = 0.5,
                 context: dict = None, source: str = 'user') -> dict:
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
            importance = min(1.0, importance + 0.2)
        
        # Step 4: Store episodic memory
        episode_id = self.storage.store_episode(
            content=content,
            vector=feature_vector,
            context=context or {},
            importance=importance,
            prediction_error=prediction_error,
            source=source
        )
        
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
        
        return {
            'episode_id': episode_id,
            'prediction_error': prediction_error,
            'domain_signals': domain_signals,
            'best_domain': best_domain.domain.value if best_domain else None,
            'importance': importance,
            'surprise_count': surprise_count,
            'consolidation_triggered': consolidation_triggered,
        }
    
    def recall(self, query: str, top_k: int = 5,
               include_semantic: bool = True,
               include_episodic: bool = True) -> List[dict]:
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
        
        # Search ALL episodic memory (consolidated + unconsolidated)
        if include_episodic:
            # Get ALL episodes (not just unconsolidated)
            all_episodes = self.storage.get_all_episodes(consolidated=None)
            if all_episodes:
                ep_ids_list = [ep['id'] for ep in all_episodes]
                ep_vectors, ep_ids = self.storage.get_episode_vectors(ep_ids_list)
                
                if len(ep_vectors) > 0:
                    similarities = ep_vectors @ query_vector
                    id_to_ep = {ep['id']: ep for ep in all_episodes}
                    
                    # First pass: compute raw similarity scores
                    sim_scored = []
                    for eid, sim in zip(ep_ids, similarities):
                        ep = id_to_ep.get(eid)
                        if ep:
                            sim_scored.append((eid, ep, float(sim)))
                    
                    # Contradiction detection: find groups of high-similarity episodes
                    # that may represent the same fact stated multiple times.
                    # Within such groups, recency should dominate.
                    sim_threshold = 0.75  # Episodes above this similarity to each other
                    
                    for eid, ep, sim in sim_scored:
                        # Recency: log-scaled so newer facts clearly win within a group
                        age_seconds = max(1.0, time.time() - ep['created_at'])
                        # Recency score: 1.0 for age=1s, 0.5 for age=1hr, 0.25 for age=1day
                        recency_boost = 1.0 / (1.0 + np.log1p(age_seconds / 3600.0))
                        
                        # Check if this episode is near-duplicate with any NEWER episode
                        # If so, penalize this one (it's likely an outdated version of the same fact)
                        is_superseded = False
                        for other_eid, other_ep, other_sim in sim_scored:
                            if other_eid == eid:
                                continue
                            if other_ep['created_at'] <= ep['created_at']:
                                continue  # other is older, not newer
                            # Check content similarity between these two episodes
                            ep_idx = ep_ids.index(eid)
                            other_idx = ep_ids.index(other_eid)
                            content_sim = float(np.dot(ep_vectors[ep_idx], ep_vectors[other_idx]))
                            if content_sim > sim_threshold:
                                is_superseded = True
                                break
                        
                        supersession_penalty = 0.2 if is_superseded else 0.0

                        # Additional penalty if this episode has been explicitly
                        # superseded by a newer fact (temporal tracking).
                        if ep.get('superseded_by') is not None:
                            supersession_penalty = max(supersession_penalty, 0.35)
                        
                        # Score: similarity dominates, importance and recency contribute,
                        # supersession penalizes outdated duplicates
                        score = (0.70 * sim + 
                                 0.15 * recency_boost + 
                                 0.15 * ep['importance'] -
                                 supersession_penalty)
                        
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
        
        # Tag each result with confidence level using PER-TIER calibrated thresholds.
        # Different embedding models have different similarity score distributions:
        # - MiniLM produces widely-spread scores (0.1 unrelated, 0.5 related)
        # - bge/e5 compress scores higher (0.35 unrelated, 0.65 related)
        # Hardcoded thresholds don't generalize; use the model's calibrated values.
        low_thresh = self.embedder._config.get('low_conf_threshold', 0.30)
        high_thresh = self.embedder._config.get('high_conf_threshold', 0.50)
        
        for r in top_results:
            if r['similarity'] >= high_thresh:
                r['confidence_tier'] = 'high'
            elif r['similarity'] >= low_thresh:
                r['confidence_tier'] = 'medium'
            else:
                r['confidence_tier'] = 'low'  # Treat as "probably don't know"
        
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
            
            # Restore column stats
            col_data = data.get('columns', [])
            for cd in col_data:
                for col in self.columns:
                    if col.domain.value == cd.get('domain'):
                        col.error_mean = cd.get('error_mean', 0.0)
                        col.error_var = cd.get('error_var', 1.0)
                        col.error_count = cd.get('error_count', 0)
                        col.expertise = cd.get('expertise', 0.0)
                        col.sample_count = cd.get('sample_count', 0)
    
    def _save_state(self):
        """Persist current state."""
        state = {
            'total_experiences': self.total_experiences,
            'consolidation_count': self.consolidation_count,
            'columns': [c.to_dict() for c in self.columns],
        }
        self.storage.set_state('arn_state', json.dumps(state))
    
    def close(self):
        """Clean shutdown."""
        self._save_state()
        self.storage.close()
        logger.info("ARN shut down cleanly.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
