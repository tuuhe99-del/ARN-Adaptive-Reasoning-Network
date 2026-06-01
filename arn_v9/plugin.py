"""
ARN v9 OpenClaw Plugin Interface
==================================
Provides the standardized API that OpenClaw agents use to interact
with ARN v9's cognitive memory system.

This replaces AGENTS.md and MEMORY.md with a live, brain-inspired
memory system that learns, consolidates, and recalls adaptively.

Usage in OpenClaw:
    from arn_v9.plugin import ARNPlugin
    
    plugin = ARNPlugin(agent_id="agent_001", data_root="./memory")
    
    # Agent stores experience
    plugin.store("User prefers Python", importance=0.8, tags=["preference"])
    
    # Agent recalls relevant context
    context = plugin.recall("What language does the user like?", top_k=5)
    
    # Background maintenance (call periodically)
    plugin.maintain()
    
    # Clean shutdown
    plugin.shutdown()
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional, Any

from .core.cognitive import ARNv9

logger = logging.getLogger("arn.plugin")


class ARNPlugin:
    """
    OpenClaw-compatible memory plugin powered by ARN v9.
    
    Each agent gets its own isolated memory namespace while sharing
    the embedding model for efficiency.
    """
    
    def __init__(self, agent_id: str = "default",
                 data_root: str = "./arn_data",
                 use_embeddings: bool = True,
                 episodic_capacity: int = 4096,
                 semantic_capacity: int = 2048,
                 embedding_fn=None):

        self.agent_id = agent_id
        self.use_embeddings = use_embeddings
        data_dir = os.path.join(data_root, agent_id)

        self._arn = ARNv9(
            data_dir=data_dir,
            use_embeddings=use_embeddings,
            embedding_fn=embedding_fn,
            episodic_capacity=episodic_capacity,
            semantic_capacity=semantic_capacity,
        )
        
        # Check if embeddings loaded. Explicit lexical fallback is acceptable
        # for offline/local bridge use; accidental model failure is not.
        if self._arn.embedder.is_degraded and use_embeddings:
            logger.critical(
                f"ARN plugin '{agent_id}' is DEGRADED: no embedding model loaded. "
                "Memory recall will use lower-quality lexical fallback. "
                "Fix: pip install sentence-transformers"
            )
            import warnings
            warnings.warn(
                "ARN is running without semantic embeddings. "
                "Memory operations will use lower-quality lexical fallback. "
                "Install sentence-transformers to fix this.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self._arn.embedder.is_degraded:
            logger.info(
                f"ARN plugin '{agent_id}' using lexical fallback embeddings "
                "because use_embeddings=False"
            )
        
        self._last_maintain = time.time()
        
        logger.info(f"ARN plugin initialized for agent '{agent_id}'")
    
    # ===========================================
    # PRIMARY API (what agents call)
    # ===========================================
    
    def store(self, content: str, importance: float = 0.5,
              tags: List[str] = None, source: str = "agent",
              context: dict = None,
              memory_type: str = "episode") -> dict:
        """Store a new experience/fact/observation."""
        ctx = context or {}
        if tags:
            ctx['tags'] = tags
        ctx['source'] = source

        result = self._arn.perceive(
            content=content,
            importance=importance,
            context=ctx,
            source=source,
            memory_type=memory_type,
        )

        return {
            'stored': True,
            'episode_id': result['episode_id'],
            'prediction_error': result['prediction_error'],
        }

    def recall(self, query: str, top_k: int = 5,
               memory_types: List[str] = None,
               memory_type: Optional[str] = None) -> List[dict]:
        """Recall relevant memories for a query."""
        include_ep = True
        include_sem = True

        if memory_types:
            include_ep = "episodic" in memory_types
            include_sem = "semantic" in memory_types

        results = self._arn.recall(
            query=query,
            top_k=top_k,
            include_episodic=include_ep,
            include_semantic=include_sem,
            memory_type=memory_type,
        )

        simplified = []
        for r in results:
            entry = {
                'id': r.get('id'),
                'content': r['content'],
                'score': round(r['score'], 4),
                'type': r['type'],
                'similarity': round(r['similarity'], 4),
                'confidence_tier': r.get('confidence_tier', 'medium'),
                'calibrated_confidence': r.get('calibrated_confidence', 0.5),
                'memory_type': r.get('memory_type', 'episode'),
                'source': r.get('source', 'unknown'),
            }

            if r['type'] == 'episodic':
                entry['importance'] = r.get('importance', 0)
                entry['created_at'] = r.get('created_at', time.time())
                entry['age_hours'] = round(
                    (time.time() - r.get('created_at', time.time())) / 3600, 1
                )
            elif r['type'] == 'semantic':
                entry['confidence'] = r.get('confidence', 0)
                entry['evidence_count'] = r.get('evidence_count', 0)

            simplified.append(entry)

        return simplified
    
    @staticmethod
    def _format_time_ago(age_hours: float) -> str:
        """Convert age in hours to a human-readable relative time."""
        if age_hours is None:
            return ""
        mins = round(age_hours * 60)
        hrs = round(age_hours)
        days = round(age_hours / 24)

        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins}m ago"
        if hrs < 24:
            return f"{hrs}h ago"
        if days == 1:
            return "yesterday"
        if days < 7:
            return f"{days}d ago"
        if days < 30:
            weeks = round(days / 7)
            return f"{weeks}w ago"
        if days < 365:
            months = round(days / 30)
            return f"{months}mo ago"
        years = round(days / 365)
        return f"{years}y ago"

    def get_context_window(self, query: str = None, max_tokens: int = 2000) -> str:
        """
        Get a formatted context string suitable for injection into
        an LLM prompt. This is the primary way agents use ARN.
        
        Args:
            query: Optional query to focus the context retrieval
            max_tokens: Approximate token budget (~4 chars per token)
        
        Returns:
            Formatted string of relevant memories
        """
        char_budget = max_tokens * 4
        
        # Get working memory contents
        wm_items = self._arn.working_memory.get_active()
        
        # Get relevant long-term memories
        if query:
            lt_results = self.recall(query, top_k=10)
        else:
            # No query — return most recent/important
            lt_results = self.recall("recent important information", top_k=10)
        
        # Build context string
        parts = []
        current_chars = 0
        
        # Working memory (most recent active context)
        if wm_items:
            parts.append("## Active Context (Working Memory)")
            for slot in wm_items[:5]:
                line = f"- {slot.content}"
                if current_chars + len(line) > char_budget:
                    break
                parts.append(line)
                current_chars += len(line)
        
        # Long-term memories
        if lt_results:
            parts.append("\n## Relevant Memories")
            for r in lt_results:
                prefix = "📌" if r['type'] == 'semantic' else "💭"
                score_str = f"[{r['score']:.2f}]"
                
                # Time context — when did this happen?
                when = self._format_time_ago(r.get('age_hours'))
                time_tag = f"[{when}]" if when else ""
                
                # Source attribution for conversation history feel
                source = r.get('source', 'unknown')
                if source == 'user':
                    attrib = "User said:"
                elif source == 'me' or source == 'agent':
                    attrib = "I said:"
                elif source.startswith('tool:'):
                    attrib = f"I used {source[5:]}:"
                elif source == 'tool_result':
                    attrib = "Tool returned:"
                elif source == 'compaction':
                    attrib = "Turn summary:"
                else:
                    attrib = ""
                
                line = f"{prefix} {time_tag} {score_str} {attrib} {r['content']}" if attrib else f"{prefix} {time_tag} {score_str} {r['content']}"
                line = " ".join(line.split())  # normalize whitespace
                
                if current_chars + len(line) > char_budget:
                    break
                parts.append(line)
                current_chars += len(line)
        
        return "\n".join(parts)
    
    def maintain(self):
        """
        Run maintenance tasks. Call this during idle periods.
        - Consolidation (episodic → semantic)
        - Working memory decay
        """
        # Consolidate if enough unconsolidated episodes
        stats = self._arn.consolidate()
        
        # Decay working memory
        elapsed = time.time() - self._last_maintain
        self._arn.working_memory.decay(elapsed_seconds=elapsed)
        self._last_maintain = time.time()
        
        return stats
    
    def get_stats(self) -> dict:
        """Get system statistics for monitoring."""
        stats = self._arn.get_stats()
        stats['agent_id'] = self.agent_id
        return stats
    
    def shutdown(self):
        """Clean shutdown — persist all state."""
        self._arn.close()
        logger.info(f"ARN plugin shut down for agent '{self.agent_id}'")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
