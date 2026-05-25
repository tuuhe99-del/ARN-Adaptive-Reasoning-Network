"""
ARN v9 Store-Time Contradiction Detector
==========================================
Lightweight, CPU-only contradiction detection for edge devices.

Detects when a newly-stored fact contradicts an existing memory.
Uses a hybrid approach:
  1. Claim extraction (regex + keyword patterns)
  2. Bi-encoder retrieval (existing embedding model) for candidates
  3. Lexical + temporal heuristics for verification
  4. Automatic supersession linking when a contradiction is found

No spaCy or cross-encoder required — works entirely with the existing
embedding model and regex patterns.  This keeps latency on Pi 5 under
~50 ms for a few-thousand-entry memory bank.

Usage:
    detector = ContradictionDetector(arn_storage, embedder)
    hits = detector.check(content="I now prefer Rust")
    if hits:
        detector.supersede_old(hits[0]['old_episode_id'], new_episode_id)
"""

import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("arn.contradictions")


# =========================================================
# TEMPORAL / STATE-CHANGE LEXICON
# =========================================================

_TEMPORAL_PAST = re.compile(
    r"\b(used to|previously|before|in the past|earlier|once|formerly|had been|was|were)\b",
    re.IGNORECASE,
)
_TEMPORAL_PRESENT = re.compile(
    r"\b(now|currently|lately|recently|today|these days|at present|is|are)\b",
    re.IGNORECASE,
)
_CHANGE_VERBS = re.compile(
    r"\b(switched|moved|changed|shifted|transitioned|adopted|quit|left|joined|"
    r"updated|replaced|migrated|converted|upgraded|downgraded|reverted)\b",
    re.IGNORECASE,
)
_NEGATION_WORDS = re.compile(
    r"\b(no longer|not|never|don't|doesn't|didn't|won't|wouldn't|can't|cannot|"
    r"none|nothing|nobody|nowhere|neither|nor|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)

# Common preference / identity relations we can detect
_PREFERENCE_PATTERNS = [
    re.compile(r"\b(prefer\w*|like\w*|love\w*|hate\w*|enjoy\w*|favorite)\b", re.I),
    re.compile(r"\b(use\w*|work\s+with|code\s+in|program\s+in|build\s+with)\b", re.I),
    re.compile(r"\b(name\s+is|am\s+called|go\s+by|call\s+me)\b", re.I),
    re.compile(r"\b(run\w*|use\w*|host\w*|deploy\w*).+?(on|with|via)\b", re.I),
]

# Relation normalization groups: verbs that express the same underlying relation
_RELATION_GROUPS = {
    "preference": {"prefer", "like", "love", "enjoy", "favorite", "hate", "dislike"},
    "usage": {"use", "work", "code", "program", "build", "run", "host", "deploy"},
    "identity": {"name", "am", "called", "go"},
    "state_change": {"switched", "moved", "changed", "shifted", "transitioned",
                     "adopted", "quit", "left", "joined", "updated", "replaced",
                     "migrated", "converted", "upgraded", "downgraded", "reverted"},
}

def _normalize_relation(relation: str) -> str:
    """Map a relation to its canonical group name."""
    rel_lower = relation.lower().strip()
    for group, members in _RELATION_GROUPS.items():
        if rel_lower in members:
            return group
    return rel_lower


@dataclass
class Claim:
    """A simplified factual claim extracted from text."""
    subject: str
    relation: str
    object: str
    temporal_tag: Optional[str] = None
    has_negation: bool = False

    def key(self) -> str:
        """Canonical key for matching claims (uses normalized relation)."""
        return f"{self.subject.lower().strip()}::{_normalize_relation(self.relation)}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "temporal_tag": self.temporal_tag,
            "has_negation": self.has_negation,
        }


class ClaimExtractor:
    """Extract structured claims from free-form text using lightweight patterns."""

    def __init__(self):
        # SVO extraction: "I prefer Python" -> (I, prefer, Python)
        self._svo_pattern = re.compile(
            r"\b(I|we|user|the\s+user|agent|they|he|she|it)\b"
            r"\s+([a-zA-Z\s]+?)\s+"
            r"\b(to\s+)?([A-Za-z0-9_#\+\-\.]+(?:\s+[A-Za-z0-9_#\+\-\.]+){0,4})\b",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> List[Claim]:
        """Extract all claims from text."""
        claims = []
        sentences = re.split(r"[.!?;]", text)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 4:
                continue

            # Temporal tagging
            temporal = None
            if _CHANGE_VERBS.search(sent):
                temporal = "CHANGE"
            elif _TEMPORAL_PAST.search(sent):
                temporal = "PAST"
            elif _TEMPORAL_PRESENT.search(sent):
                temporal = "PRESENT"

            has_neg = bool(_NEGATION_WORDS.search(sent))

            # Try SVO extraction
            found_svo = False
            for m in self._svo_pattern.finditer(sent):
                subj = m.group(1) or "user"
                rel = (m.group(2) or "states").strip().lower()
                obj = (m.group(4) or "").strip()
                if obj and len(obj) > 1:
                    claims.append(
                        Claim(
                            subject=subj,
                            relation=rel,
                            object=obj,
                            temporal_tag=temporal,
                            has_negation=has_neg,
                        )
                    )
                    found_svo = True

            # Fallback: if no SVO, try preference/identity patterns
            if not found_svo:
                for pat in _PREFERENCE_PATTERNS:
                    if pat.search(sent):
                        # Extract object as everything after the keyword
                        match = pat.search(sent)
                        if match:
                            after = sent[match.end() :].strip()
                            # Remove leading prepositions
                            after = re.sub(r"^(to|with|in|on|by|for|from)\s+", "", after, flags=re.I)
                            if after:
                                claims.append(
                                    Claim(
                                        subject="user",
                                        relation=match.group(1).lower(),
                                        object=after[:80],
                                        temporal_tag=temporal,
                                        has_negation=has_neg,
                                    )
                                )
                                break

        # Do NOT create fallback claims for unstructured text.
        # Unstructured / unparseable text cannot reliably be compared for
        # contradictions — treating it as a generic "user states X" claim
        # causes false positives where ANY two unstructured texts are flagged
        # as contradictory (same subject, same relation, different object).
        # This led to chain-supersession that destroyed nearly all memories.
        return claims


class ContradictionDetector:
    """
    Detect contradictions between a new memory and existing episodes.

    Pipeline:
        1. Extract claims from new text
        2. Retrieve candidate episodes via embedding similarity
        3. Extract claims from each candidate
        4. Score contradictions via claim-key match + object difference + temporal signals
    """

    def __init__(self, storage, embedder, similarity_threshold: float = 0.55):
        self.storage = storage
        self.embedder = embedder
        self.extractor = ClaimExtractor()
        self.similarity_threshold = similarity_threshold

    def check(
        self, content: str, top_k_candidates: int = 10
    ) -> List[Dict]:
        """
        Check if `content` contradicts any existing episode.

        Returns list of contradiction dicts, each with:
            - new_claim, old_claim, old_episode_id, contradiction_score,
              temporal_boost, similarity
        """
        new_claims = self.extractor.extract(content)
        if not new_claims:
            return []

        # Encode the new content for similarity search
        new_vec = self.embedder.encode(content, mode="passage")

        # Retrieve candidate episodes via embedding similarity
        candidates = self._retrieve_candidates(new_vec, top_k=top_k_candidates)
        if not candidates:
            return []

        hits = []
        for new_claim in new_claims:
            for candidate in candidates:
                old_claims = self.extractor.extract(candidate["content"])
                for old_claim in old_claims:
                    result = self._score_pair(new_claim, old_claim, candidate)
                    if result:
                        hits.append(result)

        # Deduplicate by old_episode_id, keep highest score
        seen = {}
        for h in hits:
            oid = h["old_episode_id"]
            if oid not in seen or h["contradiction_score"] > seen[oid]["contradiction_score"]:
                seen[oid] = h
        return list(seen.values())

    def _retrieve_candidates(self, query_vec: np.ndarray, top_k: int) -> List[dict]:
        """
        Find candidate episodes via brute-force cosine similarity.
        Uses a lower threshold than normal recall because contradictory
        statements are often semantically dissimilar (different objects).
        """
        all_eps = self.storage.get_all_episodes(consolidated=None)
        if not all_eps:
            return []

        # Exclude episodes that are already superseded, invalidated, or from protected sources
        active_eps = [
            ep for ep in all_eps
            if ep.get('superseded_by') is None
            and ep.get('invalidated_at') is None
            and ep.get('source') != 'api'
        ]
        if not active_eps:
            active_eps = all_eps

        ep_ids = [ep["id"] for ep in active_eps]
        ep_vectors, returned_ids = self.storage.get_episode_vectors(ep_ids)
        if len(ep_vectors) == 0:
            return []

        sims = ep_vectors @ query_vec
        # Use a low threshold for contradiction candidates — contradictory
        # facts often have different objects and thus lower raw similarity.
        min_sim = 0.15

        # Take top_k by similarity
        top_indices = np.argpartition(sims, -min(top_k, len(sims)))[-min(top_k, len(sims)):]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        id_to_ep = {ep["id"]: ep for ep in active_eps}
        candidates = []
        for idx in top_indices:
            eid = returned_ids[idx]
            ep = id_to_ep.get(eid)
            if ep and float(sims[idx]) >= min_sim:
                candidates.append(ep)

        # Fallback: if embedding retrieval finds nothing, scan recent episodes
        # This catches cases where contradictory statements are orthogonal in
        # embedding space (e.g. "I prefer Python" vs "I switched to Rust").
        if not candidates:
            # Return the 10 most recent active episodes as candidates
            recent = sorted(active_eps, key=lambda e: e.get('created_at', 0), reverse=True)[:10]
            candidates = recent

        return candidates

    def _score_pair(
        self, new_claim: Claim, old_claim: Claim, candidate: dict
    ) -> Optional[Dict]:
        """
        Score a new claim against an old claim.
        Returns contradiction dict if a contradiction is detected, else None.
        """
        # Must match on subject (same entity)
        if new_claim.subject.lower().strip() != old_claim.subject.lower().strip():
            return None

        # Relaxed relation matching: same normalized group OR one is a state_change
        new_rel_group = _normalize_relation(new_claim.relation)
        old_rel_group = _normalize_relation(old_claim.relation)
        same_relation_group = new_rel_group == old_rel_group
        is_state_change = new_rel_group == "state_change" or old_rel_group == "state_change"
        if not same_relation_group and not is_state_change:
            return None

        # If object is identical, it's not a contradiction
        if new_claim.object.lower().strip() == old_claim.object.lower().strip():
            return None

        # Base contradiction score: 0.55 for same-entity different-object
        # This ensures plain preference updates ("I like X" -> "I like Y")
        # are detected even without explicit temporal keywords.
        score = 0.55

        # Temporal boost: if one is past/change and other is present, boost
        temporal_boost = 0.0
        tags = {old_claim.temporal_tag, new_claim.temporal_tag}
        if tags == {"PAST", "PRESENT"} or "CHANGE" in tags:
            temporal_boost = 0.25
        # If new claim has a change verb but old doesn't, also boost
        if new_claim.temporal_tag == "CHANGE" and old_claim.temporal_tag != "CHANGE":
            temporal_boost = max(temporal_boost, 0.20)

        # State-change verbs are strong contradiction signals
        if is_state_change:
            temporal_boost = max(temporal_boost, 0.15)

        # Negation boost: if one is negated and the other isn't
        neg_boost = 0.0
        if new_claim.has_negation != old_claim.has_negation:
            neg_boost = 0.15

        final_score = min(1.0, score + temporal_boost + neg_boost)

        # Threshold: only flag if score >= 0.55 (tunable)
        if final_score < 0.55:
            return None

        return {
            "kind": "claim_conflict",
            "new_claim": new_claim.to_dict(),
            "old_claim": old_claim.to_dict(),
            "old_episode_id": candidate["id"],
            "contradiction_score": round(final_score, 3),
            "temporal_boost": temporal_boost,
            "negation_boost": neg_boost,
        }

    def supersede_old(self, old_episode_id: int, new_episode_id: int):
        """Mark an old episode as superseded by a newer one."""
        from ..extensions import supersede_episode

        try:
            supersede_episode(self.storage, old_episode_id, new_episode_id)
            logger.info(
                f"Superseded episode {old_episode_id} -> {new_episode_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to supersede episode: {e}")
