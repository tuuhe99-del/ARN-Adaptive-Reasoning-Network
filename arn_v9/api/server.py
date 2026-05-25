"""
ARN v9 REST API Server
========================
Production-ready FastAPI wrapper that turns ARN into a service.

Endpoints:
    POST /v1/memory/store     — Store a new memory
    POST /v1/memory/recall    — Recall relevant memories
    POST /v1/memory/context   — Get formatted context window
    POST /v1/memory/maintain  — Run consolidation
    GET  /v1/memory/stats     — Get system statistics
    GET  /v1/health           — Health check
    DELETE /v1/memory/agent   — Delete all agent data

Multi-tenancy:
    Each agent_id gets isolated storage. No cross-agent data leakage.
    Optional API key auth via X-API-Key header.

Deployment:
    # Local/Pi:
    uvicorn arn_v9.api.server:app --host 0.0.0.0 --port 8742

    # Docker:
    docker run -p 8742:8742 -v arn_data:/data arn-v9-api

    # Production (with workers):
    uvicorn arn_v9.api.server:app --host 0.0.0.0 --port 8742 --workers 1
    # NOTE: workers=1 because the embedding model is ~90MB per process.
    # For higher throughput, put a reverse proxy in front and scale
    # horizontally with separate containers per worker.
"""

import os
import re
import sys
import time
import json
import shutil
import logging
import asyncio
import secrets
import threading as _threading
from typing import Optional, List
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path as _Path

try:
    import psutil
    _ram_bytes = psutil.virtual_memory().total
    GRAPH_MODE = "full" if (_ram_bytes / (1024 ** 3)) >= 4.0 else "lite"
except Exception:
    GRAPH_MODE = "lite"

from fastapi import FastAPI, HTTPException, Depends, Request, Header, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure arn_v9 is importable
_api_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_api_dir))
sys.path.insert(0, _package_root)

from arn_v9.plugin import ARNPlugin

logger = logging.getLogger("arn.api")

# =========================================================
# PER-IP TOKEN BUCKET RATE LIMITER
# =========================================================

_rate_limit_lock = _threading.Lock()
_rate_buckets: dict = defaultdict(lambda: {"tokens": 60.0, "last": time.time()})
_RATE_LIMIT_RPS = float(os.environ.get("ARN_RATE_LIMIT_RPS", "60"))  # requests per second per IP


def _check_rate_limit(client_ip: str) -> bool:
    """Token bucket rate limiter. Returns True if request is allowed."""
    with _rate_limit_lock:
        now = time.time()
        bucket = _rate_buckets[client_ip]
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(_RATE_LIMIT_RPS, bucket["tokens"] + elapsed * _RATE_LIMIT_RPS)
        bucket["last"] = now
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False


async def rate_limit_dep(request: Request):
    """FastAPI dependency: per-IP token bucket rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# =========================================================
# CONTRADICTION DETECTION
# =========================================================

# Only these patterns are checked for contradictions.
# Each tuple is (compiled_regex, category_label).
# A contradiction is: same pattern matches both old and new fact,
# but the captured group (the "value") differs.
#
# Conservative design: we only supersede facts where we can prove
# the subject+attribute is the same but the value changed.
# "User is working on X" is intentionally excluded — users can
# work on multiple projects simultaneously.
_CONTRADICTION_TEMPLATES = [
    # Language/tool preference: "User prefers Python", "User uses JavaScript"
    (re.compile(
        r"^User (?:prefer|like|use|work with)s?\s+(.+)$",
        re.IGNORECASE,
    ), "tool_preference"),
    # Name: "User's name is Alex"
    (re.compile(
        r"^User'?s? name is\s+(.+)$",
        re.IGNORECASE,
    ), "name"),
    # Location: "User is based in X", "User lives in X"
    (re.compile(
        r"^User (?:is based in|lives? in|is from|is located in)\s+(.+)$",
        re.IGNORECASE,
    ), "location"),
]

# Sources that are NEVER superseded (manually seeded / authoritative facts)
_PROTECTED_SOURCES = {"api"}

# Sources eligible for supersession
_SUPERSEDABLE_SOURCES = {"user", "agent", "extracted_fact"}


def _check_and_supersede_contradictions(
    plugin,
    new_content: str,
    memory_type: str,
    source: str,
    new_episode_id: int,
) -> None:
    """
    Conservative contradiction detector.

    Runs after a new extracted_fact is stored.  For each contradiction
    template: if the new fact matches a pattern AND an existing *active*
    fact of the same memory_type also matches the same pattern with a
    DIFFERENT captured value → supersede the old fact by setting
    invalidated_at and superseded_by.

    Safety guarantees:
    - Never runs when source == 'api' (protects seeded facts)
    - Never supersedes facts whose source is in _PROTECTED_SOURCES
    - Only checks facts of the same memory_type
    - Uses exact pattern matching, NOT semantic similarity
    """
    # Guard 1: only process extracted/user/agent facts, never api stores
    if source in _PROTECTED_SOURCES:
        return

    conn = plugin._arn.storage._get_conn()
    now = time.time()

    for pattern, _category in _CONTRADICTION_TEMPLATES:
        new_match = pattern.match(new_content.strip())
        if new_match is None:
            continue

        new_value = new_match.group(1).strip().lower()

        # Fetch all active (non-invalidated) episodes of the same memory_type
        # whose source is supersedable.  Exclude the just-inserted episode.
        rows = conn.execute(
            """
            SELECT id, content, source
            FROM episodes
            WHERE memory_type = ?
              AND invalidated_at IS NULL
              AND id != ?
            """,
            (memory_type, new_episode_id),
        ).fetchall()

        for row in rows:
            old_id = row[0]
            old_content = row[1]
            old_source = row[2]

            # Guard 2: never supersede protected sources
            if old_source in _PROTECTED_SOURCES:
                continue

            # Guard 3: old source must be supersedable
            if old_source not in _SUPERSEDABLE_SOURCES:
                continue

            old_match = pattern.match(old_content.strip())
            if old_match is None:
                continue

            old_value = old_match.group(1).strip().lower()

            # Only supersede if the value actually changed
            if old_value == new_value:
                continue

            # Supersede the old fact
            conn.execute(
                """
                UPDATE episodes
                SET invalidated_at = ?,
                    superseded_by  = ?
                WHERE id = ?
                """,
                (now, new_episode_id, old_id),
            )
            conn.commit()
            logger.info(
                f"[ARN] superseded episode {old_id} with {new_episode_id}: "
                f"{old_content[:50]} → {new_content[:50]}"
            )


IDEA_PATTERNS = [
    re.compile(r"(?:I'm thinking|I am thinking|I think|I want to|I'd like to|I plan to|I intend to)\s+(.{10,120})", re.IGNORECASE),
    re.compile(r"(?:what if|what about|how about)\s+(.{10,100})\?", re.IGNORECASE),
    re.compile(r"(?:my idea is|my plan is|the plan is|the goal is)\s+(.{10,120})", re.IGNORECASE),
    re.compile(r"(?:we should|we could|we might|let's|let us)\s+(.{10,100})", re.IGNORECASE),
    re.compile(r"(?:I was thinking about|I am thinking about|thinking of|considering)\s+(.{10,120})", re.IGNORECASE),
]


def _extract_ideas(text: str) -> list:
    """
    Scan text for idea/plan/hypothesis signals using IDEA_PATTERNS.

    Returns a deduplicated list of extracted idea strings (max 5).
    Each result is stripped and bounded to 10–200 chars.
    """
    seen_lower: set = set()
    results: list = []

    for pattern in IDEA_PATTERNS:
        for m in pattern.finditer(text):
            idea = m.group(1).strip()
            if len(idea) < 10 or len(idea) > 200:
                continue
            key = idea.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            results.append(idea)
            if len(results) >= 5:
                return results

    return results


def _run_decay_pass() -> None:
    """Apply importance decay to stale episodes across all loaded agents."""
    now = time.time()
    cutoff = now - ARN_DECAY_WINDOW_DAYS * 86400

    for agent_id, plugin in list(pool._plugins.items()):
        try:
            conn = plugin._arn.storage._get_conn()
            rows = conn.execute(
                """
                SELECT id, importance, last_accessed, created_at, source
                FROM episodes
                WHERE invalidated_at IS NULL
                  AND source NOT IN ('api')
                """,
            ).fetchall()

            decayed = 0
            pruned = 0
            for row in rows:
                ep_id      = row[0]
                importance = row[1]
                last_acc   = row[2]   # may be None
                created_at = row[3]
                source     = row[4]

                if source == "conversation":
                    decay_rate = ARN_DECAY_RATE * 2.0        # 2x faster
                    window_days = ARN_DECAY_WINDOW_DAYS / 2  # half the window (e.g., 3.5 days)
                    cutoff_for_ep = now - window_days * 86400
                else:
                    decay_rate = ARN_DECAY_RATE
                    cutoff_for_ep = cutoff  # original cutoff

                # Use last_accessed if available, else fall back to created_at
                last_touch = last_acc if last_acc else created_at
                if last_touch is None or last_touch >= cutoff_for_ep:
                    continue  # recently touched — skip

                new_imp = importance * (1.0 - decay_rate)
                if new_imp < ARN_PRUNE_THRESHOLD:
                    conn.execute(
                        "UPDATE episodes SET invalidated_at = ? WHERE id = ?",
                        (now, ep_id),
                    )
                    pruned += 1
                else:
                    conn.execute(
                        "UPDATE episodes SET importance = ? WHERE id = ?",
                        (new_imp, ep_id),
                    )
                    decayed += 1

            conn.commit()
            if decayed or pruned:
                logger.info(f"[ARN decay] agent={agent_id} decayed={decayed} pruned={pruned}")
        except Exception as e:
            logger.warning(f"[ARN decay] agent={agent_id} error: {e}")


def _run_consolidation_pass() -> None:
    """Merge near-duplicate extracted_fact episodes using embedding similarity."""
    import numpy as np

    for agent_id, plugin in list(pool._plugins.items()):
        try:
            conn = plugin._arn.storage._get_conn()

            # Fetch all active extracted_fact episodes
            rows = conn.execute(
                """
                SELECT id, content, importance, access_count
                FROM episodes
                WHERE invalidated_at IS NULL
                  AND source = 'extracted_fact'
                ORDER BY id
                """,
            ).fetchall()

            if len(rows) < ARN_CONSOLIDATE_MIN_CLUSTER:
                continue

            ep_ids        = [r[0] for r in rows]
            importances   = [r[2] for r in rows]
            access_counts = [r[3] for r in rows]

            # Retrieve stored embedding vectors using the batch API
            # get_episode_vectors(episode_ids) -> (matrix, ids_in_order)
            storage = plugin._arn.storage
            mat_raw, returned_ids = storage.get_episode_vectors(ep_ids)

            if mat_raw.shape[0] < ARN_CONSOLIDATE_MIN_CLUSTER:
                continue

            # Build a lookup: episode_id -> (importance, access_count, row_in_mat)
            id_to_orig = {ep_id: i for i, ep_id in enumerate(ep_ids)}
            id_to_mat  = {ep_id: mi for mi, ep_id in enumerate(returned_ids)}

            # Work only with episodes that have a vector in the returned matrix
            valid_ids = [eid for eid in returned_ids if eid in id_to_orig]
            if len(valid_ids) < ARN_CONSOLIDATE_MIN_CLUSTER:
                continue

            # Build aligned arrays for valid episodes
            mat_indices  = [id_to_mat[eid] for eid in valid_ids]
            mat          = mat_raw[mat_indices].copy().astype(np.float32)
            v_importances   = [importances[id_to_orig[eid]]   for eid in valid_ids]
            v_access_counts = [access_counts[id_to_orig[eid]] for eid in valid_ids]

            # Normalize for cosine similarity
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            mat   = mat / norms

            # Greedy clustering: O(n^2) — fine for typical extracted_fact counts
            sim_matrix = mat @ mat.T  # shape (n, n)
            merged   = set()
            clusters = []  # list of lists of indices into valid_ids

            for i in range(len(valid_ids)):
                if i in merged:
                    continue
                cluster = [i]
                for j in range(i + 1, len(valid_ids)):
                    if j in merged:
                        continue
                    if sim_matrix[i, j] >= ARN_CONSOLIDATE_THRESHOLD:
                        cluster.append(j)
                        merged.add(j)
                merged.add(i)
                if len(cluster) >= ARN_CONSOLIDATE_MIN_CLUSTER:
                    clusters.append(cluster)

            if not clusters:
                continue

            now = time.time()
            total_merged = 0
            for cluster in clusters:
                cluster_ids  = [valid_ids[ci]        for ci in cluster]
                cluster_imps = [v_importances[ci]    for ci in cluster]
                cluster_acs  = [v_access_counts[ci]  for ci in cluster]

                # Canonical = highest importance * access_count score
                scores   = [imp * max(1, ac) for imp, ac in zip(cluster_imps, cluster_acs)]
                best_idx = scores.index(max(scores))
                canonical_id = cluster_ids[best_idx]

                # Sum access_counts across cluster, merge importance (cap 0.95)
                total_ac   = sum(cluster_acs)
                merged_imp = min(0.95, max(cluster_imps) + 0.02 * (len(cluster) - 1))

                # Update canonical episode
                conn.execute(
                    "UPDATE episodes SET importance = ?, access_count = ? WHERE id = ?",
                    (merged_imp, total_ac, canonical_id),
                )

                # Invalidate duplicates
                for ep_id in cluster_ids:
                    if ep_id != canonical_id:
                        conn.execute(
                            "UPDATE episodes SET invalidated_at = ?, superseded_by = ? WHERE id = ?",
                            (now, canonical_id, ep_id),
                        )
                        total_merged += 1

            conn.commit()
            if total_merged:
                logger.info(
                    f"[ARN consolidate] agent={agent_id} merged={total_merged} clusters={len(clusters)}"
                )

        except Exception as e:
            logger.warning(f"[ARN consolidate] agent={agent_id} error: {e}")


# =========================================================
# CONFIGURATION
# =========================================================

DATA_ROOT = os.environ.get("ARN_DATA_ROOT", os.path.expanduser("~/.arn_data"))
API_KEY = os.environ.get("ARN_API_KEY", None)  # Set to enable auth
MAX_AGENTS = int(os.environ.get("ARN_MAX_AGENTS", "100"))
RATE_LIMIT_RPM = int(os.environ.get("ARN_RATE_LIMIT_RPM", "300"))  # requests per minute
MAX_CONTENT_LENGTH = int(os.environ.get("ARN_MAX_CONTENT_LENGTH", "10000"))  # chars

ARN_DECAY_INTERVAL_SECONDS = int(os.environ.get("ARN_DECAY_INTERVAL_SECONDS", str(24 * 3600)))  # default 24h
ARN_DECAY_RATE             = float(os.environ.get("ARN_DECAY_RATE", "0.05"))        # 5% per interval
ARN_DECAY_WINDOW_DAYS      = float(os.environ.get("ARN_DECAY_WINDOW_DAYS", "7"))    # only decay if not accessed in 7d
ARN_PRUNE_THRESHOLD        = float(os.environ.get("ARN_PRUNE_THRESHOLD", "0.05"))   # invalidate if importance < 5%

ARN_CONSOLIDATE_INTERVAL_SECONDS = int(os.environ.get("ARN_CONSOLIDATE_INTERVAL_SECONDS", str(6 * 3600)))  # default 6h
ARN_CONSOLIDATE_THRESHOLD        = float(os.environ.get("ARN_CONSOLIDATE_THRESHOLD", "0.85"))  # cosine sim threshold
ARN_CONSOLIDATE_MIN_CLUSTER      = int(os.environ.get("ARN_CONSOLIDATE_MIN_CLUSTER", "2"))     # min cluster size to merge


# Auto-generate API key if none is set
if API_KEY is None:
    _key_file = _Path(DATA_ROOT) / ".api_key"
    if _key_file.exists():
        _auto_key = _key_file.read_text().strip()
    else:
        _auto_key = secrets.token_urlsafe(32)
        _key_file.parent.mkdir(parents=True, exist_ok=True)
        _key_file.write_text(_auto_key)
    os.environ["ARN_API_KEY"] = _auto_key
    API_KEY = _auto_key
    logger.warning(
        f"ARN_API_KEY not set. Auto-generated key active: {_auto_key[:8]}... "
        f"(full key in {_key_file})"
    )

logger.info(f"[ARN] API key enforced. Key starts with: {API_KEY[:8]}...")


# =========================================================
# PYDANTIC MODELS
# =========================================================

class StoreRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$',
                          description="Agent namespace identifier")
    content: str = Field(..., min_length=1, max_length=10000,
                         description="Text content to store")
    importance: float = Field(0.5, ge=0.0, le=1.0,
                              description="Importance score 0.0-1.0")
    tags: List[str] = Field(default_factory=list,
                            description="Categorical tags")
    source: str = Field("api", max_length=50,
                        description="Source identifier")
    context: dict = Field(default_factory=dict,
                          description="Additional context metadata")
    memory_type: str = Field("episode", max_length=32,
                             description="Memory category: identity, preference, procedure, error, fact, episode, ...")


class StoreResponse(BaseModel):
    stored: bool
    episode_id: int
    prediction_error: float
    domain: Optional[str]
    surprising: bool


class RecallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    query: str = Field(..., min_length=1, max_length=5000,
                       description="Natural language query")
    top_k: int = Field(5, ge=1, le=50,
                       description="Number of results to return")
    memory_types: Optional[List[str]] = Field(
        None, description="Filter: 'episodic', 'semantic', or both")
    memory_type: Optional[str] = Field(
        None, description="Filter by memory category: identity, preference, procedure, error, fact, episode")


class RecallResult(BaseModel):
    id: Optional[int] = None
    content: str
    score: float
    type: str
    similarity: float
    importance: Optional[float] = None
    confidence: Optional[float] = None
    evidence_count: Optional[int] = None
    has_contradictions: Optional[bool] = None
    age_hours: Optional[float] = None
    created_at: Optional[float] = None
    memory_type: Optional[str] = None
    source: Optional[str] = None
    confidence_tier: Optional[str] = None
    calibrated_confidence: Optional[float] = None


class RecallResponse(BaseModel):
    results: List[RecallResult]
    query: str
    agent_id: str
    latency_ms: float


class ContextRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    query: Optional[str] = Field(None, max_length=5000)
    max_tokens: int = Field(1000, ge=100, le=10000)


class ContextResponse(BaseModel):
    context: str
    agent_id: str
    latency_ms: float


class MemoryEditRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    episode_id: int = Field(..., description="Episode ID to edit")
    content: str = Field(..., min_length=1, max_length=10000)
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    memory_type: Optional[str] = Field(None, max_length=32)


class MemoryDeleteRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    episode_id: int = Field(..., description="Episode ID to delete")


class MemoryListRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    memory_type: Optional[str] = Field(None, max_length=32)
    limit: int = Field(50, ge=1, le=500)


class MaintainRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')


class StatsResponse(BaseModel):
    agent_id: str
    total_experiences: int
    consolidation_count: int
    episodic_count: int
    semantic_count: int
    working_memory_active: int
    storage_mb: float
    embedding_model_loaded: bool
    embedding_dim: int
    cache_hit_rate: float
    columns: list


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    agents_loaded: int
    data_root: str


class DeleteAgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    confirm: bool = Field(False, description="Must be true to delete")


VALID_RELATION_TYPES = {"relates_to", "used_by", "part_of", "leads_to", "contradicts"}


class LinkRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    from_episode_id: int
    to_episode_id: int
    relation_type: str = Field("relates_to", max_length=32)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class UnlinkRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    link_id: int


class LinksRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    episode_id: Optional[int] = None


class ExchangeRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    user_message: str = Field(..., min_length=1, max_length=10000)
    agent_response: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    tools_used: Optional[List[dict]] = None  # list of {name, summary, result_summary}
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)


class WorkflowStep(BaseModel):
    tool_name: str
    action_summary: str          # plain English: "Searched for Python API frameworks"
    result_summary: Optional[str] = None  # plain English: "Found Flask and FastAPI"
    success: Optional[bool] = True


class WorkflowRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    session_id: Optional[str] = None
    task_description: str        # overall task: "Debug the ARN API auth flow"
    steps: List[WorkflowStep]
    importance: Optional[float] = 0.72


class InjectRequest(BaseModel):
    agent_id: str
    query: str
    session_id: Optional[str] = None
    already_seen_ids: Optional[List[int]] = []
    top_k: Optional[int] = 6
    min_score: Optional[float] = 0.25


# =========================================================
# AGENT POOL (lazy-loaded, cached plugins per agent_id)
# =========================================================

class AgentPool:
    """
    Manages ARNPlugin instances per agent_id.
    Lazy-loaded: first request for an agent_id creates the plugin.
    All plugins share the same embedding model in memory.
    """

    def __init__(self, data_root: str, max_agents: int = 100):
        self._plugins: dict[str, ARNPlugin] = {}
        self._data_root = data_root
        self._max_agents = max_agents
        self._access_times: dict[str, float] = {}

    def get(self, agent_id: str) -> ARNPlugin:
        if agent_id not in self._plugins:
            if len(self._plugins) >= self._max_agents:
                self._evict_oldest()

            self._plugins[agent_id] = ARNPlugin(
                agent_id=agent_id,
                data_root=self._data_root,
                auto_consolidate=True,
                # High threshold prevents noise from triggering consolidation.
                # Override via ARN_CONSOLIDATION_THRESHOLD env var.
                consolidation_threshold=int(os.environ.get("ARN_CONSOLIDATION_THRESHOLD", "2048")),
            )

        self._access_times[agent_id] = time.time()
        return self._plugins[agent_id]

    def _evict_oldest(self):
        """Evict the least recently used agent to make room."""
        if not self._access_times:
            return
        oldest = min(self._access_times, key=self._access_times.get)
        plugin = self._plugins.pop(oldest, None)
        self._access_times.pop(oldest, None)
        if plugin:
            plugin.shutdown()
            logger.info(f"Evicted agent '{oldest}' from pool")

    def delete_agent(self, agent_id: str):
        """Delete all data for an agent."""
        plugin = self._plugins.pop(agent_id, None)
        self._access_times.pop(agent_id, None)
        if plugin:
            plugin.shutdown()

        agent_dir = os.path.join(self._data_root, agent_id)
        if os.path.exists(agent_dir):
            shutil.rmtree(agent_dir)
            logger.info(f"Deleted agent data: {agent_dir}")

    @property
    def loaded_count(self) -> int:
        return len(self._plugins)

    def shutdown_all(self):
        for agent_id, plugin in self._plugins.items():
            plugin.shutdown()
        self._plugins.clear()
        self._access_times.clear()


# =========================================================
# RATE LIMITER
# =========================================================

class RateLimiter:
    """Simple sliding-window rate limiter per agent_id."""

    def __init__(self, rpm: int = 300):
        self._rpm = rpm
        self._windows: dict[str, list] = defaultdict(list)

    def check(self, agent_id: str) -> bool:
        now = time.time()
        window = self._windows[agent_id]
        # Remove timestamps older than 60s
        self._windows[agent_id] = [t for t in window if now - t < 60]
        if len(self._windows[agent_id]) >= self._rpm:
            return False
        self._windows[agent_id].append(now)
        return True


# =========================================================
# APP SETUP
# =========================================================

pool: Optional[AgentPool] = None
rate_limiter: Optional[RateLimiter] = None
start_time: float = 0


async def _decay_loop():
    """Background loop: run decay pass every ARN_DECAY_INTERVAL_SECONDS."""
    await asyncio.sleep(ARN_DECAY_INTERVAL_SECONDS)  # first run after one full interval
    while True:
        try:
            await asyncio.to_thread(_run_decay_pass)
        except Exception as e:
            logger.warning(f"[ARN decay] loop error: {e}")
        await asyncio.sleep(ARN_DECAY_INTERVAL_SECONDS)


async def _consolidation_loop():
    """Background loop: run consolidation pass every ARN_CONSOLIDATE_INTERVAL_SECONDS."""
    await asyncio.sleep(ARN_CONSOLIDATE_INTERVAL_SECONDS)
    while True:
        try:
            await asyncio.to_thread(_run_consolidation_pass)
        except Exception as e:
            logger.warning(f"[ARN consolidate] loop error: {e}")
        await asyncio.sleep(ARN_CONSOLIDATE_INTERVAL_SECONDS)


async def _background_maintenance():
    DECAY_INTERVAL = int(os.environ.get("ARN_DECAY_INTERVAL_HOURS", "6")) * 3600
    CONSOLIDATION_INTERVAL = int(os.environ.get("ARN_CONSOLIDATION_INTERVAL_HOURS", "24")) * 3600
    last_decay = 0.0
    last_consolidation = 0.0
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        now = time.time()
        if now - last_decay >= DECAY_INTERVAL:
            try:
                await asyncio.to_thread(_run_decay_pass)
                last_decay = now
            except Exception as e:
                logger.warning(f"[maintenance] decay error: {e}")
        if now - last_consolidation >= CONSOLIDATION_INTERVAL:
            try:
                await asyncio.to_thread(_run_consolidation_pass)
                last_consolidation = now
            except Exception as e:
                logger.warning(f"[maintenance] consolidation error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, rate_limiter, start_time
    start_time = time.time()
    
    # Fail fast: verify the embedding model loads before accepting requests.
    # A degraded ARN (hash vectors) is worse than no ARN — it returns
    # confident-looking garbage. Better to refuse to start.
    from arn_v9.core.embeddings import EmbeddingEngine
    _test_engine = EmbeddingEngine(use_model=True)
    if _test_engine.is_degraded:
        logger.critical(
            "FATAL: Embedding model could not be loaded. "
            "The API server CANNOT function without real embeddings. "
            "Install sentence-transformers: pip install sentence-transformers "
            "and ensure the model can be downloaded (internet on first run) "
            "or pre-cached at ~/.cache/huggingface/hub/"
        )
        raise RuntimeError(
            "ARN API cannot start: embedding model unavailable. "
            "Install: pip install sentence-transformers"
        )
    del _test_engine  # Free the test instance
    
    pool = AgentPool(data_root=DATA_ROOT, max_agents=MAX_AGENTS)
    rate_limiter = RateLimiter(rpm=RATE_LIMIT_RPM)
    logger.info(f"ARN v9 API started. Data root: {DATA_ROOT}")
    asyncio.create_task(_decay_loop())
    asyncio.create_task(_consolidation_loop())
    # NOTE: _background_maintenance() is intentionally NOT started here.
    # _decay_loop() and _consolidation_loop() already cover both passes on
    # their own schedules (ARN_DECAY_INTERVAL_SECONDS /
    # ARN_CONSOLIDATE_INTERVAL_SECONDS).  Running _background_maintenance()
    # in parallel would cause every decay and consolidation pass to execute
    # twice, doubling DB writes and CPU load with no benefit.
    yield
    pool.shutdown_all()
    logger.info("ARN v9 API shut down.")


app = FastAPI(
    title="ARN v9 API",
    description="Brain-inspired cognitive memory for AI agents",
    version="9.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load dashboard HTML from file at import time
_dashboard_path = _Path(__file__).with_name("dashboard.html")
try:
    DASHBOARD_HTML = _dashboard_path.read_text()
except Exception:
    DASHBOARD_HTML = "<html><body>Dashboard unavailable</body></html>"


# =========================================================
# AUTH & RATE LIMIT MIDDLEWARE
# =========================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """API key auth. Always enforced — API_KEY is always set (auto-generated if not provided)."""
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Local R&D dashboard for browsing ARN memories."""
    return HTMLResponse(DASHBOARD_HTML.replace("{{ graph_mode }}", GRAPH_MODE))


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns "ok" if the server is running and embeddings are functional.
    Returns "degraded" if any loaded agent has lost its embedding model
    (shouldn't happen in normal operation, but catches runtime corruption).
    """
    status = "ok"
    
    # Check if any loaded agent is in degraded mode
    if pool:
        for agent_id, plugin in pool._plugins.items():
            if plugin._arn.embedder.is_degraded:
                status = "degraded"
                break
    
    return HealthResponse(
        status=status,
        version="9.0.0",
        uptime_seconds=round(time.time() - start_time, 1),
        agents_loaded=pool.loaded_count if pool else 0,
        data_root=DATA_ROOT,
    )


@app.post("/v1/memory/store", response_model=StoreResponse,
          dependencies=[Depends(verify_api_key)])
async def store_memory(req: StoreRequest):
    """
    Store a new memory for an agent.

    The memory is encoded into a 384-dim semantic vector, processed
    through domain columns for prediction error, and persisted to
    SQLite + memmap storage.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plugin = pool.get(req.agent_id)
    result = await asyncio.to_thread(
        plugin.store,
        content=req.content,
        importance=req.importance,
        tags=req.tags,
        source=req.source,
        context=req.context,
        memory_type=req.memory_type,
    )

    # Run contradiction detection for extracted facts.
    # This is intentionally gated to extracted_fact source only so that
    # raw user/agent messages (which are often partial sentences) don't
    # trigger false-positive supersessions.
    if req.source == "extracted_fact":
        await asyncio.to_thread(
            _check_and_supersede_contradictions,
            plugin,
            req.content,
            req.memory_type,
            req.source,
            result["episode_id"],
        )

    return StoreResponse(
        stored=result["stored"],
        episode_id=result["episode_id"],
        prediction_error=round(result["prediction_error"], 4),
        domain=result.get("domain"),
        surprising=result.get("surprising", False),
    )


@app.post("/v1/memory/recall", response_model=RecallResponse,
          dependencies=[Depends(verify_api_key), Depends(rate_limit_dep)])
async def recall_memory(req: RecallRequest):
    """
    Recall relevant memories for a query.

    Searches both episodic (specific events) and semantic (consolidated
    knowledge) memory, scoring by semantic similarity with importance
    and recency as minor factors.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    start = time.time()
    plugin = pool.get(req.agent_id)
    raw_results = await asyncio.to_thread(
        plugin.recall,
        query=req.query,
        top_k=req.top_k,
        memory_types=req.memory_types,
        memory_type=req.memory_type,
    )
    latency = (time.time() - start) * 1000

    results = [RecallResult(**r) for r in raw_results]

    return RecallResponse(
        results=results,
        query=req.query,
        agent_id=req.agent_id,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/memory/context", response_model=ContextResponse,
          dependencies=[Depends(verify_api_key)])
async def get_context(req: ContextRequest):
    """
    Get a formatted context window for LLM prompt injection.

    Returns a markdown-formatted string containing working memory
    contents and relevant long-term memories, suitable for prepending
    to a system prompt.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    start = time.time()
    plugin = pool.get(req.agent_id)
    context = await asyncio.to_thread(
        plugin.get_context_window,
        query=req.query,
        max_tokens=req.max_tokens,
    )
    latency = (time.time() - start) * 1000

    return ContextResponse(
        context=context,
        agent_id=req.agent_id,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/memory/edit", dependencies=[Depends(verify_api_key)])
async def edit_memory(req: MemoryEditRequest):
    """
    Edit an existing memory episode.
    Updates content and re-embeds.  Old vector is overwritten.
    """
    plugin = pool.get(req.agent_id)
    ep = plugin._arn.storage.get_episode(req.episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    # Re-embed the new content
    new_vec = plugin._arn.embedder.encode(req.content, mode="passage")

    conn = plugin._arn.storage._get_conn()
    updates = ["content = ?", "vec_index = ?"]
    params = [req.content, ep['vec_index']]

    if req.importance is not None:
        updates.append("importance = ?")
        params.append(req.importance)
    if req.memory_type is not None:
        updates.append("memory_type = ?")
        params.append(req.memory_type)

    params.append(req.episode_id)
    conn.execute(
        f"UPDATE episodes SET {', '.join(updates)} WHERE id = ?",
        params
    )
    conn.commit()

    # Overwrite the vector in memmap
    plugin._arn.storage._episodic_vectors[ep['vec_index']] = new_vec

    return {"edited": True, "episode_id": req.episode_id}


@app.post("/v1/memory/delete", dependencies=[Depends(verify_api_key)])
async def delete_memory(req: MemoryDeleteRequest):
    """Delete a single memory episode."""
    plugin = pool.get(req.agent_id)
    ep = plugin._arn.storage.get_episode(req.episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    plugin._arn.storage.delete_episodes([req.episode_id])
    return {"deleted": True, "episode_id": req.episode_id}


@app.delete("/v1/memory/{episode_id:int}", dependencies=[Depends(verify_api_key)])
async def delete_memory_rest(agent_id: str, episode_id: int):
    """Delete a single memory episode (REST-style DELETE)."""
    if not agent_id or not re.match(r'^[a-zA-Z0-9_\-]+$', agent_id):
        raise HTTPException(status_code=400, detail="Invalid agent_id")
    plugin = pool.get(agent_id)
    ep = plugin._arn.storage.get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    plugin._arn.storage.delete_episodes([episode_id])
    return {"deleted": True, "episode_id": episode_id}


@app.post("/v1/memory/list", dependencies=[Depends(verify_api_key)])
async def list_memories(req: MemoryListRequest):
    """List memories for an agent, optionally filtered by type."""
    plugin = pool.get(req.agent_id)
    episodes = plugin._arn.storage.get_all_episodes(
        memory_type=req.memory_type, limit=req.limit
    )
    return {
        "agent_id": req.agent_id,
        "count": len(episodes),
        "memories": [
            {
                "id": e["id"],
                "content": e["content"],
                "memory_type": e.get("memory_type", "episode"),
                "source": e.get("source", "unknown"),
                "importance": e["importance"],
                "created_at": e["created_at"],
                "access_count": e["access_count"],
            }
            for e in episodes
        ],
    }


@app.post("/v1/memory/maintain",
          dependencies=[Depends(verify_api_key)])
async def maintain(req: MaintainRequest):
    """
    Run memory consolidation for an agent.

    Clusters episodic memories into semantic knowledge, detects
    contradictions, and prunes old low-importance episodes.
    Call during idle periods.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plugin = pool.get(req.agent_id)
    stats = plugin.maintain()
    return stats


@app.get("/v1/memory/stats/{agent_id}", response_model=StatsResponse,
         dependencies=[Depends(verify_api_key)])
async def get_stats(agent_id: str = Path(..., pattern=r'^[a-zA-Z0-9_\-]+$', max_length=64)):
    """Get comprehensive statistics for an agent's memory system."""
    plugin = pool.get(agent_id)
    raw = plugin.get_stats()

    return StatsResponse(
        agent_id=agent_id,
        total_experiences=raw["total_experiences"],
        consolidation_count=raw["consolidation_count"],
        episodic_count=raw["episodic_count"],
        semantic_count=raw["semantic_count"],
        working_memory_active=raw["working_memory_active"],
        storage_mb=round(raw["storage"]["total_size_mb"], 2),
        embedding_model_loaded=raw["embeddings"]["model_loaded"],
        embedding_dim=raw["embeddings"]["embedding_dim"],
        cache_hit_rate=round(raw["embeddings"]["cache_hit_rate"], 4),
        columns=raw["columns"],
    )


@app.delete("/v1/memory/agent",
            dependencies=[Depends(verify_api_key)])
async def delete_agent(req: DeleteAgentRequest):
    """
    Delete ALL data for an agent. Irreversible.
    Set confirm=true to actually delete.
    """
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to delete all agent data. This is irreversible."
        )

    pool.delete_agent(req.agent_id)
    return {"deleted": True, "agent_id": req.agent_id}


@app.post("/v1/memory/extract", dependencies=[Depends(verify_api_key)])
async def extract_facts(body: dict):
    """
    Heuristic fact extraction from a user/agent exchange.

    No LLM calls — pure regex/string pattern matching.
    Returns: { "facts": [ { "content": str, "memory_type": str, "importance": float } ] }
    """
    agent_id = body.get("agent_id", "default")
    user_message = body.get("user_message", "")
    agent_reply = body.get("agent_reply", "")

    facts = []
    combined = f"{user_message}\n{agent_reply}"

    # Helper: add a fact if not duplicate in this batch
    def add_fact(content: str, memory_type: str, importance: float):
        content = content.strip()
        if not content or len(content) < 8:
            return
        for existing in facts:
            if existing["content"].lower() == content.lower():
                return
        facts.append({"content": content, "memory_type": memory_type, "importance": importance})

    # ---- Identity patterns ----
    # "my name is X" / "I'm X" / "I am X" (name introductions)
    for m in re.finditer(
        r"\bmy name is ([A-Z][a-zA-Z\-']{1,30})\b",
        combined, re.IGNORECASE
    ):
        add_fact(f"User's name is {m.group(1)}", "identity", 0.9)

    # "I'm <Name>" / "I am <Name>" — only if followed by sentence boundary or comma
    for m in re.finditer(
        r"\bI(?:'m| am) ([A-Z][a-zA-Z\-']{1,30})(?:\s*[,.]|\s+and\b|\s+from\b|$)",
        combined
    ):
        name = m.group(1)
        # Exclude common non-name words that start with capital
        skip = {"The","A","An","Not","So","Just","Here","There","Going","Working","Building","Using","Also","Now"}
        if name not in skip:
            add_fact(f"User's name is {name}", "identity", 0.9)

    # Role / project descriptions
    for m in re.finditer(
        r"\bI(?:'m| am) (?:a |an )?([a-zA-Z][a-zA-Z\s]{3,40}?) (?:at|for|by|in)\b",
        combined, re.IGNORECASE
    ):
        add_fact(f"User is a {m.group(1).strip()}", "identity", 0.85)

    for m in re.finditer(
        r"\bI(?:'m| am) working (?:on|at) ([^.,\n]{5,80})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User is working on {m.group(1).strip()}", "identity", 0.85)

    for m in re.finditer(
        r"\bI(?:'m| am) building ([^.,\n]{5,80})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User is building {m.group(1).strip()}", "identity", 0.85)

    for m in re.finditer(
        r"\bmy project is ([^.,\n]{5,80})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User's project is {m.group(1).strip()}", "identity", 0.85)

    for m in re.finditer(
        r"\bmy team ([^.,\n]{5,80})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User's team {m.group(1).strip()}", "identity", 0.85)

    # Team member role patterns: "X handles Y", "X is our Y", "X is my Y"
    for m in re.finditer(
        r"\b([A-Z][a-zA-Z]{1,20}) handles ([^.,\n]{3,60})",
        combined
    ):
        add_fact(f"{m.group(1)} handles {m.group(2).strip()}", "identity", 0.85)

    for m in re.finditer(
        r"\b([A-Z][a-zA-Z]{1,20}) is (?:our|my) ([^.,\n]{3,60})",
        combined
    ):
        add_fact(f"{m.group(1)} is user's {m.group(2).strip()}", "identity", 0.85)

    # "My co-founder/partner/CTO is NAME" — person + role relative to user
    for m in re.finditer(
        r"\bmy (co-founder|cto|ceo|coo|vp|partner|manager|boss|lead|collaborator)(?:\s+on\s+[^,.\n]{3,40})?\s+is\s+([A-Z][a-zA-Z]{1,20})\b",
        combined, re.IGNORECASE
    ):
        role = m.group(1).lower()
        name = m.group(2)
        add_fact(f"{name} is User's {role}", "identity", 0.85)

    # "NAME is my co-founder/partner/CTO" — reverse form
    for m in re.finditer(
        r"\b([A-Z][a-zA-Z]{1,20}) is my (co-founder|cto|ceo|coo|vp|partner|manager|boss|lead|collaborator)\b",
        combined
    ):
        name = m.group(1)
        role = m.group(2).lower()
        add_fact(f"{name} is User's {role}", "identity", 0.85)

    # ---- Preference patterns ----
    for m in re.finditer(
        r"\bI (?:prefer|like|love|hate|dislike|enjoy) ([^.,\n]{3,80})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User prefers/likes: {m.group(1).strip()}", "preference", 0.85)

    for m in re.finditer(
        r"\bI use ([^.,\n]{3,60})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User uses {m.group(1).strip()}", "preference", 0.85)

    for m in re.finditer(
        r"\bI work (?:with|in|using) ([^.,\n]{3,60})",
        combined, re.IGNORECASE
    ):
        add_fact(f"User works with/in {m.group(1).strip()}", "preference", 0.85)

    # ---- Procedure / decision patterns ----
    for m in re.finditer(
        r"\bwe decided to ([^.,\n]{5,120})",
        combined, re.IGNORECASE
    ):
        add_fact(f"Decision: we decided to {m.group(1).strip()}", "procedure", 0.85)

    for m in re.finditer(
        r"\bgoing forward[,\s]+(?:we(?:'ll|'re| will| are)|I(?:'ll|'m| will| am))?\s*([^.,\n]{5,120})",
        combined, re.IGNORECASE
    ):
        add_fact(f"Procedure: going forward {m.group(1).strip()}", "procedure", 0.85)

    for m in re.finditer(
        r"\bfrom now on[,\s]+([^.,\n]{5,120})",
        combined, re.IGNORECASE
    ):
        add_fact(f"Procedure: from now on {m.group(1).strip()}", "procedure", 0.85)

    # ---- Server/config facts ----
    # "my server / our server / the API runs on / is at / is on port X"
    for m in re.finditer(
        r"\b(?:my|our|the) (?:server|api|service|backend|endpoint|database|db)\b[^.,\n]{0,40}?\b(?:runs? on|is at|listens? on|on port|at port|port)\b\s*([^.,\n]{3,60})",
        combined, re.IGNORECASE
    ):
        add_fact(f"Server/API config: {m.group(0).strip()}", "procedure", 0.85)

    # explicit "localhost:PORT" or "0.0.0.0:PORT" mentions in declarative context
    for m in re.finditer(
        r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})\b",
        combined
    ):
        context_start = max(0, m.start() - 60)
        context_snippet = combined[context_start:m.end() + 60]
        add_fact(f"Server endpoint mentioned: {context_snippet.strip()}", "procedure", 0.75)

    return {"facts": facts}


@app.post("/v1/memory/feedback", dependencies=[Depends(verify_api_key)])
async def record_access_feedback(body: dict):
    """
    Record that specific episodes were shown to the agent.
    Increments access_count and applies a small importance boost.

    Body: { "agent_id": str, "episode_ids": [int, ...] }
    Returns: { "updated": int }
    """
    agent_id = body.get("agent_id", "default")
    episode_ids = body.get("episode_ids", [])

    if not episode_ids:
        return {"updated": 0}

    plugin = pool.get(agent_id)
    conn = plugin._arn.storage._get_conn()

    updated = 0
    now = time.time()
    for ep_id in episode_ids[:20]:  # cap at 20 per call
        try:
            conn.execute(
                "UPDATE episodes SET access_count = access_count + 1, "
                "last_accessed = ? "
                "WHERE id = ? AND invalidated_at IS NULL",
                (now, ep_id)
            )
            conn.execute(
                "UPDATE episodes SET importance = MIN(0.95, importance + 0.01) "
                "WHERE id = ? AND invalidated_at IS NULL",
                (ep_id,)
            )
            updated += 1
        except Exception:
            pass
    conn.commit()

    return {"updated": updated}


@app.post("/v1/memory/decay", dependencies=[Depends(verify_api_key)])
async def trigger_decay():
    """Manually trigger an importance decay pass. Safe to call at any time."""
    await asyncio.to_thread(_run_decay_pass)
    return {"status": "ok"}


@app.post("/v1/memory/consolidate", dependencies=[Depends(verify_api_key)])
async def trigger_consolidate():
    """Manually trigger a semantic consolidation pass."""
    await asyncio.to_thread(_run_consolidation_pass)
    return {"status": "ok"}


class EmbedSimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=5000)
    text_b: str = Field(..., min_length=1, max_length=5000)


@app.post("/v1/memory/embed_similarity", dependencies=[Depends(verify_api_key)])
async def embed_similarity(req: EmbedSimilarityRequest):
    """
    Compute cosine similarity between two texts using the loaded embedding model.

    Returns { "similarity": float } in range [-1, 1].
    Values near 1.0 indicate the same topic; values below ~0.45 indicate a topic shift.

    Used by the plugin for session-level topic drift detection.
    """
    import numpy as np
    from arn_v9.core.embeddings import EmbeddingEngine

    def _compute():
        # Reuse the global embedding engine from any already-loaded plugin
        # rather than paying the 3-5 second model-load cost on every call.
        loaded = list(pool._plugins.values()) if pool else []
        if loaded:
            engine = loaded[0]._arn.embedder
        else:
            engine = EmbeddingEngine(use_model=True)
        vec_a = engine.encode(req.text_a, mode="query").astype(np.float32)
        vec_b = engine.encode(req.text_b, mode="query").astype(np.float32)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clamp to [-1, 1] to guard against any floating-point drift
        return max(-1.0, min(1.0, sim))

    similarity = await asyncio.to_thread(_compute)
    return {"similarity": similarity}


@app.post("/v1/memory/link", dependencies=[Depends(verify_api_key)])
async def create_link(req: LinkRequest):
    """Create a directed link between two episodes."""
    if req.relation_type not in VALID_RELATION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid relation_type. Must be one of: {sorted(VALID_RELATION_TYPES)}",
        )
    plugin = pool.get(req.agent_id)
    storage = plugin._arn.storage
    if storage.get_episode(req.from_episode_id) is None:
        raise HTTPException(status_code=404, detail=f"Episode {req.from_episode_id} not found")
    if storage.get_episode(req.to_episode_id) is None:
        raise HTTPException(status_code=404, detail=f"Episode {req.to_episode_id} not found")
    link_id = storage.create_link(
        req.from_episode_id, req.to_episode_id,
        req.relation_type, req.confidence,
    )
    return {"linked": True, "link_id": link_id}


@app.post("/v1/memory/unlink", dependencies=[Depends(verify_api_key)])
async def delete_link(req: UnlinkRequest):
    """Delete a link by ID."""
    plugin = pool.get(req.agent_id)
    plugin._arn.storage.delete_link(req.link_id)
    return {"unlinked": True, "link_id": req.link_id}


@app.post("/v1/memory/links", dependencies=[Depends(verify_api_key)])
async def list_links(req: LinksRequest):
    """List links for a given episode, or all links for the agent."""
    plugin = pool.get(req.agent_id)
    storage = plugin._arn.storage
    if req.episode_id is not None:
        links = storage.get_links_for_episode(req.episode_id)
    else:
        links = storage.get_all_links()
    return {"agent_id": req.agent_id, "count": len(links), "links": links}


@app.post("/v1/memory/exchange", dependencies=[Depends(verify_api_key), Depends(rate_limit_dep)])
async def store_exchange(req: ExchangeRequest):
    """
    Store a complete user/agent exchange atomically.

    Stores the user message, agent response, and any tool calls as
    separate episodes with proper attribution.  Also runs heuristic
    fact extraction on the combined text so identity / preference facts
    are indexed immediately.

    Returns: { "stored": true, "episode_ids": [list of stored IDs] }
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plugin = pool.get(req.agent_id)
    episode_ids: List[int] = []

    # 1. Store user message
    user_imp = req.importance if req.importance is not None else 0.6
    user_result = await asyncio.to_thread(
        plugin.store,
        content=req.user_message,
        importance=user_imp,
        tags=[],
        source="user",
        context={"session_id": req.session_id} if req.session_id else {},
        memory_type="episode",
    )
    if user_result.get("stored"):
        episode_ids.append(user_result["episode_id"])

    # 2. Store agent response
    agent_imp = req.importance if req.importance is not None else 0.65
    agent_result = await asyncio.to_thread(
        plugin.store,
        content=req.agent_response,
        importance=agent_imp,
        tags=[],
        source="agent",
        context={"session_id": req.session_id} if req.session_id else {},
        memory_type="episode",
    )
    if agent_result.get("stored"):
        episode_ids.append(agent_result["episode_id"])

    # 3. Store combined conversation episode
    MIN_CONV_LEN = 40  # skip trivial exchanges
    if len(req.user_message.strip()) >= MIN_CONV_LEN and len(req.agent_response.strip()) >= MIN_CONV_LEN:
        user_excerpt = req.user_message.strip()[:200]
        agent_excerpt = req.agent_response.strip()[:300]
        conv_content = f"User asked: {user_excerpt}\nAgent: {agent_excerpt}"
        conv_result = await asyncio.to_thread(
            plugin.store,
            content=conv_content,
            source="conversation",
            memory_type="episode",
            context={"session_id": req.session_id or "unknown"},
            importance=req.importance if req.importance is not None else 0.55,
        )
        if conv_result and conv_result.get("episode_id"):
            episode_ids.append(conv_result["episode_id"])

    # 4. Store tool call summaries
    for tool in (req.tools_used or []):
        name = tool.get("name", "unknown")
        summary = tool.get("summary", "")
        result_summary = tool.get("result_summary", "")
        tool_content = f"Agent used {name}: {summary}. Result: {result_summary}"
        tool_result = await asyncio.to_thread(
            plugin.store,
            content=tool_content,
            importance=0.70,
            tags=[],
            source="tool_call",
            context={"session_id": req.session_id} if req.session_id else {},
            memory_type="procedure",
        )
        if tool_result.get("stored"):
            episode_ids.append(tool_result["episode_id"])

    # 5. Run heuristic fact extraction on the combined exchange text
    combined = f"{req.user_message}\n{req.agent_response}"

    def _extract_and_store_facts():
        import re as _re
        facts = []

        def add_fact(content: str, memory_type: str, importance: float):
            content = content.strip()
            if not content or len(content) < 8:
                return
            for existing in facts:
                if existing["content"].lower() == content.lower():
                    return
            facts.append({"content": content, "memory_type": memory_type, "importance": importance})

        for m in _re.finditer(r"\bmy name is ([A-Z][a-zA-Z\-']{1,30})\b", combined, _re.IGNORECASE):
            add_fact(f"User's name is {m.group(1)}", "identity", 0.9)
        for m in _re.finditer(r"\bI(?:'m| am) ([A-Z][a-zA-Z\-']{1,30})(?:\s*[,.]|\s+and\b|\s+from\b|$)", combined):
            name = m.group(1)
            skip = {"The","A","An","Not","So","Just","Here","There","Going","Working","Building","Using","Also","Now"}
            if name not in skip:
                add_fact(f"User's name is {name}", "identity", 0.9)
        for m in _re.finditer(r"\bI (?:prefer|like|love|use) ([^.,\n]{3,80})", combined, _re.IGNORECASE):
            add_fact(f"User prefers/uses: {m.group(1).strip()}", "preference", 0.85)
        for m in _re.finditer(r"\b([A-Z][a-zA-Z]{1,20}) handles ([^.,\n]{3,60})", combined):
            add_fact(f"{m.group(1)} handles {m.group(2).strip()}", "identity", 0.85)

        stored_ids = []
        for fact in facts:
            res = plugin.store(
                content=fact["content"],
                importance=fact["importance"],
                tags=[],
                source="extracted_fact",
                context={"session_id": req.session_id} if req.session_id else {},
                memory_type=fact["memory_type"],
            )
            if res.get("stored"):
                stored_ids.append(res["episode_id"])
                _check_and_supersede_contradictions(
                    plugin, fact["content"], fact["memory_type"], "extracted_fact", res["episode_id"]
                )

        # Extract ideas/plans/hypotheticals from the user message
        try:
            ideas = _extract_ideas(req.user_message)
            for idea_text in ideas:
                idea_content = f"[Idea] {idea_text}"
                if len(idea_content.strip()) < 10 or len(idea_content.strip()) > 200:
                    continue
                idea_res = plugin.store(
                    content=idea_content,
                    importance=0.45,
                    tags=[],
                    source="conversation",
                    context={"session_id": req.session_id, "type": "idea", "extracted_from": "user_message"}
                          if req.session_id else {"type": "idea", "extracted_from": "user_message"},
                    memory_type="episodic",
                )
                if idea_res and idea_res.get("episode_id"):
                    stored_ids.append(idea_res["episode_id"])
        except Exception:
            pass  # never break exchange storage

        return stored_ids

    fact_ids = await asyncio.to_thread(_extract_and_store_facts)
    episode_ids.extend(fact_ids)

    return {"stored": True, "episode_ids": episode_ids}


@app.post("/v1/memory/workflow", dependencies=[Depends(verify_api_key)])
async def store_workflow(req: WorkflowRequest):
    """
    Store a complete agent workflow as readable procedure memories.

    Creates one episode summarising the full task, plus one episode per
    step, all in plain English so recall returns human-readable workflow
    memories rather than raw JSON parameter dumps.

    Returns: { "stored": true, "workflow_episode_id": int, "step_episode_ids": [...] }
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plugin = pool.get(req.agent_id)
    ctx = {"session_id": req.session_id} if req.session_id else {}

    # 1. Store the full-task episode
    task_lines = "\n".join(
        f"- {s.tool_name}: {s.action_summary}" + (f" → {s.result_summary}" if s.result_summary else "")
        for s in req.steps
    )
    task_content = f"Task: {req.task_description}\nSteps:\n{task_lines}"
    task_result = await asyncio.to_thread(
        plugin.store,
        content=task_content,
        importance=req.importance,
        tags=[],
        source="workflow",
        context=ctx,
        memory_type="procedure",
    )
    workflow_episode_id = task_result["episode_id"]

    # 2. Store individual step episodes
    step_episode_ids: List[int] = []
    for s in req.steps:
        step_content = f"Agent used {s.tool_name}: {s.action_summary}"
        if s.result_summary:
            step_content += f". Result: {s.result_summary}"
        step_result = await asyncio.to_thread(
            plugin.store,
            content=step_content,
            importance=0.65,
            tags=[],
            source="tool_call",
            context=ctx,
            memory_type="procedure",
        )
        if step_result.get("stored"):
            step_episode_ids.append(step_result["episode_id"])

    return {
        "stored": True,
        "workflow_episode_id": workflow_episode_id,
        "step_episode_ids": step_episode_ids,
    }


@app.post("/v1/memory/inject", dependencies=[Depends(verify_api_key), Depends(rate_limit_dep)])
async def inject_memories(req: InjectRequest):
    """Return top-K memories for injection, excluding already-seen episode IDs."""
    plugin = pool.get(req.agent_id)
    results = await asyncio.to_thread(
        plugin.recall,
        query=req.query,
        top_k=(req.top_k or 6) + len(req.already_seen_ids or []),
    )
    # Filter out already-seen IDs and apply min_score threshold
    seen = set(req.already_seen_ids or [])
    min_score = req.min_score if req.min_score is not None else 0.25
    filtered = [
        r for r in results
        if r.get("id") not in seen and (r.get("score") or r.get("similarity") or 0) >= min_score
    ][:req.top_k or 6]
    return {"results": filtered, "injected_count": len(filtered)}


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# =========================================================
# ENTRYPOINT
# =========================================================

def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "arn_v9.api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("ARN_PORT", "8742")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
