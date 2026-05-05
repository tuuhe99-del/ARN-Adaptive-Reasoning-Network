"""
ARN Embedding Engine
========================
Multi-model embedding support for low-RAM devices, laptops, desktops, and
quality-focused retrieval.

Set a model with:
  ARN_EMBEDDING_TIER=nano arn check
  arn models switch --tier base --download
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
from typing import List, Optional

import numpy as np

from arn.config import get_default_tier

logger = logging.getLogger("arn.embeddings")

MODEL_CONFIGS = {
    "nano": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "quality": "good",
        "speed": "fastest",
        "approx_ram_mb": 120,
        "approx_disk_mb": 90,
        "hardware": "Raspberry Pi 5, low-RAM VPS, older laptops",
        "notes": "Best default when RAM is tight. Fast and reliable for agent memory.",
        "query_prefix": "",
        "passage_prefix": "",
        "low_conf_threshold": 0.40,
        "high_conf_threshold": 0.55,
        # Consolidation clustering threshold calibrated for MiniLM score distribution.
        # MiniLM scores spread wide (unrelated ~0.1, related ~0.5), so 0.45 safely
        # groups genuinely similar episodes without over-merging.
        "consolidation_similarity_threshold": 0.45,
    },
    "small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "quality": "better",
        "speed": "fast",
        "approx_ram_mb": 250,
        "approx_disk_mb": 135,
        "hardware": "Pi 5 with spare RAM, budget laptops, small agents",
        "notes": "Better retrieval than nano while keeping the same 384-dim storage size.",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "low_conf_threshold": 0.52,
        "high_conf_threshold": 0.64,
        # BGE-small scores are tighter than MiniLM; raise threshold slightly
        # to avoid merging episodes that are merely topically adjacent.
        "consolidation_similarity_threshold": 0.55,
    },
    "balanced": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "quality": "strong",
        "speed": "medium",
        "approx_ram_mb": 550,
        "approx_disk_mb": 420,
        "hardware": "modern laptop/desktop with 8GB+ RAM",
        "notes": "Balanced general-purpose option. Uses a separate vector store from 384-dim tiers.",
        "query_prefix": "",
        "passage_prefix": "",
        "low_conf_threshold": 0.40,
        "high_conf_threshold": 0.55,
        # mpnet scores similar to MiniLM in spread; 0.50 is a safe merge boundary.
        "consolidation_similarity_threshold": 0.50,
    },
    "base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "quality": "very strong",
        "speed": "medium",
        "approx_ram_mb": 650,
        "approx_disk_mb": 440,
        "hardware": "desktop/laptop, Pi only if enough free RAM",
        "notes": "Best default for quality-focused semantic recall.",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "low_conf_threshold": 0.55,
        "high_conf_threshold": 0.65,
        # BGE-base produces compressed scores (unrelated ~0.35, related ~0.65).
        # Use a higher threshold so only strongly similar episodes consolidate.
        "consolidation_similarity_threshold": 0.62,
    },
    "base-e5": {
        "name": "intfloat/e5-base-v2",
        "dim": 768,
        "quality": "very strong",
        "speed": "medium",
        "approx_ram_mb": 650,
        "approx_disk_mb": 440,
        "hardware": "desktop/laptop, retrieval-heavy agents",
        "notes": "Good asymmetric query/passage model. Requires query:/passage: prefixes.",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "low_conf_threshold": 0.78,
        "high_conf_threshold": 0.85,
        # e5 scores are very compressed (unrelated ~0.70+). Use a high threshold
        # to avoid collapsing distinct memories into the same semantic node.
        "consolidation_similarity_threshold": 0.82,
    },
    "large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "quality": "highest",
        "speed": "slow",
        "approx_ram_mb": 1400,
        "approx_disk_mb": 1300,
        "hardware": "desktop/server with 16GB+ RAM",
        "notes": "Highest quality option, not recommended for a busy Pi.",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "low_conf_threshold": 0.55,
        "high_conf_threshold": 0.65,
        # BGE-large same distribution as bge-base; match its consolidation threshold.
        "consolidation_similarity_threshold": 0.62,
    },
}

ALIASES = {
    "mini": "nano",
    "pi": "nano",
    "pi5": "nano",
    "mpnet": "balanced",
    "bge": "base",
    "e5": "base-e5",
}


def normalize_tier(tier: str | None) -> str:
    value = (tier or get_default_tier("nano") or "nano").strip().lower()
    value = ALIASES.get(value, value)
    if value not in MODEL_CONFIGS:
        raise ValueError(f"Unknown embedding tier '{tier}'. Available: {', '.join(MODEL_CONFIGS)}")
    return value


def get_total_ram_mb() -> int | None:
    """Best-effort physical RAM detection without extra dependencies."""
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 / 1024)
    except Exception:
        pass
    if platform.system().lower().startswith("win"):
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys / 1024 / 1024)
        except Exception:
            return None
    return None


def recommend_tier(total_ram_mb: int | None = None) -> str:
    ram = total_ram_mb if total_ram_mb is not None else get_total_ram_mb()
    if ram is None:
        return "nano"
    if ram < 4096:
        return "nano"
    if ram < 8192:
        return "small"
    if ram < 12288:
        return "base"
    if ram < 16384:
        return "balanced"
    if ram < 20480:
        return "base-e5"
    return "large"


def model_table() -> list[dict]:
    rows = []
    recommended = recommend_tier()
    for tier, cfg in MODEL_CONFIGS.items():
        rows.append({
            "tier": tier,
            "model": cfg["name"],
            "dim": cfg["dim"],
            "quality": cfg["quality"],
            "speed": cfg["speed"],
            "approx_ram_mb": cfg["approx_ram_mb"],
            "approx_disk_mb": cfg["approx_disk_mb"],
            "hardware": cfg["hardware"],
            "recommended_here": tier == recommended,
            "notes": cfg["notes"],
        })
    return rows


def download_model(tier: str, cache_folder: str | None = None) -> dict:
    tier = normalize_tier(tier)
    cfg = MODEL_CONFIGS[tier]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        return {"ok": False, "tier": tier, "model": cfg["name"], "error": f"sentence-transformers is not installed or broken: {exc}"}
    try:
        model = SentenceTransformer(cfg["name"], device="cpu", cache_folder=cache_folder)
        # Force a tiny encode so failures happen during install, not later in OpenClaw.
        vec = model.encode(["ARN model download verification"], normalize_embeddings=True, show_progress_bar=False)[0]
        return {"ok": True, "tier": tier, "model": cfg["name"], "dim": int(len(vec)), "cache_folder": cache_folder or "default"}
    except Exception as exc:
        return {"ok": False, "tier": tier, "model": cfg["name"], "error": str(exc)}


def _safe_default_tier() -> str:
    value = get_default_tier("nano")
    value = ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if value not in MODEL_CONFIGS:
        logger.warning("Invalid ARN embedding tier %r; falling back to 'nano'", value)
        return "nano"
    return value


DEFAULT_TIER = _safe_default_tier()


class EmbeddingEngine:
    """Semantic embedding engine with switchable model tiers."""

    def __init__(self, use_model: bool = True, cache_size: int = 1024, tier: Optional[str] = None):
        self._tier = normalize_tier(tier)
        self._config = MODEL_CONFIGS[self._tier]
        self.embedding_dim = self._config["dim"]
        self._model = None
        self._use_model = use_model
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size
        self._encode_count = 0
        self._cache_hits = 0
        self._degraded_warned = False
        if use_model:
            self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (%s): %s", self._tier, self._config["name"])
            self._model = SentenceTransformer(self._config["name"], device="cpu")
            logger.info("Loaded %s model — dim=%s", self._tier, self.embedding_dim)
        except ImportError:
            self._model = None
            self._use_model = False
            logger.critical("sentence-transformers is not installed. ARN is in degraded hash-vector mode.")
        except Exception as exc:
            self._model = None
            self._use_model = False
            logger.critical("Failed to load embedding model %s: %s", self._config["name"], exc)

    @property
    def is_degraded(self) -> bool:
        return self._model is None

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def model_name(self) -> str:
        return self._config["name"]

    def _prefix_for_mode(self, mode: str) -> str:
        if mode == "query":
            return self._config["query_prefix"]
        if mode == "passage":
            return self._config["passage_prefix"]
        return ""

    def encode(self, text: str, mode: str = "passage") -> np.ndarray:
        prefix = self._prefix_for_mode(mode)
        full_text = prefix + text if prefix else text
        cache_key = f"{self._tier}:{mode}:{hashlib.sha256(full_text.encode()).hexdigest()[:16]}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()
        self._encode_count += 1
        if self._model is not None:
            vec = self._model.encode([full_text], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        else:
            if not self._degraded_warned:
                logger.warning("Using degraded hash vectors. Install/download an embedding model for real recall.")
                self._degraded_warned = True
            vec = self._hash_encode(full_text)
        self._cache[cache_key] = vec.copy()
        self._cache_order.append(cache_key)
        while len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return vec

    def encode_batch(self, texts: List[str], mode: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        prefix = self._prefix_for_mode(mode)
        prefixed = [prefix + t if prefix else t for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, full_text in enumerate(prefixed):
            cache_key = f"{self._tier}:{mode}:{hashlib.sha256(full_text.encode()).hexdigest()[:16]}"
            if cache_key in self._cache:
                results[i] = self._cache[cache_key].copy()
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(full_text)
        if uncached_texts:
            self._encode_count += len(uncached_texts)
            if self._model is not None:
                vecs = self._model.encode(uncached_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32).astype(np.float32)
            else:
                vecs = np.array([self._hash_encode(t) for t in uncached_texts], dtype=np.float32)
            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                cache_key = f"{self._tier}:{mode}:{hashlib.sha256(prefixed[idx].encode()).hexdigest()[:16]}"
                self._cache[cache_key] = vec.copy()
                self._cache_order.append(cache_key)
        while len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return np.array(results, dtype=np.float32)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    def batch_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([], dtype=np.float32)
        return candidates @ query

    def _hash_encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(f"{self._tier}:{text}".encode()).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def get_stats(self) -> dict:
        return {
            "model_loaded": self._model is not None,
            "degraded": self.is_degraded,
            "tier": self._tier,
            "model_name": self._config["name"],
            "embedding_dim": self.embedding_dim,
            "quality": self._config["quality"],
            "speed": self._config["speed"],
            "approx_ram_mb": self._config["approx_ram_mb"],
            "approx_disk_mb": self._config["approx_disk_mb"],
            "uses_asymmetric_prefixes": bool(self._config["query_prefix"] or self._config["passage_prefix"]),
            "total_encodes": self._encode_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / (self._encode_count + self._cache_hits) if (self._encode_count + self._cache_hits) > 0 else 0.0,
        }

    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()


EMBEDDING_DIM = MODEL_CONFIGS[normalize_tier(DEFAULT_TIER)]["dim"]
