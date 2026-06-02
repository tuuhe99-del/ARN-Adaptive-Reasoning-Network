"""
ARN v9 Embedding Engine
========================
Default model: all-MiniLM-L6-v2 (384-dim, ~22MB, Pi-friendly).

To use a custom model, pass embedding_fn to EmbeddingEngine:
    def my_embed(texts: list[str]) -> np.ndarray: ...
    engine = EmbeddingEngine(embedding_fn=my_embed)
"""

import os
import numpy as np
from typing import List, Optional, Union, Callable
import hashlib
import logging
import re
from pathlib import Path

# Suppress noisy HuggingFace/safetensors warnings before any ML imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['SAFETENSORS_FAST_GPU'] = '0'
import warnings as _w
_w.filterwarnings('ignore', message='.*Unauthenticated.*')
_w.filterwarnings('ignore', message='.*huggingface.*')
_w.filterwarnings('ignore', category=FutureWarning)
for _name in ('sentence_transformers', 'transformers', 'huggingface_hub', 
              'safetensors', 'huggingface_hub.utils'):
    logging.getLogger(_name).setLevel(logging.ERROR)

logger = logging.getLogger("arn.embeddings")


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
_EMBEDDING_DIM = 384
_CAL_LOW = 0.28
_CAL_HIGH = 0.55


class EmbeddingEngine:
    """Semantic embedding engine. Default model: all-MiniLM-L6-v2 (384-dim).

    Args:
        use_model: if False, uses a lexical hash fallback (unit tests only)
        cache_size: LRU cache size for encoded strings
        embedding_fn: optional callable(texts: list[str]) -> np.ndarray
                      to replace the default sentence-transformers model
    """

    def __init__(self, use_model: bool = True, cache_size: int = 1024,
                 embedding_fn: Optional[Callable] = None):
        self.embedding_dim = _EMBEDDING_DIM
        self._model = None
        self._embedding_fn = embedding_fn
        self._use_model = use_model
        self._cache: dict = {}
        self._cache_order: list = []
        self._cache_size = cache_size
        self._encode_count = 0
        self._cache_hits = 0
        self._degraded_warned = False
        self._calibrator = SimilarityCalibrator(
            fixed_low=_CAL_LOW,
            fixed_high=_CAL_HIGH,
        )

        if use_model and embedding_fn is None:
            self._load_model()
    
    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        try:
            import warnings
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            warnings.filterwarnings('ignore', message='.*Unauthenticated.*')
            warnings.filterwarnings('ignore', message='.*huggingface.*')

            import logging as _log
            for noisy in ('sentence_transformers', 'transformers',
                         'huggingface_hub', 'safetensors'):
                _log.getLogger(noisy).setLevel(_log.ERROR)

            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            model_name_or_path = self._resolve_local_model_path(MODEL_NAME)
            local_files_only = model_name_or_path != MODEL_NAME
            
            # Suppress ALL model-load noise including C-level stderr writes
            # from safetensors (BertModel LOAD REPORT). Python-level redirects
            # don't catch these because they write to the raw fd, not sys.stderr.
            import sys
            _old_out, _old_err = sys.stdout, sys.stderr
            _devnull = os.open(os.devnull, os.O_WRONLY)
            _old_stderr_fd = os.dup(2)
            _old_stdout_fd = os.dup(1)
            os.dup2(_devnull, 2)
            os.dup2(_devnull, 1)
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                self._model = SentenceTransformer(
                    model_name_or_path,
                    device='cpu',
                    local_files_only=local_files_only
                )
            finally:
                # Restore everything
                sys.stdout.close()
                sys.stderr.close()
                os.dup2(_old_stderr_fd, 2)
                os.dup2(_old_stdout_fd, 1)
                os.close(_devnull)
                os.close(_old_stderr_fd)
                os.close(_old_stdout_fd)
                sys.stdout = _old_out
                sys.stderr = _old_err
            logger.info(f"Loaded {MODEL_NAME} (dim={_EMBEDDING_DIM})")
        except ImportError:
            self._model = None
            self._use_model = False
            logger.critical(
                "sentence-transformers is NOT installed. "
                "ARN is running in DEGRADED MODE with lexical hash vectors. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            self._model = None
            self._use_model = False
            logger.critical(
                f"Failed to load embedding model '{MODEL_NAME}': {e}. "
                "ARN is running in DEGRADED MODE with lexical hash vectors."
            )

    def _resolve_local_model_path(self, model_name: str) -> str:
        """
        Resolve a Hugging Face model id to an already-cached snapshot path.

        SentenceTransformer may still make network metadata requests for
        optional files such as adapter_config.json even when core model files
        are cached. In offline agent environments that turns a healthy cache
        into a hard failure, so prefer a local snapshot when it exists.
        """
        cache_home = Path(
            os.environ.get(
                "HF_HOME",
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            )
        )
        repo_cache = cache_home / "hub" / f"models--{model_name.replace('/', '--')}"
        refs_main = repo_cache / "refs" / "main"
        snapshots_dir = repo_cache / "snapshots"

        candidates = []
        try:
            if refs_main.exists():
                revision = refs_main.read_text(encoding="utf-8").strip()
                if revision:
                    candidates.append(snapshots_dir / revision)
            if snapshots_dir.exists():
                candidates.extend(
                    sorted(
                        (p for p in snapshots_dir.iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                )
        except OSError:
            return model_name

        required_any = ("model.safetensors", "pytorch_model.bin", "model.onnx")
        for candidate in candidates:
            if not (candidate / "config.json").exists():
                continue
            if not (candidate / "modules.json").exists():
                continue
            if not any((candidate / filename).exists() for filename in required_any):
                continue
            return str(candidate)

        return model_name
    
    @property
    def is_degraded(self) -> bool:
        """True if running without real embeddings."""
        return self._model is None and self._embedding_fn is None

    def encode(self, text: str, mode: str = 'passage') -> np.ndarray:
        """Encode a single text to a normalized 384-dim vector.

        mode is accepted for API compatibility but MiniLM-L6-v2 uses no prefixes.
        """
        cache_key = text[:500]
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()

        self._encode_count += 1

        if self._embedding_fn is not None:
            vecs = self._embedding_fn([text])
            vec = np.asarray(vecs[0], dtype=np.float32)
        elif self._model is not None:
            vec = self._model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0].astype(np.float32)
        else:
            if not self._degraded_warned:
                logger.info(
                    "encode() called without embedding model — using lexical hash fallback. "
                    "Recall quality is reduced, but deterministic."
                )
                self._degraded_warned = True
            vec = self._hash_encode(text)

        self._cache[cache_key] = vec.copy()
        self._cache_order.append(cache_key)
        if len(self._cache_order) > self._cache_size:
            evict_key = self._cache_order.pop(0)
            self._cache.pop(evict_key, None)

        return vec

    def encode_batch(self, texts: List[str], mode: str = 'passage') -> np.ndarray:
        """Encode multiple texts. Returns (N, dim) float32 array."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = text[:500]
            if cache_key in self._cache:
                results[i] = self._cache[cache_key].copy()
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            self._encode_count += len(uncached_texts)
            if self._embedding_fn is not None:
                vecs = np.asarray(self._embedding_fn(uncached_texts), dtype=np.float32)
            elif self._model is not None:
                vecs = self._model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                ).astype(np.float32)
            else:
                vecs = np.array([self._hash_encode(t) for t in uncached_texts])

            for local_i, (idx, vec) in enumerate(zip(uncached_indices, vecs)):
                results[idx] = vec
                cache_key = uncached_texts[local_i][:500]
                self._cache[cache_key] = vec.copy()
                self._cache_order.append(cache_key)

        while len(self._cache_order) > self._cache_size:
            evict_key = self._cache_order.pop(0)
            self._cache.pop(evict_key, None)

        return np.array(results, dtype=np.float32)
    
    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors."""
        return float(np.dot(vec_a, vec_b))
    
    def batch_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute similarity between a query vector and multiple candidates."""
        if len(candidates) == 0:
            return np.array([], dtype=np.float32)
        return candidates @ query
    
    def _hash_encode(self, text: str) -> np.ndarray:
        """
        Deterministic lexical hash encoding used when the transformer model
        cannot load.

        This is not a semantic embedding, but it is intentionally better than
        one random vector per full string: shared terms and short phrases land
        in shared dimensions, so recall remains usable for exact/near keyword
        matches during offline or model-cache failures.
        """
        normalized = text.lower()
        tokens = re.findall(r"[a-z0-9_./:-]{2,}", normalized)
        features = list(tokens)
        features.extend(" ".join(tokens[i:i + 2]) for i in range(max(0, len(tokens) - 1)))
        features.extend(" ".join(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2)))

        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.0 + min(len(feature), 32) / 64.0
            vec[bucket] += sign * weight

        if not np.any(vec):
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.embedding_dim
            vec[bucket] = 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    
    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            'model_loaded': self._model is not None or self._embedding_fn is not None,
            'degraded': self.is_degraded,
            'model_name': MODEL_NAME if self._embedding_fn is None else 'custom',
            'embedding_dim': self.embedding_dim,
            'total_encodes': self._encode_count,
            'cache_hits': self._cache_hits,
            'cache_size': len(self._cache),
            'cache_hit_rate': (
                self._cache_hits / (self._encode_count + self._cache_hits)
                if (self._encode_count + self._cache_hits) > 0 else 0.0
            ),
        }
    
    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()

    def calibrate_similarity(self, raw_similarity: float) -> float:
        """Convert a raw cosine similarity to a calibrated confidence."""
        self._calibrator.record(raw_similarity)
        return self._calibrator.calibrate(raw_similarity)

    def confidence_tier(self, raw_similarity: float) -> str:
        """Return 'high'/'medium'/'low' based on calibrated score."""
        conf = self.calibrate_similarity(raw_similarity)
        return self._calibrator.tier(conf)

    def get_calibrator_stats(self) -> dict:
        """Return calibration statistics."""
        return self._calibrator.get_stats()


# =========================================================
# SIMILARITY CALIBRATION
# =========================================================

class SimilarityCalibrator:
    """
    Calibrate raw cosine similarity scores into probability-like confidences.

    Uses fixed model-specific bounds (cal_low/cal_high from MODEL_CONFIGS)
    for a sigmoid that maps raw similarity → calibrated confidence [0, 1].
    observe() tracks history for stats but does not adjust the sigmoid —
    the old adaptive-percentile approach drifted when recall history was
    skewed (small DB, high-sim queries), producing near-zero confidence
    for clearly relevant memories.

    Usage:
        calibrator = SimilarityCalibrator(fixed_low=0.28, fixed_high=0.55)
        calibrator.record(raw_sim)               # stats only, no drift
        conf = calibrator.calibrate(raw_sim)     # 0.0 .. 1.0
        tier = calibrator.tier(conf)             # 'high' | 'medium' | 'low'
    """

    def __init__(self, window_size: int = 2000,
                 fixed_low: float = 0.28,
                 fixed_high: float = 0.55):
        self._window_size = window_size
        self._fixed_low = fixed_low
        self._fixed_high = max(fixed_high, fixed_low + 0.05)
        self._history: list = []
        # Pre-compute sigmoid bounds from fixed values
        self._low_thresh: float = self._fixed_low
        self._high_thresh: float = self._fixed_high

    def record(self, raw_similarity: float):
        """Records the similarity score for statistical tracking.
        
        Does not adapt calibration parameters.
        """
        self._history.append(float(raw_similarity))
        if len(self._history) > self._window_size:
            self._history.pop(0)

    def calibrate(self, raw_similarity: float) -> float:
        """
        Map raw similarity to a calibrated confidence in [0, 1].
        Uses a sigmoid centered at the midpoint of [fixed_low, fixed_high].
        The bounds are set from the model config at construction time and do
        not drift with observed data (the old adaptive approach caused
        near-zero confidence for relevant memories when recall history was
        skewed toward high-similarity queries in small databases).
        """
        lo = self._low_thresh
        hi = self._high_thresh
        mid = (lo + hi) / 2.0
        scale = max(0.01, (hi - lo) / 3.0)
        conf = 1.0 / (1.0 + np.exp(-(raw_similarity - mid) / scale))
        return float(conf)

    def tier(self, calibrated_confidence: float) -> str:
        """Return confidence tier string."""
        if calibrated_confidence >= 0.70:
            return 'high'
        elif calibrated_confidence >= 0.40:
            return 'medium'
        return 'low'

    def get_stats(self) -> dict:
        return {
            'observed_count': len(self._history),
            'fixed_low': self._fixed_low,
            'fixed_high': self._fixed_high,
            'low_threshold': self._low_thresh,
            'high_threshold': self._high_thresh,
        }


# =========================================================
# MODULE-LEVEL CONSTANTS (backwards compatible)
# =========================================================

EMBEDDING_DIM = _EMBEDDING_DIM
