"""
ARN v9 Embedding Engine
========================
Supports multiple embedding models with different size/quality tradeoffs.

Model options (set via ARN_EMBEDDING_MODEL env var or EmbeddingEngine param):

| Model tier      | Model                           | Size    | Dim  | MTEB  | Notes |
|-----------------|---------------------------------|---------|------|-------|-------|
| nano (default)  | all-MiniLM-L6-v2                | ~22MB   | 384  | 56.3  | Fast, Pi-friendly |
| small           | all-mpnet-base-v2               | ~420MB  | 768  | 57.8  | Balanced |
| base (RECO)     | BAAI/bge-base-en-v1.5           | ~440MB  | 768  | 63.6  | Best retrieval quality |
| base-e5         | intfloat/e5-base-v2             | ~440MB  | 768  | 61.5  | Good general-purpose |

The "base" tier models use QUERY/PASSAGE prefix asymmetry — queries get
a "Represent this sentence..." or "query:" prefix while stored passages
get different (or no) prefix. This is what actually moves the needle
on the temporal/paraphrase problems, not just raw dimension count.

For Pi 5 deployment:
- nano:  ~90MB RAM, ~30ms/encode
- base:  ~500MB RAM, ~80ms/encode — still viable on 8GB Pi 5
"""

import os
import numpy as np
from typing import List, Optional, Union
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


# =========================================================
# MODEL REGISTRY
# =========================================================

MODEL_CONFIGS = {
    'nano': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dim': 384,
        'query_prefix': '',
        'passage_prefix': '',
        'approx_ram_mb': 90,
        # Empirically calibrated from stress test data
        # "low" means: even the top match is weak — caller should be skeptical
        'low_conf_threshold': 0.40,
        'high_conf_threshold': 0.55,
        # Sigmoid bounds for calibrator: span = [cal_low, cal_high]
        # cal_low is roughly "minimum relevance" (below → clearly irrelevant)
        # cal_high equals high_conf_threshold
        'cal_low': 0.28,
        'cal_high': 0.55,
    },
    'nano2': {
        # Snowflake Arctic Embed XS: same 22M params / 384 dims as MiniLM-L6-v2
        # but +19.5% higher retrieval NDCG@10 (50.15 vs 41.95). Drop-in upgrade
        # for Pi 5 with no RAM increase. Requires sentence-transformers >= 2.3.0.
        'name': 'snowflake/snowflake-arctic-embed-xs',
        'dim': 384,
        'query_prefix': '',
        'passage_prefix': '',
        'approx_ram_mb': 90,
        # Arctic Embed compresses scores differently; thresholds tuned from
        # MTEB retrieval score distributions on English passage pairs.
        'low_conf_threshold': 0.45,
        'high_conf_threshold': 0.60,
        'cal_low': 0.33,
        'cal_high': 0.60,
    },
    'small': {
        'name': 'sentence-transformers/all-mpnet-base-v2',
        'dim': 768,
        'query_prefix': '',
        'passage_prefix': '',
        'approx_ram_mb': 420,
        'low_conf_threshold': 0.40,
        'high_conf_threshold': 0.55,
        'cal_low': 0.28,
        'cal_high': 0.55,
    },
    'base': {
        'name': 'BAAI/bge-base-en-v1.5',
        'dim': 768,
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
        'approx_ram_mb': 440,
        # bge scores compress higher
        'low_conf_threshold': 0.55,
        'high_conf_threshold': 0.65,
        'cal_low': 0.45,
        'cal_high': 0.65,
    },
    'base-e5': {
        'name': 'intfloat/e5-base-v2',
        'dim': 768,
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'approx_ram_mb': 440,
        'low_conf_threshold': 0.78,
        'high_conf_threshold': 0.85,
        'cal_low': 0.68,
        'cal_high': 0.85,
    },
}

# Default tier — can be overridden via env or parameter
DEFAULT_TIER = os.environ.get('ARN_EMBEDDING_TIER', 'nano')


class EmbeddingEngine:
    """
    Semantic embedding engine with tiered model support and query/passage
    asymmetry for retrieval quality.
    """
    
    def __init__(self, use_model: bool = True, cache_size: int = 1024,
                 tier: Optional[str] = None):
        """
        Args:
            use_model: if False, uses hash fallback (for unit tests only)
            cache_size: LRU cache size for encoded strings
            tier: one of 'nano', 'small', 'base', 'base-e5'.
                  Defaults to ARN_EMBEDDING_TIER env or 'nano'.
        """
        self._tier = tier or DEFAULT_TIER
        if self._tier not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown tier '{self._tier}'. "
                f"Available: {list(MODEL_CONFIGS.keys())}"
            )
        
        self._config = MODEL_CONFIGS[self._tier]
        self.embedding_dim = self._config['dim']
        
        self._model = None
        self._use_model = use_model
        self._cache: dict = {}
        self._cache_order: list = []
        self._cache_size = cache_size
        self._encode_count = 0
        self._cache_hits = 0
        self._degraded_warned = False
        self._calibrator = SimilarityCalibrator(
            fixed_low=self._config['cal_low'],
            fixed_high=self._config['cal_high'],
        )
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Lazy-load the configured sentence transformer model."""
        try:
            # Suppress noisy HuggingFace warnings that alarm non-technical users
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
            logger.info(f"Loading embedding model ({self._tier}): {self._config['name']}")
            model_name_or_path = self._resolve_local_model_path(self._config['name'])
            local_files_only = model_name_or_path != self._config['name']
            
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
            logger.info(
                f"Loaded {self._tier} model — dim={self._config['dim']}, "
                f"~{self._config['approx_ram_mb']}MB RAM"
            )
        except ImportError:
            self._model = None
            self._use_model = False
            logger.critical(
                "sentence-transformers is NOT installed. "
                "ARN is running in DEGRADED MODE with lexical hash vectors. "
                "Recall quality is reduced until sentence-transformers is available. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            self._model = None
            self._use_model = False
            logger.critical(
                f"Failed to load embedding model '{self._config['name']}': {e}. "
                "ARN is running in DEGRADED MODE with lexical hash vectors. "
                "Recall quality is reduced until the model loads correctly."
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
        """True if running without real embeddings. Memory is non-functional."""
        return self._model is None
    
    @property
    def tier(self) -> str:
        return self._tier
    
    def _prefix_for_mode(self, mode: str) -> str:
        """
        Return the appropriate prefix for the given mode.
        
        Args:
            mode: 'query' (for recall queries) or 'passage' (for stored content)
        """
        if mode == 'query':
            return self._config['query_prefix']
        elif mode == 'passage':
            return self._config['passage_prefix']
        return ''
    
    def encode(self, text: str, mode: str = 'passage') -> np.ndarray:
        """
        Encode a single text string to a normalized vector.
        
        Args:
            text: the text to encode
            mode: 'query' (retrieval query) or 'passage' (stored content).
                  Some models (bge, e5) use different prefixes for each.
        """
        prefix = self._prefix_for_mode(mode)
        full_text = prefix + text if prefix else text
        
        cache_key = f"{mode}:{full_text[:500]}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()
        
        self._encode_count += 1
        
        if self._model is not None:
            vec = self._model.encode(
                [full_text],
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
            vec = self._hash_encode(full_text)
        
        # LRU cache
        self._cache[cache_key] = vec.copy()
        self._cache_order.append(cache_key)
        if len(self._cache_order) > self._cache_size:
            evict_key = self._cache_order.pop(0)
            self._cache.pop(evict_key, None)
        
        return vec
    
    def encode_batch(self, texts: List[str], mode: str = 'passage') -> np.ndarray:
        """
        Encode multiple texts at once. All texts use the same mode.
        Returns (N, dim) array of normalized vectors.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        prefix = self._prefix_for_mode(mode)
        prefixed = [prefix + t if prefix else t for t in texts]
        
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for i, full_text in enumerate(prefixed):
            cache_key = f"{mode}:{full_text[:500]}"
            if cache_key in self._cache:
                results[i] = self._cache[cache_key].copy()
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(full_text)
        
        if uncached_texts:
            self._encode_count += len(uncached_texts)
            if self._model is not None:
                vecs = self._model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                ).astype(np.float32)
            else:
                vecs = np.array([self._hash_encode(t) for t in uncached_texts])
            
            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                full_text = prefixed[idx]
                cache_key = f"{mode}:{full_text[:500]}"
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
            'model_loaded': self._model is not None,
            'degraded': self.is_degraded,
            'tier': self._tier,
            'model_name': self._config['name'],
            'embedding_dim': self.embedding_dim,
            'approx_ram_mb': self._config['approx_ram_mb'],
            'uses_asymmetric_prefixes': bool(
                self._config['query_prefix'] or self._config['passage_prefix']
            ),
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

# Legacy constant — kept for backwards compat, but code should query
# engine.embedding_dim since it varies by tier now
EMBEDDING_DIM = MODEL_CONFIGS[DEFAULT_TIER]['dim']
