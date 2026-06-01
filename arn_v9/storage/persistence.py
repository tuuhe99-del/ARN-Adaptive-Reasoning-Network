"""
ARN v9 Persistence Layer
=========================
SQLite for metadata + memory-mapped NumPy arrays for vectors.

Design choices:
- SQLite WAL mode for crash safety and concurrent reads
- Memory-mapped vectors for zero-copy access (OS handles paging)
- Atomic writes via temp-file-then-rename for vector files
- Batch operations to minimize SD card wear

Storage layout:
  {data_dir}/
    arn_metadata.db          # SQLite: all metadata
    episodic_vectors.npy     # memmap: N x 384 float32
    semantic_vectors.npy     # memmap: M x 384 float32
"""

import sqlite3
import numpy as np
import os
import shutil
import json
import time
import hashlib
import logging
import threading
import tempfile
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path


class _ThreadLocalConnection:
    """Thread-local SQLite connection wrapper.
    
    SQLite connections cannot be shared across threads safely.
    Each thread gets its own connection to the same database.
    """
    def __init__(self, db_path: Path, row_factory=sqlite3.Row):
        self.db_path = db_path
        self.row_factory = row_factory
        self._local = threading.local()
    
    def get(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=2000")
            conn.row_factory = self.row_factory
            self._local.conn = conn
        return conn

    def commit(self):
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.commit()

    def close(self):
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

from ..core.embeddings import EMBEDDING_DIM

logger = logging.getLogger("arn.storage")

# Schema version for migrations
SCHEMA_VERSION = 4


class StorageEngine:
    """
    Persistent storage backend for ARN v9.
    
    Handles:
    - Episode metadata and vectors
    - Semantic memory metadata and vectors  
    - System configuration and stats
    - Crash-safe writes with WAL mode
    """
    
    def __init__(self, data_dir: str, max_episodes: int = 4096,
                 max_semantics: int = 2048, embedding_dim: int = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "arn_metadata.db"
        self.episodic_vec_path = self.data_dir / "episodic_vectors.npy"
        self.semantic_vec_path = self.data_dir / "semantic_vectors.npy"
        
        self.max_episodes = max_episodes
        self.max_semantics = max_semantics
        
        # Dimension handling: use provided, else default to legacy constant.
        # If existing vector files exist with a different dim, respect THAT.
        if embedding_dim is None:
            embedding_dim = EMBEDDING_DIM
        
        # If vectors already exist on disk, infer the dim from them
        # to preserve backward compatibility with existing deployments.
        if self.episodic_vec_path.exists():
            try:
                existing = np.load(str(self.episodic_vec_path), mmap_mode='r')
                if existing.ndim != 2:
                    raise ValueError(f"expected 2D vector store, got shape={existing.shape}")
                existing_dim = existing.shape[1]
                if existing_dim != embedding_dim:
                    logger.warning(
                        f"Existing vectors have dim={existing_dim} but engine "
                        f"configured for dim={embedding_dim}. Using existing dim "
                        f"to preserve data. Delete the data directory to switch models."
                    )
                    embedding_dim = existing_dim
                del existing  # Close the mmap before reopening below
            except Exception as exc:
                logger.warning(
                    "Could not inspect existing episodic vectors; startup will "
                    f"attempt recovery with default dim={embedding_dim}: {exc}"
                )
        
        self.embedding_dim = embedding_dim

        # Initialize database
        self._conn = _ThreadLocalConnection(self.db_path, row_factory=sqlite3.Row)
        self._init_db()

        # Initialize vector stores
        self._episodic_vectors: Optional[np.ndarray] = None
        self._semantic_vectors: Optional[np.ndarray] = None
        self._init_vectors()

        # Write buffer for batched operations
        self._pending_episode_writes: List[Tuple[int, np.ndarray]] = []
        self._pending_semantic_writes: List[Tuple[int, np.ndarray]] = []
        self._lock = threading.Lock()

        # Optional sqlite-vec accelerator for fast ANN search
        try:
            from arn_v9.storage.vec_accelerator import VecAccelerator
            self._vec_acc = VecAccelerator(self.data_dir, self.embedding_dim)
            if self._vec_acc.available:
                synced = self._vec_acc.sync_from_storage(self)
                logger.info(f"[storage] sqlite-vec accelerator ready ({synced} vectors synced)")
        except Exception as _vec_exc:
            logger.debug(f"[storage] sqlite-vec accelerator skipped: {_vec_exc}")
            self._vec_acc = None

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn.get()
    
    def _init_db(self):
        """Create tables if they don't exist, migrate if needed."""
        conn = self._get_conn()
        
        # Create schema_version table first (needed to check version)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)
        conn.commit()
        
        # Check and run migrations BEFORE creating tables with new columns
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if existing is not None and existing[0] < SCHEMA_VERSION:
            self._migrate_schema(conn, existing[0])
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
            conn.commit()
        
        # Now create all tables (safe for both fresh installs and migrated dbs)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT,
                context_json TEXT DEFAULT '{}',
                importance REAL DEFAULT 0.5,
                prediction_error REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                replay_priority REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                last_accessed REAL,
                consolidated INTEGER DEFAULT 0,
                source TEXT DEFAULT 'user',
                expires_at REAL,
                superseded_by INTEGER,
                invalidated_at REAL,
                user_id TEXT,
                memory_type TEXT DEFAULT 'episode'
            );
            
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec_index INTEGER NOT NULL,
                concept_label TEXT NOT NULL,
                confidence REAL DEFAULT 0.1,
                evidence_count INTEGER DEFAULT 0,
                contradiction_log TEXT DEFAULT '[]',
                schema_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_episodes_importance 
                ON episodes(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_created 
                ON episodes(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_consolidated 
                ON episodes(consolidated);
            CREATE INDEX IF NOT EXISTS idx_episodes_hash
                ON episodes(content_hash);
            CREATE INDEX IF NOT EXISTS idx_episodes_expires
                ON episodes(expires_at);
            CREATE INDEX IF NOT EXISTS idx_episodes_user
                ON episodes(user_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_memory_type
                ON episodes(memory_type);
            CREATE INDEX IF NOT EXISTS idx_semantic_confidence 
                ON semantic_nodes(confidence DESC);
            
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_episode_id INTEGER NOT NULL,
                to_episode_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (from_episode_id) REFERENCES episodes(id),
                FOREIGN KEY (to_episode_id) REFERENCES episodes(id),
                UNIQUE (from_episode_id, to_episode_id, relation_type)
            );

            CREATE INDEX IF NOT EXISTS idx_links_from
                ON memory_links(from_episode_id);
            CREATE INDEX IF NOT EXISTS idx_links_to
                ON memory_links(to_episode_id);
        """)
        
        # Set schema version for fresh installs
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if existing is None:
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        
        conn.commit()
    
    def _migrate_schema(self, conn, from_version: int):
        """Migrate database schema from older versions."""
        if from_version < 2:
            # v1 → v2: add new columns to episodes, create entities tables
            migrations = [
                "ALTER TABLE episodes ADD COLUMN content_hash TEXT",
                "ALTER TABLE episodes ADD COLUMN expires_at REAL",
                "ALTER TABLE episodes ADD COLUMN superseded_by INTEGER",
                "ALTER TABLE episodes ADD COLUMN invalidated_at REAL",
                "ALTER TABLE episodes ADD COLUMN user_id TEXT",
            ]
            for sql in migrations:
                try:
                    conn.execute(sql)
                except Exception:
                    pass  # Column may already exist
            
            # Backfill content_hash for existing episodes
            rows = conn.execute("SELECT id, content FROM episodes WHERE content_hash IS NULL").fetchall()
            for row in rows:
                normalized = ' '.join(row['content'].lower().split())
                h = hashlib.sha256(normalized.encode()).hexdigest()[:16]
                conn.execute("UPDATE episodes SET content_hash = ? WHERE id = ?", (h, row['id']))
            
            # Create indexes for new columns
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_expires ON episodes(expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)",
            ]:
                conn.execute(idx_sql)
            
            logger.info(f"Migrated schema v1 → v2 ({len(rows)} episodes hash-backfilled)")
        
        if from_version < 3:
            # v2 → v3: add memory_type column for typed retrieval
            try:
                conn.execute("ALTER TABLE episodes ADD COLUMN memory_type TEXT DEFAULT 'episode'")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_memory_type ON episodes(memory_type)")
                logger.info("Migrated schema v2 → v3 (memory_type column added)")
            except Exception:
                pass  # Column may already exist

        if from_version < 4:
            # v3 → v4: add memory_links table for manual graph wiring
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_links (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        from_episode_id INTEGER NOT NULL,
                        to_episode_id INTEGER NOT NULL,
                        relation_type TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        FOREIGN KEY (from_episode_id) REFERENCES episodes(id),
                        FOREIGN KEY (to_episode_id) REFERENCES episodes(id),
                        UNIQUE (from_episode_id, to_episode_id, relation_type)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_links_from ON memory_links(from_episode_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_links_to ON memory_links(to_episode_id)")
                logger.info("Migrated schema v3 → v4 (memory_links table added)")
            except Exception:
                pass
    
    def _init_vectors(self):
        """Initialize or load memory-mapped vector files."""
        self._episodic_vectors = self._load_or_create_vectors(
            self.episodic_vec_path, self.max_episodes, "episodic", "episodes"
        )
        self._semantic_vectors = self._load_or_create_vectors(
            self.semantic_vec_path, self.max_semantics, "semantic", "semantic_nodes"
        )

    def _load_or_create_vectors(self, path: Path, capacity: int, label: str,
                                table: str = None) -> np.ndarray:
        """Load a memmap vector store, replacing corrupt files with a fresh store."""
        if path.exists():
            try:
                vectors = np.load(str(path), mmap_mode='r+')
                self._validate_vector_store(vectors, label)
                logger.info(f"Loaded {label} vectors: {vectors.shape}")
                return vectors
            except Exception as exc:
                corrupt_path = self._quarantine_vector_file(path)
                affected_msg = ""
                if table:
                    try:
                        count = self._get_conn().execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                        affected_msg = (
                            f" {count} existing rows will have zero vectors "
                            "until re-embedded."
                        )
                    except Exception:
                        pass
                logger.warning(
                    f"Could not load {label} vectors from {path}: {exc}. "
                    f"Moved corrupt file to {corrupt_path} and creating a fresh "
                    f"store.{affected_msg}"
                )

        vectors = np.zeros((capacity, self.embedding_dim), dtype=np.float32)
        np.save(str(path), vectors)
        return np.load(str(path), mmap_mode='r+')

    def _atomic_save_vectors(self, path: Path, vectors: np.ndarray):
        """Persist a vector store by replacing the active .npy atomically."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                np.save(temp_file, vectors)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(str(temp_path), str(path))
            temp_path = None
            self._fsync_directory(path.parent)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def _fsync_directory(self, path: Path):
        """Best-effort directory fsync so atomic replaces survive power loss."""
        try:
            dir_fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    def _validate_vector_store(self, vectors: np.ndarray, label: str):
        if vectors.ndim != 2:
            raise ValueError(f"{label} vector store must be 2D, got shape={vectors.shape}")
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"{label} vector dim={vectors.shape[1]} does not match "
                f"configured dim={self.embedding_dim}"
            )
        if vectors.dtype != np.float32:
            raise ValueError(f"{label} vector dtype must be float32, got {vectors.dtype}")

    def _quarantine_vector_file(self, path: Path) -> Path:
        suffix = f"{path.suffix}.corrupt-{int(time.time())}"
        corrupt_path = path.with_suffix(suffix)
        counter = 1
        while corrupt_path.exists():
            corrupt_path = path.with_suffix(f"{suffix}-{counter}")
            counter += 1
        path.replace(corrupt_path)
        return corrupt_path
    
    # =========================================================
    # EPISODIC MEMORY OPERATIONS
    # =========================================================
    
    def store_episode(self, content: str, vector: np.ndarray,
                      context: dict = None, importance: float = 0.5,
                      prediction_error: float = 0.0,
                      source: str = 'user',
                      expires_at: float = None,
                      user_id: str = None,
                      memory_type: str = 'episode') -> int:
        """Store a new episodic memory. Returns episode ID."""
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            
            # Content hash for deduplication
            normalized = ' '.join(content.lower().split())
            c_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            
            # Find next available vector index
            # CRITICAL: Use MAX(vec_index)+1, NOT COUNT(*).
            # COUNT-based allocation causes collisions when episodes are
            # consolidated or deleted — new episodes get indices that
            # already belong to other episodes, overwriting their vectors.
            row = conn.execute("SELECT MAX(vec_index) FROM episodes").fetchone()
            vec_index = (row[0] + 1) if row[0] is not None else 0
            
            # Handle overflow
            if vec_index >= self.max_episodes:
                vec_index = self._find_free_episode_slot(conn)
            
            # Store vector
            if vec_index < self._episodic_vectors.shape[0]:
                self._episodic_vectors[vec_index] = vector
            else:
                logger.warning(f"Vector index {vec_index} out of bounds, expanding")
                self._expand_episodic_vectors()
                self._episodic_vectors[vec_index] = vector
            
            # Store metadata
            cursor = conn.execute("""
                INSERT INTO episodes (vec_index, content, content_hash, context_json, 
                                      importance, prediction_error, created_at, source,
                                      expires_at, user_id, memory_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vec_index,
                content,
                c_hash,
                json.dumps(context or {}),
                importance,
                prediction_error,
                now,
                source,
                expires_at,
                user_id,
                memory_type,
            ))
            
            conn.commit()
            ep_id = cursor.lastrowid

            # Sync new episode to sqlite-vec accelerator (best-effort)
            if self._vec_acc and self._vec_acc.available:
                self._vec_acc.upsert(ep_id, vector)

            return ep_id

    def get_episode(self, episode_id: int) -> Optional[dict]:
        """Retrieve episode by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)
    
    def get_all_episodes(self, consolidated: Optional[bool] = None,
                         limit: int = None,
                         memory_type: Optional[str] = None) -> List[dict]:
        """Retrieve episodes with optional filtering."""
        conn = self._get_conn()
        query = "SELECT * FROM episodes"
        params = []
        conditions = []
        
        if consolidated is not None:
            conditions.append("consolidated = ?")
            params.append(int(consolidated))
        
        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(r) for r in rows]
    
    def get_episode_vectors(self, episode_ids: List[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Get vectors for episodes. Returns (vectors_matrix, vec_indices).
        If episode_ids is None, returns all unconsolidated episode vectors.
        """
        conn = self._get_conn()
        
        if episode_ids is None:
            rows = conn.execute(
                "SELECT id, vec_index FROM episodes WHERE consolidated=0"
            ).fetchall()
        else:
            placeholders = ','.join('?' * len(episode_ids))
            rows = conn.execute(
                f"SELECT id, vec_index FROM episodes WHERE id IN ({placeholders})",
                episode_ids
            ).fetchall()
        
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        
        indices = [r['vec_index'] for r in rows]
        ids = [r['id'] for r in rows]
        vectors = self._episodic_vectors[indices].copy()
        return vectors, ids
    
    def update_episode_access(self, episode_id: int):
        """Increment access count and update last_accessed."""
        conn = self._get_conn()
        conn.execute("""
            UPDATE episodes 
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (time.time(), episode_id))
        conn.commit()
    
    def mark_episodes_consolidated(self, episode_ids: List[int]):
        """Mark episodes as consolidated."""
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        conn.execute(
            f"UPDATE episodes SET consolidated = 1 WHERE id IN ({placeholders})",
            episode_ids
        )
        conn.commit()
    
    def delete_episodes(self, episode_ids: List[int]):
        """Delete episodes permanently."""
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        conn.execute(
            f"DELETE FROM episodes WHERE id IN ({placeholders})",
            episode_ids
        )
        conn.commit()
        # Sync deletions to sqlite-vec accelerator (best-effort)
        if self._vec_acc and self._vec_acc.available:
            for ep_id in episode_ids:
                self._vec_acc.delete(ep_id)
    
    def count_episodes(self, consolidated: Optional[bool] = None) -> int:
        conn = self._get_conn()
        if consolidated is None:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated=?",
            (int(consolidated),)
        ).fetchone()[0]
    
    # =========================================================
    # SEMANTIC MEMORY OPERATIONS
    # =========================================================
    
    def store_semantic(self, concept_label: str, vector: np.ndarray,
                       confidence: float = 0.1, evidence_count: int = 1,
                       schema: dict = None) -> int:
        """Store a new semantic memory node. Returns node ID."""
        with self._lock:
            conn = self._get_conn()
            now = time.time()

            # Use MAX(vec_index)+1, not COUNT(*). COUNT causes collisions when
            # nodes are deleted — new entries get indices already owned by others.
            row = conn.execute("SELECT MAX(vec_index) FROM semantic_nodes").fetchone()
            vec_index = (row[0] + 1) if row[0] is not None else 0

            if vec_index >= self._semantic_vectors.shape[0]:
                self._expand_semantic_vectors()

            self._semantic_vectors[vec_index] = vector

            cursor = conn.execute("""
                INSERT INTO semantic_nodes (vec_index, concept_label, confidence,
                                           evidence_count, schema_json, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                vec_index, concept_label, confidence,
                evidence_count, json.dumps(schema or {}), now, now
            ))

            conn.commit()
            return cursor.lastrowid
    
    def update_semantic(self, node_id: int, vector: np.ndarray = None,
                        confidence: float = None, evidence_count: int = None,
                        contradiction_log: list = None, schema: dict = None):
        """Update an existing semantic node."""
        conn = self._get_conn()
        
        if vector is not None:
            row = conn.execute(
                "SELECT vec_index FROM semantic_nodes WHERE id=?", (node_id,)
            ).fetchone()
            if row:
                self._semantic_vectors[row['vec_index']] = vector
        
        updates = []
        params = []
        
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if evidence_count is not None:
            updates.append("evidence_count = ?")
            params.append(evidence_count)
        if contradiction_log is not None:
            updates.append("contradiction_log = ?")
            params.append(json.dumps(contradiction_log))
        if schema is not None:
            updates.append("schema_json = ?")
            params.append(json.dumps(schema))
        
        updates.append("last_updated = ?")
        params.append(time.time())
        params.append(node_id)
        
        conn.execute(
            f"UPDATE semantic_nodes SET {', '.join(updates)} WHERE id = ?",
            params
        )
        conn.commit()
    
    def get_all_semantics(self) -> List[dict]:
        """Retrieve all semantic nodes."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM semantic_nodes ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_semantic(r) for r in rows]
    
    def get_semantic_vectors(self) -> Tuple[np.ndarray, List[int]]:
        """Get all semantic vectors. Returns (matrix, node_ids)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, vec_index FROM semantic_nodes"
        ).fetchall()
        
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        
        indices = [r['vec_index'] for r in rows]
        ids = [r['id'] for r in rows]
        vectors = self._semantic_vectors[indices].copy()
        return vectors, ids
    
    def count_semantics(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM semantic_nodes").fetchone()[0]
    
    def delete_semantics(self, node_ids: List[int]):
        """Delete semantic nodes."""
        conn = self._get_conn()
        placeholders = ','.join('?' * len(node_ids))
        conn.execute(
            f"DELETE FROM semantic_nodes WHERE id IN ({placeholders})",
            node_ids
        )
        conn.commit()
    
    # =========================================================
    # SYSTEM STATE
    # =========================================================
    
    def get_state(self, key: str, default: str = None) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else default
    
    def set_state(self, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            (key, value)
        )
        conn.commit()
    
    # =========================================================
    # MEMORY LINK OPERATIONS
    # =========================================================

    def create_link(self, from_id: int, to_id: int,
                    relation_type: str, confidence: float = 1.0) -> int:
        """Create a directed link between two episodes. Returns link ID.
        If the link already exists, returns the existing link's ID."""
        conn = self._get_conn()
        now = time.time()
        try:
            cursor = conn.execute("""
                INSERT INTO memory_links
                    (from_episode_id, to_episode_id, relation_type, created_at, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (from_id, to_id, relation_type, now, confidence))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            row = conn.execute("""
                SELECT id FROM memory_links
                WHERE from_episode_id=? AND to_episode_id=? AND relation_type=?
            """, (from_id, to_id, relation_type)).fetchone()
            return row['id'] if row else -1

    def get_links_for_episode(self, episode_id: int) -> List[dict]:
        """Return all links where the episode is source or target."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM memory_links
            WHERE from_episode_id=? OR to_episode_id=?
            ORDER BY created_at DESC
        """, (episode_id, episode_id)).fetchall()
        return [self._row_to_link(r) for r in rows]

    def delete_link(self, link_id: int):
        """Delete a link by ID."""
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_links WHERE id=?", (link_id,))
        conn.commit()

    def get_all_links(self) -> List[dict]:
        """Return all links for this agent's storage."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memory_links ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def _row_to_link(self, row) -> dict:
        return {
            'id': row['id'],
            'from_episode_id': row['from_episode_id'],
            'to_episode_id': row['to_episode_id'],
            'relation_type': row['relation_type'],
            'created_at': row['created_at'],
            'confidence': row['confidence'],
        }

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================
    
    def _row_to_episode(self, row) -> dict:
        return {
            'id': row['id'],
            'vec_index': row['vec_index'],
            'content': row['content'],
            'context': json.loads(row['context_json']),
            'importance': row['importance'],
            'prediction_error': row['prediction_error'],
            'access_count': row['access_count'],
            'replay_priority': row['replay_priority'],
            'created_at': row['created_at'],
            'last_accessed': row['last_accessed'],
            'consolidated': bool(row['consolidated']),
            'source': row['source'],
            'content_hash': row['content_hash'] if 'content_hash' in row.keys() else None,
            'expires_at': row['expires_at'] if 'expires_at' in row.keys() else None,
            'superseded_by': row['superseded_by'] if 'superseded_by' in row.keys() else None,
            'invalidated_at': row['invalidated_at'] if 'invalidated_at' in row.keys() else None,
            'user_id': row['user_id'] if 'user_id' in row.keys() else None,
            'memory_type': row['memory_type'] if 'memory_type' in row.keys() else 'episode',
        }
    
    def _row_to_semantic(self, row) -> dict:
        return {
            'id': row['id'],
            'vec_index': row['vec_index'],
            'concept_label': row['concept_label'],
            'confidence': row['confidence'],
            'evidence_count': row['evidence_count'],
            'contradiction_log': json.loads(row['contradiction_log']),
            'schema': json.loads(row['schema_json']),
            'created_at': row['created_at'],
            'last_updated': row['last_updated'],
            'access_count': row['access_count'],
        }
    
    def _find_free_episode_slot(self, conn) -> int:
        """Find a reusable vector slot from consolidated/deleted episodes.
        
        If no free slot exists, evict the oldest episode to make room.
        """
        # Find lowest-importance consolidated episode and reuse its slot
        row = conn.execute("""
            SELECT vec_index FROM episodes 
            WHERE consolidated = 1 
            ORDER BY importance ASC LIMIT 1
        """).fetchone()
        if row:
            return row['vec_index']
        # All slots full — evict the oldest episode, protecting source='api'
        row = conn.execute("""
            SELECT id, vec_index, created_at FROM episodes
            WHERE (source IS NULL OR source != 'api')
            ORDER BY created_at ASC LIMIT 1
        """).fetchone()
        if row is None:
            # Pathological case: all episodes are source='api' — fall back to
            # evicting the oldest regardless of source.
            row = conn.execute("""
                SELECT id, vec_index, created_at FROM episodes
                ORDER BY created_at ASC LIMIT 1
            """).fetchone()
        if row:
            oldest_id = row['id']
            oldest_slot = row['vec_index']
            logger.warning(
                f"Memory capacity reached, evicting oldest episode "
                f"(id={oldest_id}) to make room."
            )
            conn.execute("DELETE FROM episodes WHERE id = ?", (oldest_id,))
            conn.commit()
            return oldest_slot
        # Should never reach here (empty DB but overflow?), fallback to 0
        return 0
    
    def _expand_episodic_vectors(self):
        """Double the episodic vector capacity."""
        old_size = self._episodic_vectors.shape[0]
        new_size = old_size * 2
        new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
        new_vectors[:old_size] = self._episodic_vectors[:]
        self._atomic_save_vectors(self.episodic_vec_path, new_vectors)
        self._episodic_vectors = np.load(
            str(self.episodic_vec_path), mmap_mode='r+'
        )
        logger.info(f"Expanded episodic vectors: {old_size} -> {new_size}")
    
    def _expand_semantic_vectors(self):
        """Double the semantic vector capacity."""
        old_size = self._semantic_vectors.shape[0]
        new_size = old_size * 2
        new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
        new_vectors[:old_size] = self._semantic_vectors[:]
        self._atomic_save_vectors(self.semantic_vec_path, new_vectors)
        self._semantic_vectors = np.load(
            str(self.semantic_vec_path), mmap_mode='r+'
        )
        logger.info(f"Expanded semantic vectors: {old_size} -> {new_size}")
    
    def flush(self):
        """Flush all pending writes to disk."""
        if self._conn:
            self._conn.commit()
        if self._episodic_vectors is not None:
            self._episodic_vectors.flush()
        if self._semantic_vectors is not None:
            self._semantic_vectors.flush()
    
    def close(self):
        """Close all resources."""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def get_storage_stats(self) -> dict:
        """Return storage statistics."""
        db_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        ep_vec_size = os.path.getsize(self.episodic_vec_path) if self.episodic_vec_path.exists() else 0
        sem_vec_size = os.path.getsize(self.semantic_vec_path) if self.semantic_vec_path.exists() else 0
        
        return {
            'db_size_kb': db_size / 1024,
            'episodic_vectors_kb': ep_vec_size / 1024,
            'semantic_vectors_kb': sem_vec_size / 1024,
            'total_size_kb': (db_size + ep_vec_size + sem_vec_size) / 1024,
            'total_size_mb': (db_size + ep_vec_size + sem_vec_size) / 1024 / 1024,
            'episode_count': self.count_episodes(),
            'semantic_count': self.count_semantics(),
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
