"""
ARN v9 Persistence Layer
=========================
SQLite for metadata + sqlite-vec (vec0) for vectors + FTS5 for full-text search.

Storage layout:
  {data_dir}/
    arn_metadata.db          # SQLite: all metadata, vectors (vec0), full-text (FTS5)
"""

import sqlite3
import numpy as np
import os
import json
import time
import hashlib
import logging
import threading
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False

from ..core.embeddings import EMBEDDING_DIM

logger = logging.getLogger("arn.storage")

SCHEMA_VERSION = 6


class _ThreadLocalConnection:
    """Thread-local SQLite connection wrapper."""

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
            # Load sqlite-vec extension per-connection (extensions are connection-local)
            if _SQLITE_VEC_AVAILABLE:
                try:
                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                except Exception as e:
                    logger.warning(f"sqlite-vec load failed: {e}")
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


class StorageEngine:
    """
    Persistent storage backend for ARN v9.

    Uses sqlite-vec (vec0) for vector storage and KNN search,
    and FTS5 for full-text keyword search.
    """

    def __init__(self, data_dir: str, max_episodes: int = 4096,
                 max_semantics: int = 2048, embedding_dim: int = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "arn_metadata.db"
        self.max_episodes = max_episodes
        self.max_semantics = max_semantics
        self.embedding_dim = embedding_dim if embedding_dim is not None else EMBEDDING_DIM

        self._conn = _ThreadLocalConnection(self.db_path, row_factory=sqlite3.Row)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn.get()

    # =========================================================
    # SCHEMA INIT + MIGRATION
    # =========================================================

    def _init_db(self):
        """Create tables if they don't exist; run migrations if needed."""
        conn = self._get_conn()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)
        """)
        conn.commit()

        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        current_ver = existing[0] if existing is not None else None

        if current_ver is not None and current_ver < SCHEMA_VERSION:
            self._migrate_schema(conn, current_ver)
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
            conn.commit()

        # Core tables (safe for both fresh and migrated dbs)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec_index INTEGER DEFAULT -1,
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
                memory_type TEXT DEFAULT 'episode',
                valid_from REAL,
                valid_until REAL,
                supersedes INTEGER,
                pinned INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec_index INTEGER DEFAULT -1,
                concept_label TEXT NOT NULL,
                confidence REAL DEFAULT 0.1,
                evidence_count INTEGER DEFAULT 0,
                contradiction_log TEXT DEFAULT '[]',
                schema_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                review_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                resolved_at REAL,
                resolution TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            )
        """)

        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance DESC)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_expires ON episodes(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_memory_type ON episodes(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_pinned ON episodes(pinned)",
            "CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_nodes(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_links_from ON memory_links(from_episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_links_to ON memory_links(to_episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text)",
            "CREATE INDEX IF NOT EXISTS idx_entities_episode ON entities(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_review_priority ON memory_review_queue(priority DESC)",
        ]:
            conn.execute(idx_sql)

        # sqlite-vec virtual tables
        dim = self.embedding_dim
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_embeddings "
            f"USING vec0(embedding float[{dim}])"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS semantic_embeddings "
            f"USING vec0(embedding float[{dim}])"
        )

        # FTS5 full-text index
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
            USING fts5(content, content='episodes', content_rowid='id',
                       tokenize='porter ascii')
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_fts_insert
            AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_fts_delete
            AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_fts_update
            AFTER UPDATE OF content ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

        # Fresh install: set schema version
        if current_ver is None:
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))

        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Run migrations from from_version up to SCHEMA_VERSION."""
        if from_version < 2:
            for sql in [
                "ALTER TABLE episodes ADD COLUMN content_hash TEXT",
                "ALTER TABLE episodes ADD COLUMN expires_at REAL",
                "ALTER TABLE episodes ADD COLUMN superseded_by INTEGER",
                "ALTER TABLE episodes ADD COLUMN invalidated_at REAL",
                "ALTER TABLE episodes ADD COLUMN user_id TEXT",
            ]:
                try:
                    conn.execute(sql)
                except Exception:
                    pass
            rows = conn.execute(
                "SELECT id, content FROM episodes WHERE content_hash IS NULL"
            ).fetchall()
            for row in rows:
                h = hashlib.sha256(' '.join(row[1].lower().split()).encode()).hexdigest()[:16]
                conn.execute("UPDATE episodes SET content_hash=? WHERE id=?", (h, row[0]))
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_expires ON episodes(expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)",
            ]:
                conn.execute(idx_sql)
            logger.info(f"Migrated schema v1 → v2 ({len(rows)} episodes hash-backfilled)")

        if from_version < 3:
            try:
                conn.execute("ALTER TABLE episodes ADD COLUMN memory_type TEXT DEFAULT 'episode'")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_memory_type ON episodes(memory_type)"
                )
                logger.info("Migrated schema v2 → v3 (memory_type)")
            except Exception:
                pass

        if from_version < 4:
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_links_from ON memory_links(from_episode_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_links_to ON memory_links(to_episode_id)"
                )
                logger.info("Migrated schema v3 → v4 (memory_links)")
            except Exception:
                pass

        if from_version < 5:
            self._migrate_v4_to_v5(conn)

        if from_version < 6:
            self._migrate_v5_to_v6(conn)

        conn.commit()

    def _migrate_v4_to_v5(self, conn: sqlite3.Connection):
        """v4 → v5: Replace memmap vectors with sqlite-vec; add FTS5."""
        dim = self.embedding_dim

        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_embeddings "
            f"USING vec0(embedding float[{dim}])"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS semantic_embeddings "
            f"USING vec0(embedding float[{dim}])"
        )

        # Migrate episodic memmap vectors
        ep_vec_path = self.data_dir / "episodic_vectors.npy"
        if ep_vec_path.exists():
            try:
                old_vecs = np.load(str(ep_vec_path))
                rows = conn.execute("SELECT id, vec_index FROM episodes").fetchall()
                migrated = 0
                for row in rows:
                    ep_id, vi = row[0], row[1]
                    if vi is not None and 0 <= vi < old_vecs.shape[0]:
                        vec = old_vecs[vi].astype(np.float32)
                        if np.any(vec != 0) and vec.shape[0] == dim:
                            conn.execute(
                                "INSERT OR REPLACE INTO episode_embeddings(rowid, embedding) VALUES (?, ?)",
                                (ep_id, vec.tobytes())
                            )
                            migrated += 1
                logger.info(f"Migrated {migrated} episode vectors to sqlite-vec")
            except Exception as e:
                logger.warning(f"Could not migrate episode vectors: {e}")

        sem_vec_path = self.data_dir / "semantic_vectors.npy"
        if sem_vec_path.exists():
            try:
                old_vecs = np.load(str(sem_vec_path))
                rows = conn.execute("SELECT id, vec_index FROM semantic_nodes").fetchall()
                migrated = 0
                for row in rows:
                    sem_id, vi = row[0], row[1]
                    if vi is not None and 0 <= vi < old_vecs.shape[0]:
                        vec = old_vecs[vi].astype(np.float32)
                        if np.any(vec != 0) and vec.shape[0] == dim:
                            conn.execute(
                                "INSERT OR REPLACE INTO semantic_embeddings(rowid, embedding) VALUES (?, ?)",
                                (sem_id, vec.tobytes())
                            )
                            migrated += 1
                logger.info(f"Migrated {migrated} semantic vectors to sqlite-vec")
            except Exception as e:
                logger.warning(f"Could not migrate semantic vectors: {e}")

        # FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
            USING fts5(content, content='episodes', content_rowid='id',
                       tokenize='porter ascii')
        """)
        conn.execute(
            "INSERT INTO episodes_fts(rowid, content) SELECT id, content FROM episodes"
        )
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_fts_insert
            AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_fts_delete
            AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_fts_update
            AFTER UPDATE OF content ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

        conn.commit()
        logger.info("Migrated schema v4 → v5 (sqlite-vec + FTS5)")

    def _migrate_v5_to_v6(self, conn: sqlite3.Connection):
        """v5 → v6: bi-temporal columns, supersedes, pinned, entities, review queue."""
        for sql in [
            "ALTER TABLE episodes ADD COLUMN valid_from REAL",
            "ALTER TABLE episodes ADD COLUMN valid_until REAL",
            "ALTER TABLE episodes ADD COLUMN supersedes INTEGER",
            "ALTER TABLE episodes ADD COLUMN pinned INTEGER DEFAULT 0",
        ]:
            try:
                conn.execute(sql)
            except Exception:
                pass  # column already exists

        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_episode ON entities(episode_id)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                review_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                resolved_at REAL,
                resolution TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_priority ON memory_review_queue(priority DESC)"
        )
        conn.commit()
        logger.info("Migrated schema v5 → v6 (bi-temporal, entities, review queue)")

    # =========================================================
    # EPISODIC MEMORY OPERATIONS
    # =========================================================

    def store_episode(self, content: str, vector: np.ndarray,
                      context: dict = None, importance: float = 0.5,
                      prediction_error: float = 0.0,
                      source: str = 'user',
                      expires_at: float = None,
                      user_id: str = None,
                      memory_type: str = 'episode',
                      valid_from: float = None) -> int:
        """Store a new episodic memory. Returns episode ID."""
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            normalized = ' '.join(content.lower().split())
            c_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            vf = valid_from if valid_from is not None else now

            cursor = conn.execute("""
                INSERT INTO episodes
                    (vec_index, content, content_hash, context_json,
                     importance, prediction_error, created_at, source,
                     expires_at, user_id, memory_type, valid_from)
                VALUES (-1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content, c_hash, json.dumps(context or {}),
                importance, prediction_error, now, source,
                expires_at, user_id, memory_type, vf,
            ))
            ep_id = cursor.lastrowid

            # Store vector in sqlite-vec
            conn.execute(
                "INSERT OR REPLACE INTO episode_embeddings(rowid, embedding) VALUES (?, ?)",
                (ep_id, vector.astype(np.float32).tobytes())
            )
            conn.commit()
            return ep_id

    def get_episode(self, episode_id: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return self._row_to_episode(row) if row else None

    def get_episodes_by_ids(self, episode_ids: List[int]) -> List[dict]:
        if not episode_ids:
            return []
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE id IN ({placeholders})", episode_ids
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_all_episodes(self, consolidated: Optional[bool] = None,
                         limit: int = None,
                         memory_type: Optional[str] = None) -> List[dict]:
        conn = self._get_conn()
        conditions = []
        params = []

        if consolidated is not None:
            conditions.append("consolidated = ?")
            params.append(int(consolidated))
        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        query = "SELECT * FROM episodes"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_episode_vectors(self, episode_ids: List[int] = None) -> Tuple[np.ndarray, List[int]]:
        """Get vectors for episodes. Returns (matrix, ids)."""
        conn = self._get_conn()

        if episode_ids is None:
            rows = conn.execute(
                "SELECT id FROM episodes WHERE consolidated=0"
            ).fetchall()
            episode_ids = [r[0] for r in rows]

        if not episode_ids:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        placeholders = ','.join('?' * len(episode_ids))
        rows = conn.execute(
            f"SELECT rowid, embedding FROM episode_embeddings WHERE rowid IN ({placeholders})",
            episode_ids
        ).fetchall()

        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        ids = []
        vectors = []
        for row in rows:
            eid = row[0]
            vec = np.frombuffer(row[1], dtype=np.float32).copy()
            if vec.shape[0] == self.embedding_dim:
                ids.append(eid)
                vectors.append(vec)

        if not vectors:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        return np.array(vectors, dtype=np.float32), ids

    def knn_search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Tuple[int, float]]:
        """KNN search via sqlite-vec. Returns [(episode_id, score), ...] sorted by score desc."""
        conn = self._get_conn()
        blob = query_vector.astype(np.float32).tobytes()
        try:
            rows = conn.execute(
                "SELECT rowid, distance FROM episode_embeddings "
                "WHERE embedding MATCH ? AND k=?",
                (blob, top_k)
            ).fetchall()
            # L2 distance → similarity score: 1/(1+d), higher is better
            return [(r[0], 1.0 / (1.0 + float(r[1]))) for r in rows]
        except Exception as e:
            logger.debug(f"knn_search failed: {e}")
            return []

    def fts_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """FTS5 full-text search. Returns [(episode_id, score), ...] sorted by score desc."""
        conn = self._get_conn()
        # Sanitize query: keep only words of length ≥ 2, no FTS5 special chars
        clean_words = [
            w for w in query.split()
            if len(w) >= 2 and not any(c in w for c in '"-*:^()')
        ]
        if not clean_words:
            return []
        fts_query = ' '.join(clean_words)
        try:
            rows = conn.execute(
                "SELECT rowid, rank FROM episodes_fts WHERE episodes_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, top_k)
            ).fetchall()
        except Exception as e:
            logger.debug(f"fts_search failed for query '{fts_query}': {e}")
            return []

        if not rows:
            return []

        # FTS5 rank is negative BM25; flip and normalize 0→1
        results = [(r[0], float(-r[1])) for r in rows]
        max_s = max(s for _, s in results)
        if max_s <= 0:
            return results
        return [(eid, s / max_s) for eid, s in results]

    def update_episode_access(self, episode_id: int):
        conn = self._get_conn()
        conn.execute(
            "UPDATE episodes SET access_count=access_count+1, last_accessed=? WHERE id=?",
            (time.time(), episode_id)
        )
        conn.commit()

    def mark_episodes_consolidated(self, episode_ids: List[int]):
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        conn.execute(
            f"UPDATE episodes SET consolidated=1 WHERE id IN ({placeholders})",
            episode_ids
        )
        conn.commit()

    def delete_episodes(self, episode_ids: List[int]):
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        for eid in episode_ids:
            try:
                conn.execute("DELETE FROM episode_embeddings WHERE rowid=?", (eid,))
            except Exception:
                pass
        conn.execute(
            f"DELETE FROM episodes WHERE id IN ({placeholders})", episode_ids
        )
        conn.commit()

    def count_episodes(self, consolidated: Optional[bool] = None) -> int:
        conn = self._get_conn()
        if consolidated is None:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated=?", (int(consolidated),)
        ).fetchone()[0]

    def supersede_episode(self, old_id: int, new_id: int):
        """Mark old_id as superseded by new_id (bi-temporal invalidation)."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "UPDATE episodes SET superseded_by=?, valid_until=?, invalidated_at=? WHERE id=?",
            (new_id, now, now, old_id)
        )
        conn.execute("UPDATE episodes SET supersedes=? WHERE id=?", (old_id, new_id))
        conn.commit()

    def set_pinned(self, episode_id: int, pinned: bool) -> bool:
        conn = self._get_conn()
        r = conn.execute(
            "UPDATE episodes SET pinned=? WHERE id=?", (int(pinned), episode_id)
        )
        conn.commit()
        return r.rowcount > 0

    def update_episode(self, episode_id: int, updates: dict,
                       new_vector: np.ndarray = None):
        """Update episode fields. Re-embeds if new_vector is provided."""
        conn = self._get_conn()
        if not updates and new_vector is None:
            return
        parts = []
        params = []
        for k, v in updates.items():
            parts.append(f"{k} = ?")
            params.append(v)
        if parts:
            params.append(episode_id)
            conn.execute(
                f"UPDATE episodes SET {', '.join(parts)} WHERE id = ?", params
            )
        if new_vector is not None:
            conn.execute(
                "INSERT OR REPLACE INTO episode_embeddings(rowid, embedding) VALUES (?, ?)",
                (episode_id, new_vector.astype(np.float32).tobytes())
            )
        conn.commit()

    def invalidate_episode(self, episode_id: int):
        """Soft-delete: set invalidated_at timestamp."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE episodes SET invalidated_at=? WHERE id=?",
            (time.time(), episode_id)
        )
        conn.commit()

    def get_supersession_chain(self, episode_id: int) -> List[dict]:
        """Return the full supersession chain anchored at episode_id."""
        conn = self._get_conn()
        visited = set()
        chain = []

        # Walk backwards (supersedes)
        cur_id = episode_id
        while cur_id and cur_id not in visited:
            row = conn.execute("SELECT * FROM episodes WHERE id=?", (cur_id,)).fetchone()
            if row is None:
                break
            visited.add(cur_id)
            chain.insert(0, self._row_to_episode(row))
            cur_id = row['supersedes'] if 'supersedes' in row.keys() else None

        # Walk forwards (superseded_by)
        cur_id = episode_id
        while True:
            row = conn.execute("SELECT * FROM episodes WHERE id=?", (cur_id,)).fetchone()
            if row is None:
                break
            next_id = row['superseded_by'] if 'superseded_by' in row.keys() else None
            if not next_id or next_id in visited:
                break
            visited.add(next_id)
            fwd = conn.execute("SELECT * FROM episodes WHERE id=?", (next_id,)).fetchone()
            if fwd:
                chain.append(self._row_to_episode(fwd))
            cur_id = next_id

        return sorted(chain, key=lambda e: e['created_at'])

    # =========================================================
    # ENTITY OPERATIONS
    # =========================================================

    def store_entities(self, episode_id: int, entities: List[Tuple[str, str]]):
        """Store extracted entities for an episode."""
        if not entities:
            return
        conn = self._get_conn()
        now = time.time()
        conn.executemany(
            "INSERT INTO entities (episode_id, entity_text, entity_type, created_at) VALUES (?, ?, ?, ?)",
            [(episode_id, text, etype, now) for text, etype in entities]
        )
        conn.commit()

    def search_entities(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Find episodes matching query tokens via entity table."""
        tokens = [t.lower() for t in query.split() if len(t) >= 3]
        if not tokens:
            return []
        conn = self._get_conn()
        placeholders = ','.join('?' * len(tokens))
        rows = conn.execute(
            f"SELECT episode_id, COUNT(*) as hits FROM entities "
            f"WHERE lower(entity_text) IN ({placeholders}) "
            f"GROUP BY episode_id ORDER BY hits DESC LIMIT ?",
            tokens + [top_k]
        ).fetchall()
        if not rows:
            return []
        max_hits = max(r[1] for r in rows)
        return [(r[0], r[1] / max_hits) for r in rows]

    # =========================================================
    # REVIEW QUEUE OPERATIONS
    # =========================================================

    def enqueue_review(self, episode_id: int, review_type: str,
                       reason: str, priority: float = 0.5) -> int:
        """Add to review queue; deduplicated by (episode_id, review_type) for open items."""
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id FROM memory_review_queue "
            "WHERE episode_id=? AND review_type=? AND resolved_at IS NULL",
            (episode_id, review_type)
        ).fetchone()
        if existing:
            return existing[0]
        cursor = conn.execute(
            "INSERT INTO memory_review_queue "
            "(episode_id, review_type, reason, priority, created_at) VALUES (?, ?, ?, ?, ?)",
            (episode_id, review_type, reason, priority, time.time())
        )
        conn.commit()
        return cursor.lastrowid

    def get_pending_reviews(self, limit: int = 10) -> List[dict]:
        """Return open review items joined with episode content."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT r.id, r.episode_id, r.review_type, r.reason, r.priority,
                   r.created_at, e.content, e.importance
            FROM memory_review_queue r
            JOIN episodes e ON e.id = r.episode_id
            WHERE r.resolved_at IS NULL
            ORDER BY r.priority DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [
            {
                'id': r[0], 'episode_id': r[1], 'review_type': r[2],
                'reason': r[3], 'priority': r[4], 'created_at': r[5],
                'content': r[6], 'importance': r[7],
            }
            for r in rows
        ]

    def resolve_review(self, review_id: int, resolution: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE memory_review_queue SET resolved_at=?, resolution=? WHERE id=?",
            (time.time(), resolution, review_id)
        )
        conn.commit()

    # =========================================================
    # SEMANTIC MEMORY OPERATIONS
    # =========================================================

    def store_semantic(self, concept_label: str, vector: np.ndarray,
                       confidence: float = 0.1, evidence_count: int = 1,
                       schema: dict = None) -> int:
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            cursor = conn.execute("""
                INSERT INTO semantic_nodes
                    (vec_index, concept_label, confidence, evidence_count,
                     schema_json, created_at, last_updated)
                VALUES (-1, ?, ?, ?, ?, ?, ?)
            """, (concept_label, confidence, evidence_count,
                  json.dumps(schema or {}), now, now))
            sem_id = cursor.lastrowid
            conn.execute(
                "INSERT OR REPLACE INTO semantic_embeddings(rowid, embedding) VALUES (?, ?)",
                (sem_id, vector.astype(np.float32).tobytes())
            )
            conn.commit()
            return sem_id

    def update_semantic(self, node_id: int, vector: np.ndarray = None,
                        confidence: float = None, evidence_count: int = None,
                        contradiction_log: list = None, schema: dict = None):
        conn = self._get_conn()

        if vector is not None:
            conn.execute(
                "INSERT OR REPLACE INTO semantic_embeddings(rowid, embedding) VALUES (?, ?)",
                (node_id, vector.astype(np.float32).tobytes())
            )

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
            f"UPDATE semantic_nodes SET {', '.join(updates)} WHERE id = ?", params
        )
        conn.commit()

    def get_all_semantics(self) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM semantic_nodes ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_semantic(r) for r in rows]

    def get_semantic_vectors(self) -> Tuple[np.ndarray, List[int]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT rowid, embedding FROM semantic_embeddings"
        ).fetchall()
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        ids = []
        vectors = []
        for row in rows:
            vec = np.frombuffer(row[1], dtype=np.float32).copy()
            if vec.shape[0] == self.embedding_dim:
                ids.append(row[0])
                vectors.append(vec)

        if not vectors:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        return np.array(vectors, dtype=np.float32), ids

    def count_semantics(self) -> int:
        return self._get_conn().execute(
            "SELECT COUNT(*) FROM semantic_nodes"
        ).fetchone()[0]

    def delete_semantics(self, node_ids: List[int]):
        conn = self._get_conn()
        for sid in node_ids:
            try:
                conn.execute("DELETE FROM semantic_embeddings WHERE rowid=?", (sid,))
            except Exception:
                pass
        placeholders = ','.join('?' * len(node_ids))
        conn.execute(
            f"DELETE FROM semantic_nodes WHERE id IN ({placeholders})", node_ids
        )
        conn.commit()

    # =========================================================
    # SYSTEM STATE
    # =========================================================

    def get_state(self, key: str, default: str = None) -> Optional[str]:
        row = self._get_conn().execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else default

    def set_state(self, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)", (key, value)
        )
        conn.commit()

    # =========================================================
    # MEMORY LINK OPERATIONS
    # =========================================================

    def create_link(self, from_id: int, to_id: int,
                    relation_type: str, confidence: float = 1.0) -> int:
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
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM memory_links
            WHERE from_episode_id=? OR to_episode_id=?
            ORDER BY created_at DESC
        """, (episode_id, episode_id)).fetchall()
        return [self._row_to_link(r) for r in rows]

    def delete_link(self, link_id: int):
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_links WHERE id=?", (link_id,))
        conn.commit()

    def get_all_links(self) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memory_links ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _row_to_episode(self, row) -> dict:
        keys = row.keys() if hasattr(row, 'keys') else []
        return {
            'id': row['id'],
            'vec_index': row['vec_index'] if 'vec_index' in keys else -1,
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
            'content_hash': row['content_hash'] if 'content_hash' in keys else None,
            'expires_at': row['expires_at'] if 'expires_at' in keys else None,
            'superseded_by': row['superseded_by'] if 'superseded_by' in keys else None,
            'invalidated_at': row['invalidated_at'] if 'invalidated_at' in keys else None,
            'user_id': row['user_id'] if 'user_id' in keys else None,
            'memory_type': row['memory_type'] if 'memory_type' in keys else 'episode',
            'valid_from': row['valid_from'] if 'valid_from' in keys else None,
            'valid_until': row['valid_until'] if 'valid_until' in keys else None,
            'supersedes': row['supersedes'] if 'supersedes' in keys else None,
            'pinned': bool(row['pinned']) if 'pinned' in keys else False,
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

    def _row_to_link(self, row) -> dict:
        return {
            'id': row['id'],
            'from_episode_id': row['from_episode_id'],
            'to_episode_id': row['to_episode_id'],
            'relation_type': row['relation_type'],
            'created_at': row['created_at'],
            'confidence': row['confidence'],
        }

    def flush(self):
        """Flush pending writes to disk."""
        self._conn.commit()

    def close(self):
        """Close all resources."""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_storage_stats(self) -> dict:
        db_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        return {
            'db_size_kb': db_size / 1024,
            'total_size_kb': db_size / 1024,
            'total_size_mb': db_size / 1024 / 1024,
            'episode_count': self.count_episodes(),
            'semantic_count': self.count_semantics(),
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
