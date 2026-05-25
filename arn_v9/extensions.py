"""
ARN v9 Feature Extensions
===========================
Minimal extensions module. Previously held BM25 hybrid retrieval,
entity extraction, callbacks, import/export, and TTL features.
These were removed because they had no callers in the production path.

Kept:
- supersede_episode: used by contradiction detector
"""

import time
import logging

logger = logging.getLogger("arn.extensions")


def supersede_episode(storage, old_id: int, new_id: int):
    """
    Mark an old episode as superseded by a new one.
    The old episode is preserved with a superseded_by pointer
    and an invalidated_at timestamp.
    """
    conn = storage._get_conn()
    now = time.time()
    conn.execute("""
        UPDATE episodes 
        SET superseded_by = ?, invalidated_at = ?
        WHERE id = ?
    """, (new_id, now, old_id))
    conn.commit()
