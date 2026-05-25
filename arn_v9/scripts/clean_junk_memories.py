"""
Clean junk memories from ARN database.

Removes stored error/warning messages that should have been filtered by
shouldSkipContent before the filter was implemented.

Run:
    python3 arn_v9/scripts/clean_junk_memories.py [--agent-id default] [--dry-run]
"""

import os
import sys
import argparse
import sqlite3

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

JUNK_PATTERNS = [
    "⚠️ Something went wrong",
    "⚠️ API provider returned a billing error",
    "⚠️ Error",
    "⚠️ Warning",
]


def is_junk(content: str) -> bool:
    c = content.strip()
    return any(c.startswith(pat) for pat in JUNK_PATTERNS)


def clean_agent(data_root: str, agent_id: str, dry_run: bool):
    db_path = os.path.join(data_root, agent_id, "arn_metadata.db")
    if not os.path.exists(db_path):
        print(f"  No DB found for agent '{agent_id}' at {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, substr(content, 1, 120) FROM episodes ORDER BY id"
    ).fetchall()

    to_delete = [(row[0], row[1]) for row in rows if is_junk(row[1])]

    if not to_delete:
        print(f"  [{agent_id}] No junk memories found ({len(rows)} episodes checked).")
        conn.close()
        return 0

    print(f"  [{agent_id}] Found {len(to_delete)} junk episodes:")
    for eid, content in to_delete:
        print(f"    id={eid}: {content[:80]!r}")

    if dry_run:
        print(f"  [{agent_id}] DRY RUN — not deleting.")
        conn.close()
        return len(to_delete)

    ids = [eid for eid, _ in to_delete]
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()
    print(f"  [{agent_id}] Deleted {len(to_delete)} episodes.")
    return len(to_delete)


def main():
    parser = argparse.ArgumentParser(description="Remove junk error/warning memories from ARN DB")
    parser.add_argument("--agent-id", default="default",
                        help="Agent namespace to clean (default: 'default')")
    parser.add_argument("--all-agents", action="store_true",
                        help="Clean all agent namespaces found in data root")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting")
    parser.add_argument("--data-root",
                        default=os.environ.get("ARN_DATA_ROOT",
                                               os.path.expanduser("~/.arn_data")),
                        help="Path to ARN data root directory")
    args = parser.parse_args()

    print(f"ARN junk-memory cleaner  data_root={args.data_root}")
    print(f"Dry run: {args.dry_run}")
    print()

    if args.all_agents:
        agents = [
            d for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ]
    else:
        agents = [args.agent_id]

    total = 0
    for agent in sorted(agents):
        total += clean_agent(args.data_root, agent, args.dry_run)

    print()
    print(f"Total junk memories {'found' if args.dry_run else 'deleted'}: {total}")


if __name__ == "__main__":
    main()
