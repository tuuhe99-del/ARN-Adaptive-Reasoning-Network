#!/usr/bin/env python3
"""
ARN Base-Tier Migration Script
================================
Migrates stored episodic and semantic vectors between embedding tiers.

Usage:
    # Dry-run (shows what would happen, no writes)
    python3 -m arn_v9.scripts.migrate_to_base_tier --agent-dir ~/.arn_data/main --dry-run

    # Live migration (nano → base, the default)
    python3 -m arn_v9.scripts.migrate_to_base_tier --agent-dir ~/.arn_data/main

    # Explicit tier selection
    python3 -m arn_v9.scripts.migrate_to_base_tier --agent-dir ~/.arn_data/main \\
        --from-tier nano --to-tier base

    # Same-dimension model swap (e.g. switching base models with same dim)
    python3 -m arn_v9.scripts.migrate_to_base_tier --agent-dir ~/.arn_data/main \\
        --from-tier base --to-tier base-e5

    # Migrate all agents under a data root
    python3 -m arn_v9.scripts.migrate_to_base_tier --data-root ~/.arn_data

The ARN server MUST be stopped before running this script.
After migration, restart with the appropriate ARN_EMBEDDING_TIER env var.
"""

import argparse
import sqlite3
import shutil
import sys
import time
import numpy as np
from pathlib import Path


def _check_server_not_running(agent_dir: Path, dry_run: bool = False, force: bool = False) -> None:
    """Best-effort check that the ARN server is not holding the WAL file open."""
    if dry_run or force:
        return
    wal = agent_dir / "arn_metadata.db-wal"
    if wal.exists() and wal.stat().st_size > 0:
        print(f"  WARNING: {wal.name} is non-empty — ARN server may still be running.")
        print("  Stop the server before migrating, then run:")
        print(f"    sqlite3 {agent_dir}/arn_metadata.db 'PRAGMA wal_checkpoint(FULL)'")
        print("  Then re-run with --force to skip this check.")
        sys.exit(1)


def _load_model(tier: str = "base"):
    """Load the sentence-transformers model for the given tier."""
    from arn_v9.core.embeddings import EmbeddingEngine
    print(f"  Loading embedding model for tier='{tier}'...")
    t0 = time.time()
    engine = EmbeddingEngine(use_model=True, tier=tier)
    engine._load_model()
    elapsed = time.time() - t0
    print(f"  Model loaded in {elapsed:.1f}s  (dim={engine.embedding_dim})")
    return engine


def _get_existing_dim(agent_dir: Path) -> int:
    ep_path = agent_dir / "episodic_vectors.npy"
    if not ep_path.exists():
        return None
    arr = np.load(str(ep_path), mmap_mode='r')
    return arr.shape[1] if arr.ndim == 2 else None


def _get_from_tier_from_fingerprint(agent_dir: Path, default: str = "nano") -> str:
    """Read the stored model fingerprint to determine the from-tier, or return default."""
    import json
    fp = agent_dir / ".model_fingerprint"
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
            stored = data.get("tier")
            if stored:
                return stored
        except Exception:
            pass
    return default


def migrate_agent(agent_dir: Path, engine, dry_run: bool = False, force: bool = False,
                  from_tier: str = "nano", to_tier: str = "base") -> dict:
    """
    Migrate one agent directory's vectors to the new model.
    Returns stats dict.
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating: {agent_dir}")

    ep_path = agent_dir / "episodic_vectors.npy"
    sem_path = agent_dir / "semantic_vectors.npy"
    db_path = agent_dir / "arn_metadata.db"

    if not db_path.exists():
        print("  SKIP — no arn_metadata.db found")
        return {"skipped": True}

    # Detect existing dimension
    existing_dim = _get_existing_dim(agent_dir)
    if existing_dim is None:
        print("  SKIP — no episodic_vectors.npy found")
        return {"skipped": True}

    target_dim = engine.embedding_dim
    if existing_dim == target_dim and from_tier == to_tier:
        print(f"  SKIP — already at dim={target_dim}")
        return {"already_done": True, "dim": target_dim}

    # Same-dim model swap: warn but proceed
    if existing_dim == target_dim and from_tier != to_tier:
        print(f"  WARNING: Same-dim migration: vectors will be replaced but dim check will not fail")
        print(f"  (from_tier={from_tier} → to_tier={to_tier}, dim={target_dim})")

    print(f"  Existing dim={existing_dim}, migrating → dim={target_dim}")

    # Load existing vector arrays
    old_ep = np.load(str(ep_path), mmap_mode='r').copy()  # (N, 384)
    old_sem = np.load(str(sem_path), mmap_mode='r').copy() if sem_path.exists() else None

    # Read episode metadata from SQLite
    _check_server_not_running(agent_dir, dry_run=dry_run, force=force)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    episodes = conn.execute(
        "SELECT id, vec_index, content FROM episodes "
        "WHERE invalidated_at IS NULL ORDER BY vec_index"
    ).fetchall()

    sem_nodes = conn.execute(
        "SELECT id, vec_index, concept_label as content FROM semantic_nodes ORDER BY vec_index"
    ).fetchall()

    conn.close()

    print(f"  Episodes to re-embed: {len(episodes)}, semantic nodes: {len(sem_nodes)}")

    if dry_run:
        print("  [DRY RUN] Would re-embed and write new vector files.")
        return {"dry_run": True, "episodes": len(episodes), "semantics": len(sem_nodes)}

    # Create new vector arrays with target dim
    new_ep = np.zeros((old_ep.shape[0], target_dim), dtype=np.float32)
    new_sem = np.zeros((old_sem.shape[0], target_dim), dtype=np.float32) if old_sem is not None else None

    # Re-embed episodes
    print(f"  Re-embedding {len(episodes)} episodes...", flush=True)
    t0 = time.time()
    for i, row in enumerate(episodes):
        vec = engine.encode(row['content'], mode='passage')
        new_ep[row['vec_index']] = vec
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(episodes) - i - 1) / rate
            print(f"    {i+1}/{len(episodes)} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    ep_elapsed = time.time() - t0
    print(f"  Episodes done in {ep_elapsed:.1f}s ({len(episodes)/ep_elapsed:.1f}/s)")

    # Re-embed semantic nodes
    if sem_nodes and new_sem is not None:
        print(f"  Re-embedding {len(sem_nodes)} semantic nodes...", flush=True)
        t1 = time.time()
        for row in sem_nodes:
            vec = engine.encode(row['content'], mode='passage')
            new_sem[row['vec_index']] = vec
        print(f"  Semantic nodes done in {time.time()-t1:.1f}s")

    # Atomic write: backup originals, then write new files
    ts = int(time.time())
    backup_ep = ep_path.with_suffix(f".npy.pre-{to_tier}-{ts}")
    backup_sem = sem_path.with_suffix(f".npy.pre-{to_tier}-{ts}") if sem_path.exists() else None

    print(f"  Backing up originals: {backup_ep.name}, {backup_sem.name if backup_sem else 'n/a'}")
    shutil.copy2(str(ep_path), str(backup_ep))
    if sem_path.exists() and backup_sem:
        shutil.copy2(str(sem_path), str(backup_sem))

    # Write new files via temp → rename for atomicity
    # Note: np.save() appends ".npy" if not already present, so use a temp
    # name that already ends in .npy to avoid a double-extension.
    ep_tmp = ep_path.parent / "episodic_vectors_migration_tmp.npy"
    np.save(str(ep_tmp), new_ep)
    ep_tmp.rename(ep_path)
    print(f"  episodic_vectors.npy written  ({new_ep.shape})")

    if new_sem is not None and sem_path.exists():
        sem_tmp = sem_path.parent / "semantic_vectors_migration_tmp.npy"
        np.save(str(sem_tmp), new_sem)
        sem_tmp.rename(sem_path)
        print(f"  semantic_vectors.npy written ({new_sem.shape})")

    # Update model fingerprint to reflect the new tier
    import json as _json
    fp = agent_dir / ".model_fingerprint"
    fp.write_text(_json.dumps({"tier": to_tier, "dim": target_dim, "written_at": time.time()}))
    print(f"  .model_fingerprint updated: tier={to_tier}, dim={target_dim}")

    print(f"  Migration complete. Old files backed up as *.pre-{to_tier}-*")
    print(f"  Now restart ARN server with: ARN_EMBEDDING_TIER={to_tier}")

    return {
        "episodes_migrated": len(episodes),
        "semantics_migrated": len(sem_nodes),
        "old_dim": existing_dim,
        "new_dim": target_dim,
        "ep_elapsed_s": round(ep_elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ARN vectors between embedding tiers"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent-dir", type=Path, help="Single agent directory to migrate")
    group.add_argument("--data-root", type=Path, help="Migrate all agent dirs under this root")
    parser.add_argument("--from-tier", default=None,
                        help="Source embedding tier (default: auto-detect from fingerprint or 'nano')")
    parser.add_argument("--to-tier", default="base", choices=["nano", "small", "base", "base-e5"],
                        help="Target embedding tier (default: base)")
    # Keep --tier as a legacy alias for --to-tier
    parser.add_argument("--tier", default=None, choices=["small", "base", "base-e5"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, no writes")
    parser.add_argument("--force", action="store_true", help="Skip WAL/server check (use after stopping server)")
    args = parser.parse_args()

    # --tier is a legacy alias; --to-tier takes precedence
    to_tier = args.to_tier
    if args.tier and args.to_tier == "base":
        to_tier = args.tier

    # Resolve from_tier: explicit flag > fingerprint > default "nano"
    # For --data-root we'll resolve per-agent inside the loop
    from_tier_global = args.from_tier  # may be None (auto-detect per agent)

    print("=" * 60)
    print(f"ARN Tier Migration  (to_tier={to_tier}, dry_run={args.dry_run})")
    print("=" * 60)

    if not args.dry_run:
        print("\nWARNING: This will modify vector files in place.")
        print(f"Backups of originals will be saved as *.pre-{to_tier}-<timestamp>")
        print("The ARN server must be STOPPED before proceeding.\n")
        response = input("Confirm migration? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    # Load model once (expensive)
    engine = _load_model(to_tier)

    results = {}

    if args.agent_dir:
        agent_dir = args.agent_dir.expanduser()
        from_tier = from_tier_global or _get_from_tier_from_fingerprint(agent_dir, default="nano")
        results[str(agent_dir)] = migrate_agent(
            agent_dir, engine,
            dry_run=args.dry_run, force=args.force,
            from_tier=from_tier, to_tier=to_tier,
        )
    else:
        data_root = args.data_root.expanduser()
        candidates = [d for d in data_root.iterdir()
                      if d.is_dir() and (d / "arn_metadata.db").exists()]
        if not candidates:
            print(f"No agent directories found under {data_root}")
            sys.exit(1)
        print(f"\nFound {len(candidates)} agent directories to check:")
        for d in sorted(candidates):
            print(f"  {d.name}")
        print()
        for d in sorted(candidates):
            from_tier = from_tier_global or _get_from_tier_from_fingerprint(d, default="nano")
            results[d.name] = migrate_agent(
                d, engine,
                dry_run=args.dry_run, force=args.force,
                from_tier=from_tier, to_tier=to_tier,
            )

    print("\n" + "=" * 60)
    print("Summary:")
    for name, r in results.items():
        if r.get("skipped"):
            print(f"  {name}: SKIPPED")
        elif r.get("already_done"):
            print(f"  {name}: already at dim={r['dim']}")
        elif r.get("dry_run"):
            print(f"  {name}: DRY RUN — {r['episodes']} episodes, {r['semantics']} nodes")
        else:
            print(f"  {name}: {r['episodes_migrated']} episodes, {r['semantics_migrated']} nodes "
                  f"({r['old_dim']}→{r['new_dim']} dim, {r['ep_elapsed_s']}s)")

    if not args.dry_run:
        print(f"\nNext step: restart ARN server with ARN_EMBEDDING_TIER={to_tier}")


if __name__ == "__main__":
    main()
