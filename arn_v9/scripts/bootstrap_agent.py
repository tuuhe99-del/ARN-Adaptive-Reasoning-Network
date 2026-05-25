#!/usr/bin/env python3
"""
ARN Agent Bootstrap — Migrate OpenClaw Markdown Files into ARN
===============================================================

Reads an OpenClaw agent's markdown memory files and stores them into ARN
with appropriate memory types.  This is the one-time migration step when
switching from markdown-based memory to ARN semantic memory.

Usage:
    # Migrate a single agent
    python -m arn_v9.scripts.bootstrap_agent \
        --agent_id catcher \
        --openclaw_dir ~/.openclaw/agents/catcher \
        --data_root ~/.arn_data

    # Migrate all agents
    for agent in ~/.openclaw/agents/*; do
        python -m arn_v9.scripts.bootstrap_agent \
            --agent_id $(basename $agent) \
            --openclaw_dir $agent \
            --data_root ~/.arn_data
    done

What gets stored where:
    SOUL.md         → memory_type="identity"    (who the agent is)
    IDENTITY.md     → memory_type="identity"    (agent identity)
    AGENTS.md       → memory_type="identity"    (team map)
    USER.md         → memory_type="preference"  (user info)
    MEMORY.md       → memory_type="episode"     (past conversations)
    BOOTSTRAP.md    → memory_type="procedure"   (startup steps)
    TOOLS.md        → memory_type="procedure"   (tool descriptions)
    ERRORS.md       → memory_type="error"       (past mistakes)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure arn_v9 is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _package_root)

from arn_v9.plugin import ARNPlugin

# Map filename stem → (memory_type, importance)
FILE_TYPE_MAP = {
    "soul": ("identity", 0.95),
    "identity": ("identity", 0.95),
    "agents": ("identity", 0.90),
    "user": ("preference", 0.85),
    "memory": ("episode", 0.60),
    "bootstrap": ("procedure", 0.90),
    "tools": ("procedure", 0.80),
    "errors": ("error", 0.75),
    "procedures": ("procedure", 0.85),
    "preferences": ("preference", 0.85),
    "team": ("identity", 0.80),
}


def read_markdown_files(agent_dir: str) -> list[dict]:
    """Find all .md files in an agent's OpenClaw directory."""
    agent_path = Path(agent_dir)
    if not agent_path.exists():
        print(f"Agent directory not found: {agent_dir}")
        return []

    # Look in workspace and config dirs
    search_paths = [
        agent_path,
        agent_path / "workspace",
        agent_path / "memories",
    ]

    files = []
    for sp in search_paths:
        if not sp.exists():
            continue
        for md_file in sp.glob("*.md"):
            stem = md_file.stem.lower()
            mem_type, importance = FILE_TYPE_MAP.get(stem, ("episode", 0.5))
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                files.append({
                    "path": str(md_file),
                    "stem": stem,
                    "content": content,
                    "memory_type": mem_type,
                    "importance": importance,
                })

    return files


def chunk_content(content: str, max_chars: int = 1500) -> list[str]:
    """Split long markdown into paragraph-sized chunks."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current)

    return chunks if chunks else [content[:max_chars]]


def migrate_agent(agent_id: str, openclaw_dir: str, data_root: str, dry_run: bool = False):
    files = read_markdown_files(openclaw_dir)
    if not files:
        print(f"No markdown files found for agent '{agent_id}' in {openclaw_dir}")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would migrate {len(files)} files for '{agent_id}':")
        for f in files:
            print(f"  {f['stem']}.md → {f['memory_type']} (importance={f['importance']})")
        return

    plugin = ARNPlugin(agent_id=agent_id, data_root=data_root)
    stored_count = 0

    try:
        for f in files:
            chunks = chunk_content(f["content"])
            for chunk in chunks:
                result = plugin.store(
                    content=chunk,
                    importance=f["importance"],
                    source="bootstrap",
                    memory_type=f["memory_type"],
                    context={"origin": f["path"], "bootstrapped": True},
                )
                stored_count += 1
                print(f"  Stored {f['stem']}.md chunk → episode {result['episode_id']} "
                      f"({f['memory_type']}, importance={f['importance']})")

        stats = plugin.get_stats()
        print(f"\n✓ Agent '{agent_id}' migrated: {stored_count} chunks stored. "
              f"Total episodes: {stats['episodic_count']}")

    finally:
        plugin.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Bootstrap ARN from OpenClaw markdown files")
    parser.add_argument("--agent_id", required=True, help="Agent namespace")
    parser.add_argument("--openclaw_dir", required=True, help="Path to agent's OpenClaw directory")
    parser.add_argument("--data_root", default=None, help="ARN data root (default: ~/.arn_data)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be stored without storing")
    args = parser.parse_args()

    data_root = args.data_root or os.path.expanduser("~/.arn_data")
    migrate_agent(args.agent_id, args.openclaw_dir, data_root, args.dry_run)


if __name__ == "__main__":
    main()
