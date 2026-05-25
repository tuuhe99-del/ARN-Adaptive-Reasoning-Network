#!/usr/bin/env python3
"""
ARN Memory Editor — Human-in-the-Loop CLI
===========================================
Lets humans directly view and edit agent memories stored in ARN.

Usage:
    python -m arn_v9.scripts.memory_editor list --agent_id catcher
    python -m arn_v9.scripts.memory_editor edit --agent_id catcher --id 42
    python -m arn_v9.scripts.memory_editor add --agent_id catcher \
        --content "User prefers dark mode" --type preference
    python -m arn_v9.scripts.memory_editor delete --agent_id catcher --id 42

This is the `nano IDENTITY.md` equivalent for ARN.
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile

# Ensure arn_v9 is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _package_root)

from arn_v9.plugin import ARNPlugin


def get_plugin(agent_id: str, data_root: str = None):
    data_root = data_root or os.path.expanduser("~/.arn_data")
    return ARNPlugin(agent_id=agent_id, data_root=data_root)


def cmd_list(args):
    plugin = get_plugin(args.agent_id, args.data_root)
    episodes = plugin._arn.storage.get_all_episodes(
        memory_type=args.type, limit=args.limit
    )
    if not episodes:
        print(f"No memories found for agent '{args.agent_id}'")
        if args.type:
            print(f"  (filtered by type: {args.type})")
        return

    print(f"\n{'ID':<6} {'Type':<14} {'Importance':<10} {'Content'}")
    print("-" * 80)
    for ep in episodes:
        mem_type = ep.get('memory_type', 'episode')
        content = ep['content'].replace('\n', ' ')[:55]
        print(f"{ep['id']:<6} {mem_type:<14} {ep['importance']:<10.2f} {content}...")
    print(f"\nTotal: {len(episodes)} memories")


def cmd_edit(args):
    plugin = get_plugin(args.agent_id, args.data_root)
    ep = plugin._arn.storage.get_episode(args.id)
    if ep is None:
        print(f"Error: episode {args.id} not found for agent '{args.agent_id}'")
        sys.exit(1)

    editor = os.environ.get("EDITOR", "nano")
    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False)
    try:
        header = f"""# ARN Memory Editor
# Agent: {args.agent_id} | Episode: {ep['id']} | Type: {ep.get('memory_type', 'episode')}
# Importance: {ep['importance']:.2f} | Accesses: {ep['access_count']}
# --- Edit content below this line. Lines starting with # are ignored. ---
"""
        tmp.write(header + ep['content'])
        tmp.close()
        subprocess.call([editor, tmp.name])

        with open(tmp.name, "r") as f:
            lines = f.readlines()
        # Strip comment lines
        new_lines = [ln for ln in lines if not ln.startswith("#")]
        new_content = "".join(new_lines).strip()
    finally:
        os.unlink(tmp.name)

    if new_content == ep['content'].strip():
        print("No changes made.")
        return

    # Re-embed and update
    new_vec = plugin._arn.embedder.encode(new_content, mode="passage")
    conn = plugin._arn.storage._get_conn()
    conn.execute(
        "UPDATE episodes SET content = ?, content_hash = NULL WHERE id = ?",
        (new_content, args.id)
    )
    conn.commit()
    plugin._arn.storage._episodic_vectors[ep['vec_index']] = new_vec
    print(f"Updated episode {args.id}.")


def cmd_add(args):
    plugin = get_plugin(args.agent_id, args.data_root)
    result = plugin.store(
        content=args.content,
        importance=args.importance,
        source="human-edit",
        memory_type=args.type or "episode",
    )
    print(f"Created episode {result['episode_id']} for agent '{args.agent_id}'")


def cmd_delete(args):
    plugin = get_plugin(args.agent_id, args.data_root)
    ep = plugin._arn.storage.get_episode(args.id)
    if ep is None:
        print(f"Error: episode {args.id} not found")
        sys.exit(1)

    if not args.force:
        confirm = input(f"Delete episode {args.id}? '{ep['content'][:60]}...' [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return

    plugin._arn.storage.delete_episodes([args.id])
    print(f"Deleted episode {args.id}.")


def cmd_show(args):
    plugin = get_plugin(args.agent_id, args.data_root)
    ep = plugin._arn.storage.get_episode(args.id)
    if ep is None:
        print(f"Error: episode {args.id} not found")
        sys.exit(1)

    print(json.dumps(ep, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="ARN Memory Editor")
    parser.add_argument("--agent_id", required=True, help="Agent namespace")
    parser.add_argument("--data_root", default=None, help="ARN data root (default: ~/.arn_data)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--type", default=None, help="Filter by memory_type")
    p_list.add_argument("--limit", type=int, default=50)

    p_edit = sub.add_parser("edit", help="Edit a memory in $EDITOR")
    p_edit.add_argument("--id", type=int, required=True)

    p_add = sub.add_parser("add", help="Add a new memory")
    p_add.add_argument("--content", required=True)
    p_add.add_argument("--importance", type=float, default=0.7)
    p_add.add_argument("--type", default="episode", help="Memory category")

    p_del = sub.add_parser("delete", help="Delete a memory")
    p_del.add_argument("--id", type=int, required=True)
    p_del.add_argument("--force", action="store_true")

    p_show = sub.add_parser("show", help="Show raw episode JSON")
    p_show.add_argument("--id", type=int, required=True)

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "edit":
        cmd_edit(args)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "delete":
        cmd_delete(args)
    elif args.cmd == "show":
        cmd_show(args)


if __name__ == "__main__":
    main()
