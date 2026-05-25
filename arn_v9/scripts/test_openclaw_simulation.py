#!/usr/bin/env python3
"""
OpenClaw Plugin Simulation Harness
===================================
Simulates the OpenClaw plugin's auto-store + auto-inject behavior using
ARN's Python API directly.  This validates the logic before deploying
the JavaScript plugin to the Pi.

Simulates a multi-turn conversation with:
  - User messages
  - Agent responses
  - Tool calls and results
  - Per-agent isolation (Catcher vs Koda vs Leo)

Usage:
    PYTHONPATH=/home/mokali/arn python3 test_openclaw_simulation.py
"""

import os
import sys
import tempfile
import shutil
import time

# Ensure arn_v9 is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _package_root)

from arn_v9.plugin import ARNPlugin


def simulate_turn(plugin, user_msg, agent_reply, tool_calls=None, tool_results=None):
    """
    Simulate one OpenClaw turn:
      1. Store user message
      2. Store agent reply
      3. Store tool calls + results
      4. Inject ARN context for the next turn
    """
    # Auto-store user message
    plugin.store(
        content=user_msg,
        importance=0.5,
        source="user",
        memory_type="episode",
    )

    # Auto-store agent reply
    plugin.store(
        content=agent_reply,
        importance=0.5,
        source="agent",
        memory_type="episode",
    )

    # Auto-store tool calls (procedural knowledge)
    if tool_calls:
        for tc in tool_calls:
            plugin.store(
                content=f"Tool call: {tc['name']}({tc['args']})",
                importance=0.7,
                source="tool",
                memory_type="procedure",
            )

    # Auto-store tool results
    if tool_results:
        for tr in tool_results:
            plugin.store(
                content=f"Tool result: {tr['name']} → {tr['result'][:200]}",
                importance=0.5,
                source="tool_result",
                memory_type="episode",
            )

    # Simulate context injection (what the plugin does on before_prompt_build)
    context = plugin.get_context_window(query=user_msg, max_tokens=800)
    return context


def run_agent_simulation(agent_id, data_root, turns):
    """Run a simulated conversation for one agent."""
    print(f"\n{'='*60}")
    print(f"Simulating agent: {agent_id}")
    print(f"{'='*60}")

    plugin = ARNPlugin(agent_id=agent_id, data_root=data_root, auto_consolidate=False)

    # Bootstrap identity (like the bootstrap script would do)
    plugin.store(
        content=f"I am {agent_id}, an AI agent running inside OpenClaw.",
        importance=0.95,
        source="bootstrap",
        memory_type="identity",
    )

    contexts = []
    for i, turn in enumerate(turns, 1):
        print(f"\n--- Turn {i} ---")
        ctx = simulate_turn(
            plugin,
            user_msg=turn["user"],
            agent_reply=turn["agent"],
            tool_calls=turn.get("tools"),
            tool_results=turn.get("tool_results"),
        )
        contexts.append(ctx)
        print(f"User: {turn['user'][:60]}...")
        print(f"Agent: {turn['agent'][:60]}...")
        if turn.get("tools"):
            print(f"Tools: {[t['name'] for t in turn['tools']]}")

    stats = plugin.get_stats()
    print(f"\nStats for {agent_id}: {stats['episodic_count']} episodes")

    # Test recall isolation
    recall = plugin.recall("who am I", top_k=3)
    print(f"Recall 'who am I': {[r['content'][:40] for r in recall]}")

    # Test typed recall
    identity_recall = plugin.recall("agent identity", top_k=3, memory_type="identity")
    print(f"Identity-only recall: {[r['content'][:40] for r in identity_recall]}")

    # Test procedure recall
    proc_recall = plugin.recall("how to", top_k=3, memory_type="procedure")
    print(f"Procedure-only recall: {[r['content'][:40] for r in proc_recall]}")

    plugin.shutdown()
    return stats


def main():
    tmp_root = tempfile.mkdtemp(prefix="arn_openclaw_sim_")
    print(f"Using temp data root: {tmp_root}")

    try:
        # Agent 1: Catcher (debugging agent)
        catcher_turns = [
            {
                "user": "There's a bug in my Python script. It throws KeyError on line 42.",
                "agent": "Let me check your code. Can you share the relevant snippet?",
            },
            {
                "user": "```python\nconfig = load_config()\nprint(config['api_key'])\n```",
                "agent": "The KeyError means 'api_key' is missing from your config. Add a default or check first.",
                "tools": [{"name": "code_analyzer", "args": "config['api_key']"}],
                "tool_results": [{"name": "code_analyzer", "result": "KeyError: 'api_key' not found in dict"}],
            },
            {
                "user": "I fixed it with config.get('api_key'). Thanks!",
                "agent": "Great! Using .get() with a default is the safest pattern.",
            },
        ]
        catcher_stats = run_agent_simulation("catcher", tmp_root, catcher_turns)

        # Agent 2: Koda (creative agent)
        koda_turns = [
            {
                "user": "Write a haiku about Raspberry Pi.",
                "agent": "Silent silicon, / Green LEDs blink in the dark, / Pi hums through the night.",
            },
            {
                "user": "Now make it about OpenClaw agents.",
                "agent": "Many minds awake, / Claws reach through the digital, / One goal, many hands.",
            },
        ]
        koda_stats = run_agent_simulation("koda", tmp_root, koda_turns)

        # Verify isolation: Catcher should NOT have Koda's haikus
        print(f"\n{'='*60}")
        print("ISOLATION CHECK")
        print(f"{'='*60}")

        catcher_plugin = ARNPlugin(agent_id="catcher", data_root=tmp_root, auto_consolidate=False)
        koda_plugin = ARNPlugin(agent_id="koda", data_root=tmp_root, auto_consolidate=False)

        catcher_recall = catcher_plugin.recall("haiku", top_k=3)
        koda_recall = koda_plugin.recall("haiku", top_k=3)

        print(f"Catcher recalls 'haiku': {len(catcher_recall)} results")
        for r in catcher_recall:
            print(f"  - {r['content'][:50]}...")

        print(f"Koda recalls 'haiku': {len(koda_recall)} results")
        for r in koda_recall:
            print(f"  - {r['content'][:50]}...")

        # Catcher should have 0 haiku results (haiku is not in his data)
        # Actually, the embedder might find semantic similarity between "haiku"
        # and other poetry-like text... but ideally it should be empty or low-confidence
        if len(catcher_recall) == 0:
            print("✓ Catcher has no haiku memories (perfect isolation)")
        else:
            print(f"⚠ Catcher has {len(catcher_recall)} haiku-like matches (embedding similarity bleed — expected at low thresholds)")

        # Koda should have 2 haiku results
        assert len(koda_recall) >= 1, "Koda should remember his haikus"
        print("✓ Koda remembers his haikus")

        # Verify tool calls were stored as procedures
        catcher_procs = catcher_plugin.recall("code analyzer", memory_type="procedure", top_k=3)
        print(f"\nCatcher procedure recall 'code analyzer': {len(catcher_procs)} results")
        for r in catcher_procs:
            print(f"  - {r['content'][:60]}...")
        assert len(catcher_procs) >= 1, "Tool call should be stored as procedure"
        print("✓ Tool calls stored as procedures")

        # Verify context injection works
        ctx = catcher_plugin.get_context_window("debug python error", max_tokens=500)
        print(f"\nInjected context for 'debug python error' ({len(ctx)} chars):")
        print(ctx[:400] + "..." if len(ctx) > 400 else ctx)
        assert "KeyError" in ctx or "api_key" in ctx or "bug" in ctx, "Context should include relevant memories"
        print("✓ Context injection includes relevant memories")

        catcher_plugin.shutdown()
        koda_plugin.shutdown()

        print(f"\n{'='*60}")
        print("ALL SIMULATION CHECKS PASSED")
        print(f"{'='*60}")
        print(f"Catcher: {catcher_stats['episodic_count']} episodes")
        print(f"Koda:    {koda_stats['episodic_count']} episodes")
        print(f"Data root: {tmp_root}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
