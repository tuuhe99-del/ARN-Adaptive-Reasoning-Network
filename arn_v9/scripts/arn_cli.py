#!/usr/bin/env python3
"""
ARN v9 CLI
===========
Persistent memory from the command line. Works with any AI coding assistant:
Codex, Claude Code, Kimi, Aider, OpenClaw, or plain terminal.

Commands:
    arn store   -c "fact to remember" -i 0.8
    arn recall  -q "what do I know?" -k 5
    arn context -q "current topic" -m 1000
    arn forget  -q "topic to forget"
    arn stats
    arn maintain
    arn setup   [--tier nano|base] [--client codex|claude|kimi|openclaw]
    arn export  -o backup.json
    arn import  -f backup.json

Environment (all optional, sensible defaults):
    ARN_DATA_DIR          Storage directory (default: ~/.arn_data)
    ARN_EMBEDDING_TIER    Model tier: nano|small|base|base-e5 (default: nano)
    ARN_AGENT_ID          Agent namespace (default: default)

Bug fixes over v9.0:
    - Uses ARN_EMBEDDING_TIER not ARN_EMBEDDING_MODEL (was TypeError)
    - Consistent data directory (always ARN_DATA_DIR, no more cli vs default)
    - Suppresses HuggingFace unauthenticated warnings
    - Model pre-download in setup avoids degraded-mode surprise
"""

import sys
import os
import json
import argparse
import hashlib
import time
import logging
import warnings
from pathlib import Path

# ─── Suppress noisy warnings BEFORE any imports ───
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', message='.*Unauthenticated.*')
warnings.filterwarnings('ignore', message='.*huggingface.*')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# ─── Make arn_v9 importable from any install location ───
_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parent.parent  # arn_v9/scripts → arn_v9 → root
sys.path.insert(0, str(_package_root))
# Fallback: check if installed under ~/
sys.path.insert(0, str(Path.home()))


# ─── Resolve config from env with correct variable names ───
def get_config():
    """
    Single source of truth for all configuration.
    Reads env vars with correct names, provides sensible defaults.
    """
    # ARN_DATA_DIR is the canonical var. Also check legacy ARN_DATA_ROOT.
    data_dir = os.environ.get(
        'ARN_DATA_DIR',
        os.environ.get('ARN_DATA_ROOT', str(Path.home() / '.arn_data'))
    )
    
    # ARN_EMBEDDING_TIER is the canonical var. 
    # If someone set ARN_EMBEDDING_MODEL, translate it.
    tier = os.environ.get('ARN_EMBEDDING_TIER', None)
    if tier is None:
        model = os.environ.get('ARN_EMBEDDING_MODEL', '')
        tier = _model_to_tier(model) if model else 'nano'
    
    agent_id = os.environ.get(
        'ARN_AGENT_ID',
        os.environ.get('OPENCLAW_AGENT_ID', 'default')
    )
    use_embeddings_raw = os.environ.get('ARN_USE_EMBEDDINGS', '1').strip().lower()
    use_embeddings = use_embeddings_raw not in {'0', 'false', 'no', 'off'}
    
    return {
        'data_dir': data_dir,
        'tier': tier,
        'agent_id': agent_id,
        'use_embeddings': use_embeddings,
    }


def _model_to_tier(model_name: str) -> str:
    """Translate a model name to the correct tier string."""
    mapping = {
        'all-MiniLM-L6-v2': 'nano',
        'sentence-transformers/all-MiniLM-L6-v2': 'nano',
        'all-mpnet-base-v2': 'small',
        'sentence-transformers/all-mpnet-base-v2': 'small',
        'bge-base-en-v1.5': 'base',
        'BAAI/bge-base-en-v1.5': 'base',
        'e5-base-v2': 'base-e5',
        'intfloat/e5-base-v2': 'base-e5',
    }
    for key, tier in mapping.items():
        if key in model_name:
            return tier
    return 'nano'  # Safe default


# ─── Plugin factory ───
def get_plugin(strict: bool = False, config: dict = None):
    """
    Create a plugin instance with correct config.
    Always uses ARN_EMBEDDING_TIER (not model name) and
    consistent data directory.
    """
    from arn_v9.plugin import ARNPlugin
    
    if config is None:
        config = get_config()
    
    plugin = ARNPlugin(
        agent_id=config['agent_id'],
        data_root=config['data_dir'],
        use_embeddings=config.get('use_embeddings', True),
        embedding_tier=config['tier'],
        auto_consolidate=True,
        consolidation_threshold=128,
    )
    
    if plugin._arn.embedder.is_degraded and config.get('use_embeddings', True):
        msg = (
            "Embedding model not loaded. Memory will not work correctly.\n"
            "Fix: pip install sentence-transformers\n"
            "Then run: arn setup"
        )
        if strict:
            print(msg, file=sys.stderr)
            plugin.shutdown()
            sys.exit(1)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    
    return plugin


# ═══════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════

def cmd_store(args):
    """Store a new memory."""
    with get_plugin(strict=args.strict) as plugin:
        tags = [t.strip() for t in args.tags.split(',') if t.strip()] if args.tags else []
        result = plugin.store(
            content=args.content,
            importance=args.importance,
            tags=tags,
            source=args.source,
            time_context=args.time_context,
        )
        print(json.dumps(result, indent=2))


def cmd_recall(args):
    """Recall relevant memories."""
    with get_plugin(strict=args.strict) as plugin:
        results = plugin.recall(
            query=args.query,
            top_k=args.top_k,
            time_filter=args.time_filter,
        )
        print(json.dumps(results, indent=2))


def cmd_context(args):
    """Get formatted context window for prompt injection."""
    with get_plugin(strict=args.strict) as plugin:
        context = plugin.get_context_window(
            query=args.query if args.query else None,
            max_tokens=args.max_tokens,
        )
        print(context)


def cmd_forget(args):
    """Forget memories matching a query."""
    with get_plugin(strict=args.strict) as plugin:
        results = plugin.recall(query=args.query, top_k=args.top_k)
        strong = [r for r in results
                  if r.get('similarity', 0) >= args.min_similarity]
        
        if not strong:
            print(json.dumps({"forgotten": 0, "message": "No matching memories found"}))
            return
        
        ids = [r['id'] for r in strong if r.get('type') == 'episodic']
        if ids:
            plugin._arn.storage.delete_episodes(ids)
        
        print(json.dumps({
            "forgotten": len(ids),
            "matched": [r['content'][:80] for r in strong],
        }, indent=2))


def cmd_maintain(args):
    """Run consolidation and maintenance."""
    with get_plugin(strict=args.strict) as plugin:
        stats = plugin.maintain()
        print(json.dumps(stats, indent=2))


def cmd_stats(args):
    """Print system statistics."""
    with get_plugin(strict=args.strict) as plugin:
        stats = plugin.get_stats()
        config = get_config()
        stats['config'] = config
        print(json.dumps(stats, indent=2, default=str))


def cmd_export(args):
    """Export all memories to JSON."""
    with get_plugin(strict=args.strict) as plugin:
        episodes = plugin._arn.storage.get_all_episodes()
        export = {
            'version': 'arn_v9_export_v1',
            'exported_at': time.time(),
            'agent_id': get_config()['agent_id'],
            'episode_count': len(episodes),
            'episodes': episodes,
        }
        
        outpath = args.output or 'arn_backup.json'
        with open(outpath, 'w') as f:
            json.dump(export, f, indent=2, default=str)
        
        print(json.dumps({"exported": len(episodes), "file": outpath}))


def cmd_import(args):
    """Import memories from JSON."""
    with get_plugin(strict=args.strict) as plugin:
        with open(args.file, 'r') as f:
            data = json.load(f)
        
        imported = 0
        skipped = 0
        for ep in data.get('episodes', []):
            content = ep.get('content', '')
            if not content:
                continue
            try:
                plugin.store(
                    content=content,
                    importance=ep.get('importance', 0.5),
                    source=ep.get('source', 'import'),
                )
                imported += 1
            except Exception:
                skipped += 1
        
        print(json.dumps({
            "imported": imported,
            "skipped": skipped,
            "total_in_file": len(data.get('episodes', [])),
        }, indent=2))


def cmd_collab(args):
    """Manage serial multi-agent collaboration handoffs."""
    from arn_v9 import collab

    config = get_config()
    data_dir = args.data_dir or config['data_dir']

    try:
        if args.collab_command == 'init':
            chain = collab.sanitize_review_chain(args.review_chain)
            state = collab.init_collab(
                data_dir=data_dir,
                task_id=args.task_id,
                review_chain=chain,
                force=args.force,
            )
            print(json.dumps(collab.summarize_state(state), indent=2))

        elif args.collab_command == 'status':
            state = collab.read_state(data_dir)
            print(json.dumps(collab.summarize_state(state), indent=2))

        elif args.collab_command == 'next':
            state = collab.read_state(data_dir)
            print(json.dumps({
                "next_agent": collab.next_agent(state),
                "status": state.get("status"),
                "task_id": state.get("task_id"),
                "lock_stale": collab.is_stale(state),
            }, indent=2))

        elif args.collab_command == 'claim':
            state = collab.claim_task(
                data_dir=data_dir,
                agent=args.agent,
                task_id=args.task_id,
                steal_stale=args.steal_stale,
            )
            print(json.dumps(collab.summarize_state(state), indent=2))

        elif args.collab_command == 'release':
            state = collab.release_task(data_dir=data_dir, agent=args.agent)
            print(json.dumps(collab.summarize_state(state), indent=2))

        elif args.collab_command == 'handoff':
            path, validation = collab.create_handoff(
                data_dir=data_dir,
                agent=args.agent,
                status=args.status,
                task_summary=args.task,
                changes=args.changes,
                verification=args.verification,
                concerns=args.concerns,
                next_focus=args.next_focus,
                repo_dir=args.repo_dir or Path.cwd(),
            )
            print(json.dumps({
                "handoff": str(path),
                "validation": validation,
                "state": collab.summarize_state(collab.read_state(data_dir)),
            }, indent=2))

        elif args.collab_command == 'validate-handoff':
            print(json.dumps(collab.validate_handoff(args.file), indent=2))

        elif args.collab_command == 'agents':
            health = collab.agent_health()
            print(json.dumps(health, indent=2))

        elif args.collab_command == 'history':
            limit = getattr(args, 'limit', 10)
            handoffs = collab.list_handoffs(data_dir, limit=limit)
            if getattr(args, 'file', None):
                # cat a specific handoff
                p = Path(args.file)
                if not p.exists():
                    raise FileNotFoundError(f"handoff not found: {args.file}")
                print(p.read_text(encoding="utf-8"))
            else:
                print(json.dumps(handoffs, indent=2))

        elif args.collab_command == 'feed':
            entry = collab.write_feed(
                data_dir=data_dir,
                message=args.message,
                target=args.agent,
            )
            print(json.dumps(entry, indent=2))

        elif args.collab_command == 'run':
            from arn_v9 import collab_runner
            task_id = args.task_id
            chain = collab.sanitize_review_chain(args.review_chain) if args.review_chain else None
            review_chain_str = ",".join(chain) if chain else None
            result = collab_runner.run_cycle(
                repo_dir=Path(args.repo_dir or Path.cwd()).resolve(),
                data_dir=Path(data_dir).expanduser(),
                task_id=task_id,
                review_chain=review_chain_str,
                execute=not args.dry_run,
                force=getattr(args, 'force', False),
                timeout=getattr(args, 'timeout', 7200),
            )
            print(json.dumps(result, indent=2, default=str))
            if result["state"].get("status") != collab.DONE_STATUS:
                sys.exit(2 if not args.dry_run else 0)

        elif args.collab_command == 'dashboard':
            _cmd_collab_dashboard(
                data_dir=data_dir,
                once=getattr(args, 'once', False),
                refresh=getattr(args, 'refresh', 0),
            )

        else:
            raise ValueError("missing collab subcommand")
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _cmd_collab_dashboard(data_dir: str, once: bool, refresh: int) -> None:
    """Render a human-readable collaboration status board."""
    import shutil

    def _render(data_dir: str) -> None:
        from arn_v9 import collab as _collab
        from arn_v9.collab_runner import DEFAULT_COMMANDS, kimi_auth_status

        state = _collab.read_state(data_dir)
        summary = _collab.summarize_state(state)

        width = min(shutil.get_terminal_size((80, 24)).columns, 80)
        ts = time.strftime("%H:%M:%S")
        chain = " → ".join(state.get("review_chain") or [])

        def _pad(s: str, w: int) -> str:
            return s[:w].ljust(w)

        inner = width - 4  # two border chars + two spaces
        hline = "─" * (width - 2)

        def _row(label: str, value: str) -> str:
            text = f"{label}: {value}"
            return "│ " + _pad(text, inner) + " │"

        def _header(title: str) -> str:
            return "│ " + _pad(title, inner) + " │"

        def _section(title: str) -> str:
            return "│ " + _pad(title, inner) + " │" + "\n│ " + _pad("─" * len(title), inner) + " │"

        lines = [
            f"┌{'─' * (width - 2)}┐",
            "│ " + _pad(f"ARN Collaboration Dashboard              Refresh: {ts}", inner) + " │",
            f"├{hline}┤",
            _row("Task", str(state.get("task_id") or "none")),
            _row("Status", f"{state.get('status', 'IDLE'):<24} Chain: {chain}"),
            "│" + " " * (width - 2) + "│",
        ]

        # Agent status section
        lines += [_section("Agent Status"), "│" + " " * (width - 2) + "│"]
        for agent_name, cmd in DEFAULT_COMMANDS.items():
            binary = cmd[0]
            exists = Path(binary).exists()
            if agent_name == "kimi" and exists:
                auth = kimi_auth_status()
                if auth["ok"]:
                    status_str = f"● auth OK   expires in {int(auth['seconds_left'])}s"
                else:
                    status_str = f"● auth expired"
            else:
                status_str = "● ready" if exists else "✗ missing"
            label = f"{agent_name:<7} {status_str:<20} {binary}"
            lines.append("│ " + _pad(label, inner) + " │")

        lines.append("│" + " " * (width - 2) + "│")

        # Lock section
        locked_by = state.get("locked_by")
        locked_at = state.get("locked_at")
        stale = summary.get("lock_stale", False)
        lines += [_section("Lock"), "│" + " " * (width - 2) + "│"]
        if locked_by:
            lines.append(_row("Locked by", locked_by))
            lines.append(_row("Since", str(locked_at or "unknown")))
            if locked_at:
                from datetime import datetime, timezone
                try:
                    t0 = datetime.fromisoformat(str(locked_at).replace("Z", "+00:00"))
                    elapsed_m = int((datetime.now(timezone.utc) - t0).total_seconds() / 60)
                    max_m = int(state.get("stale_after_minutes", 120))
                    stale_str = f"yes ({elapsed_m}m / {max_m}m)" if stale else f"no ({elapsed_m}m / {max_m}m)"
                    lines.append(_row("Stale", stale_str))
                except Exception:
                    pass
        else:
            lines.append(_header("  (not locked)"))

        lines.append("│" + " " * (width - 2) + "│")

        # Recent handoffs
        handoffs = _collab.list_handoffs(data_dir, limit=3)
        lines += [_section("Recent Handoffs"), "│" + " " * (width - 2) + "│"]
        if handoffs:
            for h in handoffs:
                if "error" in h:
                    fname = Path(h.get("file", "")).name
                    summary_h = f"  (unparseable handoff) {fname}"
                else:
                    ts_h = str(h.get("timestamp", ""))[:16].replace("T", " ")
                    agent_h = str(h.get("agent", "?"))
                    status_h = str(h.get("status", "?"))
                    fname = Path(h.get("file", "")).name
                    summary_h = f"{ts_h}  {agent_h:<7} {status_h:<15} {fname}"
                lines.append("│ " + _pad(summary_h, inner) + " │")
        else:
            lines.append(_header("  (no handoffs yet)"))

        lines.append("│" + " " * (width - 2) + "│")

        # Recent report
        report_dir = _collab.reports_dir(data_dir)
        reports = sorted(report_dir.glob("*.md"), reverse=True) if report_dir.exists() else []
        lines += [_section("Recent Report"), "│" + " " * (width - 2) + "│"]
        if reports:
            lines.append("│ " + _pad(str(reports[0]), inner) + " │")
        else:
            lines.append(_header("  (no reports yet)"))

        lines.append(f"└{'─' * (width - 2)}┘")
        print("\n".join(lines))

    if once or refresh <= 0:
        _render(data_dir)
        return

    try:
        import os as _os
        while True:
            _os.system("clear")
            _render(data_dir)
            time.sleep(refresh)
    except KeyboardInterrupt:
        pass


# ═══════════════════════════════════════════
# SETUP COMMAND
# ═══════════════════════════════════════════

# Client instruction templates
_CLIENT_INSTRUCTIONS = {
    'codex': {
        'filename': 'arn.md',
        'paths': [
            Path.home() / '.codex' / 'memories',
            Path.home() / '.codex',
        ],
        'content': """# ARN Memory

You have persistent memory via ARN. Use it to remember facts across sessions.

## Store a fact
```bash
arn store -c "fact to remember" -i 0.8 --time-context current
```
Importance: 0.9 for identity/names, 0.7-0.8 for preferences/decisions, 0.5 for general facts, 0.1-0.2 for chitchat.
Time context: "current" for present facts, "past" for "used to" facts, "future" for plans.

## Recall memories
```bash
arn recall -q "what is relevant to this topic" -k 5
```
Check the `confidence_tier` field: high = state it, medium = hedge, low = don't use it.

## Get context for your prompt
```bash
arn context -q "current conversation topic" -m 1000
```

## Rules
- Store important user facts immediately
- Recall before answering questions about past discussions
- Never store passwords, API keys, or credentials
- Run `arn maintain` during idle time
""",
    },
    'claude': {
        'filename': 'arn-memory.md',
        'paths': [
            Path.home() / '.claude',
            Path.home() / '.claude' / 'memories',
        ],
        'content': """# ARN Memory Integration

You have persistent memory through the `arn` CLI. It survives across sessions.

## Commands
- `arn store -c "fact" -i 0.8` — store a fact (importance 0.0-1.0)
- `arn recall -q "query" -k 5` — find relevant memories
- `arn context -q "topic"` — get formatted context block
- `arn forget -q "topic"` — delete memories about a topic
- `arn stats` — check memory status

## When to use
- **Store** when the user shares identity, preferences, decisions, or project facts
- **Recall** before answering questions that reference previous conversations
- **Context** when you need comprehensive background for a complex task

## Confidence tiers
Each recall result has a `confidence_tier`:
- `high` — state the fact plainly
- `medium` — say "I believe you mentioned..."
- `low` — don't use this result, it's a weak match

## Important
- Never store credentials (passwords, API keys, tokens)
- Use `--time-context past` for "used to" facts, `future` for plans
- Run `arn maintain` periodically to consolidate memories
""",
    },
    'kimi': {
        'filename': 'arn-memory.md',
        'paths': [
            Path.home() / '.kimi',
            Path.home() / '.kimi' / 'memories',
        ],
        'content': """# ARN Memory

Persistent memory via `arn` CLI. Store facts, recall context across sessions.

## Quick reference
```bash
arn store -c "user's name is Mohamed" -i 0.9
arn recall -q "user's name" -k 3
arn context -q "current topic" -m 1000
arn forget -q "outdated information"
arn stats
arn maintain
```

## Rules
- Store important facts immediately (names=0.9, preferences=0.7, general=0.5)
- Recall before answering history-dependent questions
- Check confidence_tier: high=certain, medium=hedge, low=ignore
- Never store passwords or API keys
""",
    },
    'openclaw': {
        'filename': 'SKILL.md',
        'paths': [
            Path.home() / '.openclaw' / 'skills' / 'arn-memory',
        ],
        'content': None,  # Uses the full SKILL.md from the package
    },
}


def cmd_setup(args):
    """
    One-command setup for ARN + AI client integration.
    
    Handles:
    1. Dependency verification
    2. Data directory creation
    3. Environment variables (persistent via ~/.bashrc)
    4. Model download and verification
    5. Store/recall round-trip test
    6. Client-specific instruction files
    """
    tier = args.tier
    client = args.client
    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / '.arn_data'
    
    print(f"\nARN v9 Setup")
    print(f"  Tier:   {tier}")
    print(f"  Client: {client or 'none'}")
    print(f"  Data:   {data_dir}\n")
    
    # Step 1: Check dependencies
    print("Checking dependencies...")
    
    missing = []
    try:
        import numpy
        print(f"  numpy: ok")
    except ImportError:
        missing.append('numpy')
        print(f"  numpy: MISSING")
    
    try:
        import sentence_transformers
        print(f"  sentence-transformers: ok")
    except ImportError:
        missing.append('sentence-transformers')
        print(f"  sentence-transformers: MISSING")
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\nThen re-run: arn setup --tier {tier}")
        sys.exit(1)
    
    # Optional deps
    for pkg, label in [('rank_bm25', 'BM25 search'), ('spacy', 'entity extraction')]:
        try:
            __import__(pkg)
            print(f"  {label}: ok")
        except ImportError:
            print(f"  {label}: not installed (optional)")
    
    # Step 2: Create data directory
    print(f"\nSetting up directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'default').mkdir(exist_ok=True)
    print(f"  Created: {data_dir}")
    
    # Step 3: Set environment variables persistently
    print(f"\nConfiguring environment...")
    env_lines = {
        'ARN_DATA_DIR': str(data_dir),
        'ARN_EMBEDDING_TIER': tier,
        'ARN_AGENT_ID': 'default',
    }
    
    bashrc = Path.home() / '.bashrc'
    if bashrc.exists():
        content = bashrc.read_text()
        additions = []
        for var, val in env_lines.items():
            # Remove any existing ARN env lines first
            lines = content.split('\n')
            lines = [l for l in lines if not l.strip().startswith(f'export {var}=')]
            content = '\n'.join(lines)
            additions.append(f'export {var}="{val}"')
        
        with open(bashrc, 'w') as f:
            f.write(content.rstrip() + '\n\n# ARN v9 configuration\n')
            f.write('\n'.join(additions) + '\n')
        
        print(f"  Updated ~/.bashrc")
    
    # Set for current process too
    for var, val in env_lines.items():
        os.environ[var] = val
        print(f"  {var}={val}")
    
    # Step 4: Download model
    print(f"\nDownloading embedding model ({tier})...")
    print(f"  This may take 1-2 minutes on first run...")
    
    try:
        from arn_v9.core.embeddings import EmbeddingEngine, MODEL_CONFIGS
        model_info = MODEL_CONFIGS.get(tier, MODEL_CONFIGS['nano'])
        print(f"  Model: {model_info['name']}")
        
        engine = EmbeddingEngine(use_model=True, tier=tier)
        if engine.is_degraded:
            print(f"  WARNING: Model failed to load. Check internet connection.")
            print(f"  ARN will retry on next use.")
        else:
            test_vec = engine.encode("test")
            print(f"  Loaded: {test_vec.shape[0]}-dimensional vectors")
            print(f"  Status: ready")
    except Exception as e:
        print(f"  Model download issue: {e}")
        print(f"  ARN will retry on first use.")
    
    # Step 5: Test store/recall
    print(f"\nTesting memory...")
    try:
        from arn_v9.plugin import ARNPlugin
        plugin = ARNPlugin(
            agent_id='default',
            data_root=str(data_dir),
            embedding_tier=tier,
        )
        
        plugin.store(
            content="ARN setup test — this will be deleted",
            importance=0.5,
            source='setup',
        )
        
        results = plugin.recall("ARN setup test", top_k=1)
        if results and 'setup test' in results[0].get('content', ''):
            print(f"  Store:  ok")
            print(f"  Recall: ok (confidence: {results[0].get('confidence_tier', 'unknown')})")
            # Clean up test memory
            if results[0].get('type') == 'episodic':
                plugin._arn.storage.delete_episodes([results[0]['id']])
        else:
            print(f"  Store:  ok")
            print(f"  Recall: returned different result (model may still be loading)")
        
        plugin.shutdown()
    except Exception as e:
        print(f"  Test failed: {e}")
    
    # Step 6: Client integration
    if client:
        print(f"\nSetting up {client} integration...")
        _setup_client(client, tier)
    
    # Done
    print(f"\n{'='*50}")
    print(f"ARN memory is ready.")
    print(f"  Model:  {tier}")
    print(f"  Data:   {data_dir}")
    print(f"  Agent:  default")
    if client:
        print(f"  Client: {client}")
    print(f"\nQuick test:")
    print(f'  arn store -c "My name is Mohamed" -i 0.9')
    print(f'  arn recall -q "what is my name" -k 1')
    if bashrc.exists():
        print(f"\nRestart your terminal or run: source ~/.bashrc")
    print()


def _setup_client(client: str, tier: str):
    """Write client-specific instruction files."""
    template = _CLIENT_INSTRUCTIONS.get(client)
    if not template:
        print(f"  Unknown client: {client}")
        print(f"  Supported: codex, claude, kimi, openclaw")
        return
    
    # Find or create the target directory
    target_dir = None
    for p in template['paths']:
        if p.exists():
            target_dir = p
            break
    
    if target_dir is None:
        # Create the first path option
        target_dir = template['paths'][0]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {target_dir}")
    
    # Write the instruction file
    if client == 'openclaw':
        # Copy the full SKILL.md from the package
        skill_src = _script_dir.parent / 'openclaw_skill' / 'SKILL.md'
        if skill_src.exists():
            import shutil
            target_file = target_dir / 'SKILL.md'
            shutil.copy2(str(skill_src), str(target_file))
            
            # Fix CLI paths in the copied SKILL.md
            content = target_file.read_text()
            cli_path = str(_script_dir / 'arn_cli.py')
            content = content.replace(
                '~/arn_v9/scripts/arn_cli.py',
                cli_path
            )
            target_file.write_text(content)
            print(f"  Wrote: {target_file}")
        else:
            print(f"  SKILL.md not found at {skill_src}")
    else:
        content = template['content']
        target_file = target_dir / template['filename']
        target_file.write_text(content)
        print(f"  Wrote: {target_file}")
    
    print(f"  {client} integration configured")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='arn',
        description='ARN v9 — Persistent memory for AI agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  arn setup --tier nano --client codex
  arn store -c "User prefers Python" -i 0.8
  arn recall -q "programming preferences" -k 3
  arn stats

Clients: codex, claude, kimi, openclaw
Tiers:   nano (22MB, fast), small (420MB), base (440MB, best), base-e5 (440MB)
        """
    )
    parser.add_argument('--strict', action='store_true', default=False,
                        help='Exit with error if embedding model unavailable')
    
    sub = parser.add_subparsers(dest='command')
    sub.required = True
    
    # ─── setup ───
    p_setup = sub.add_parser('setup', help='One-command setup and integration')
    p_setup.add_argument('--tier', '-t', default='nano',
                         choices=['nano', 'small', 'base', 'base-e5'],
                         help='Embedding model tier (default: nano)')
    p_setup.add_argument('--client', '-c', default=None,
                         choices=['codex', 'claude', 'kimi', 'openclaw'],
                         help='AI client to integrate with')
    p_setup.add_argument('--data-dir', '-d', default=None,
                         help=f'Data directory (default: ~/.arn_data)')
    p_setup.set_defaults(func=cmd_setup)
    
    # ─── store ───
    p_store = sub.add_parser('store', help='Store a memory')
    p_store.add_argument('--content', '-c', required=True,
                         help='Text to remember')
    p_store.add_argument('--importance', '-i', type=float, default=0.5,
                         help='Importance 0.0-1.0 (default: 0.5)')
    p_store.add_argument('--tags', default='',
                         help='Comma-separated tags')
    p_store.add_argument('--source', '-s', default='agent',
                         help='Source (default: agent)')
    p_store.add_argument('--time-context', default='current',
                         choices=['past', 'current', 'future'],
                         help='Temporal scope (default: current)')
    p_store.set_defaults(func=cmd_store)
    
    # ─── recall ───
    p_recall = sub.add_parser('recall', help='Recall relevant memories')
    p_recall.add_argument('--query', '-q', required=True,
                          help='Natural language query')
    p_recall.add_argument('--top-k', '-k', type=int, default=5,
                          help='Number of results (default: 5)')
    p_recall.add_argument('--time-filter', default=None,
                          choices=['past', 'current', 'future'],
                          help='Temporal filter')
    p_recall.set_defaults(func=cmd_recall)
    
    # ─── context ───
    p_ctx = sub.add_parser('context', help='Get context for prompt injection')
    p_ctx.add_argument('--query', '-q', default='',
                       help='Focus query')
    p_ctx.add_argument('--max-tokens', '-m', type=int, default=1000,
                       help='Token budget (default: 1000)')
    p_ctx.set_defaults(func=cmd_context)
    
    # ─── forget ───
    p_forget = sub.add_parser('forget', help='Forget memories about a topic')
    p_forget.add_argument('--query', '-q', required=True,
                          help='What to forget')
    p_forget.add_argument('--top-k', '-k', type=int, default=5,
                          help='Max memories to forget (default: 5)')
    p_forget.add_argument('--min-similarity', type=float, default=0.5,
                          help='Min similarity to delete (default: 0.5)')
    p_forget.set_defaults(func=cmd_forget)
    
    # ─── maintain ───
    p_maint = sub.add_parser('maintain', help='Run memory consolidation')
    p_maint.set_defaults(func=cmd_maintain)
    
    # ─── stats ───
    p_stats = sub.add_parser('stats', help='Show memory statistics')
    p_stats.set_defaults(func=cmd_stats)

    # ─── collab ───
    p_collab = sub.add_parser('collab', help='Manage multi-agent handoffs')
    p_collab.add_argument('--data-dir', '-d', default=None,
                          help='Data directory (default: ARN_DATA_DIR or ~/.arn_data)')
    collab_sub = p_collab.add_subparsers(dest='collab_command')
    collab_sub.required = True

    p_collab_init = collab_sub.add_parser('init', help='Initialize collaboration state')
    p_collab_init.add_argument('--task-id', default=None,
                               help='Task identifier for the current cycle')
    p_collab_init.add_argument('--review-chain', default=None,
                               help='Comma-separated agents (default: codex,claude,kimi)')
    p_collab_init.add_argument('--force', action='store_true',
                               help='Overwrite existing collaboration state')
    p_collab_init.set_defaults(func=cmd_collab)

    p_collab_status = collab_sub.add_parser('status', help='Show collaboration state')
    p_collab_status.set_defaults(func=cmd_collab)

    p_collab_next = collab_sub.add_parser('next', help='Show the next expected agent')
    p_collab_next.set_defaults(func=cmd_collab)

    p_collab_claim = collab_sub.add_parser('claim', help='Claim the next workflow step')
    p_collab_claim.add_argument('--agent', required=True,
                                choices=['codex', 'claude', 'kimi'])
    p_collab_claim.add_argument('--task-id', default=None,
                                help='Task identifier, required for first claim if state has none')
    p_collab_claim.add_argument('--steal-stale', action='store_true',
                                help='Take over an expired lock')
    p_collab_claim.set_defaults(func=cmd_collab)

    p_collab_release = collab_sub.add_parser('release', help='Release a claimed step')
    p_collab_release.add_argument('--agent', required=True,
                                  choices=['codex', 'claude', 'kimi'])
    p_collab_release.set_defaults(func=cmd_collab)

    p_collab_handoff = collab_sub.add_parser('handoff', help='Write a handoff and advance state')
    p_collab_handoff.add_argument('--agent', required=True,
                                  choices=['codex', 'claude', 'kimi'])
    p_collab_handoff.add_argument('--status', required=True,
                                  choices=['complete', 'blocked', 'needs_review', 'no_issues'])
    p_collab_handoff.add_argument('--task', required=True,
                                  help='Task summary')
    p_collab_handoff.add_argument('--changes', required=True,
                                  help='Semantic summary of changes or review result')
    p_collab_handoff.add_argument('--verification', required=True,
                                  help='Tests or checks run')
    p_collab_handoff.add_argument('--concerns', default='None',
                                  help='Known concerns for the next agent')
    p_collab_handoff.add_argument('--next-focus', default='None',
                                  help='What the next agent should inspect')
    p_collab_handoff.add_argument('--repo-dir', default=None,
                                  help='Repository directory for git metadata')
    p_collab_handoff.set_defaults(func=cmd_collab)

    p_collab_validate = collab_sub.add_parser('validate-handoff',
                                              help='Validate a handoff frontmatter block')
    p_collab_validate.add_argument('file', help='Handoff markdown file')
    p_collab_validate.set_defaults(func=cmd_collab)

    p_collab_agents = collab_sub.add_parser('agents', help='Show agent binary health')
    p_collab_agents.set_defaults(func=cmd_collab)

    p_collab_history = collab_sub.add_parser('history', help='List recent handoffs')
    p_collab_history.add_argument('--limit', '-n', type=int, default=10,
                                  help='Number of handoffs to show (default: 10)')
    p_collab_history.add_argument('--file', default=None,
                                  help='Cat a specific handoff file')
    p_collab_history.set_defaults(func=cmd_collab)

    p_collab_feed = collab_sub.add_parser('feed',
                                          help='Broadcast a message to one or all agents')
    p_collab_feed.add_argument('--message', '-m', required=True,
                               help='Message to broadcast')
    p_collab_feed.add_argument('--agent', '-a', default='all',
                               choices=['codex', 'claude', 'kimi', 'all'],
                               help='Target agent (default: all)')
    p_collab_feed.set_defaults(func=cmd_collab)

    p_collab_run = collab_sub.add_parser('run', help='Trigger a collaboration cycle')
    p_collab_run.add_argument('--task-id', default=None,
                              help='Task identifier for the cycle')
    p_collab_run.add_argument('--review-chain', default=None,
                              help='Comma-separated agents override')
    p_collab_run.add_argument('--repo-dir', default=None,
                              help='Repository directory (default: cwd)')
    p_collab_run.add_argument('--dry-run', action='store_true',
                              help='Show what would run without executing')
    p_collab_run.add_argument('--force', action='store_true',
                              help='Start a fresh cycle (overwrite existing state)')
    p_collab_run.add_argument('--timeout', type=int, default=7200,
                              help='Per-agent timeout in seconds (default: 7200)')
    p_collab_run.set_defaults(func=cmd_collab)

    p_collab_dashboard = collab_sub.add_parser('dashboard',
                                               help='Show live collaboration status board')
    p_collab_dashboard.add_argument('--once', action='store_true',
                                    help='Render once and exit (default when --refresh not set)')
    p_collab_dashboard.add_argument('--refresh', type=int, default=0,
                                    help='Auto-refresh interval in seconds (0 = once)')
    p_collab_dashboard.set_defaults(func=cmd_collab)

    # ─── export ───
    p_export = sub.add_parser('export', help='Export memories to JSON')
    p_export.add_argument('--output', '-o', default='arn_backup.json',
                          help='Output file (default: arn_backup.json)')
    p_export.set_defaults(func=cmd_export)
    
    # ─── import ───
    p_import = sub.add_parser('import', help='Import memories from JSON')
    p_import.add_argument('--file', '-f', required=True,
                          help='JSON file to import')
    p_import.set_defaults(func=cmd_import)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
