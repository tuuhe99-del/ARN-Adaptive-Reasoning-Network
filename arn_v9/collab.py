"""
File-based collaboration state for serial AI agent handoffs.

This module intentionally avoids ARN's SQLite/memmap storage. Collaboration
state is operational metadata, not user memory, and must stay easy to inspect.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


SCHEMA_VERSION = "1.0"
HANDOFF_VERSION = "1.0"
DEFAULT_REVIEW_CHAIN = ["codex", "claude", "kimi"]
VALID_AGENTS = set(DEFAULT_REVIEW_CHAIN)
VALID_HANDOFF_STATUS = {"complete", "blocked", "needs_review", "no_issues"}
DONE_STATUS = "DONE"
IDLE_STATUS = "IDLE"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collab_root(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / "collab"


def state_path(data_dir: str | Path) -> Path:
    return collab_root(data_dir) / "state.json"


def handoffs_dir(data_dir: str | Path) -> Path:
    return collab_root(data_dir) / "handoffs"


def reports_dir(data_dir: str | Path) -> Path:
    return collab_root(data_dir) / "reports"


def logs_dir(data_dir: str | Path) -> Path:
    return collab_root(data_dir) / "logs"


def feeds_dir(data_dir: str | Path) -> Path:
    return collab_root(data_dir) / "feeds"


def ensure_collab_dirs(data_dir: str | Path) -> Path:
    root = collab_root(data_dir)
    for path in (root, handoffs_dir(data_dir), reports_dir(data_dir), logs_dir(data_dir)):
        path.mkdir(parents=True, exist_ok=True)
    return root


def default_state(task_id: str | None = None,
                  review_chain: List[str] | None = None,
                  stale_after_minutes: int = 120) -> Dict[str, Any]:
    chain = review_chain or DEFAULT_REVIEW_CHAIN
    return {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": datetime.now(timezone.utc).strftime("%Y-%m-%d-manual"),
        "task_id": task_id,
        "status": IDLE_STATUS,
        "review_chain": chain,
        "current_step": 0,
        "locked_by": None,
        "locked_at": None,
        "stale_after_minutes": stale_after_minutes,
        "last_handoff": None,
        "updated_at": utc_now(),
    }


def read_state(data_dir: str | Path) -> Dict[str, Any]:
    path = state_path(data_dir)
    if not path.exists():
        return default_state()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated_at"] = utc_now()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_state(data_dir: str | Path, state: Dict[str, Any]) -> None:
    atomic_write_json(state_path(data_dir), state)


def init_collab(data_dir: str | Path, task_id: str | None = None,
                review_chain: List[str] | None = None,
                force: bool = False) -> Dict[str, Any]:
    ensure_collab_dirs(data_dir)
    path = state_path(data_dir)
    if path.exists() and not force:
        return read_state(data_dir)
    state = default_state(task_id=task_id, review_chain=review_chain)
    write_state(data_dir, state)
    return read_state(data_dir)


def _handoff_status(agent: str) -> str:
    return f"HANDOFF_{agent.upper()}"


def _claimed_status(agent: str) -> str:
    return f"CLAIMED_{agent.upper()}"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_stale(state: Dict[str, Any], now: datetime | None = None) -> bool:
    locked_at = _parse_time(state.get("locked_at"))
    if locked_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    max_age = int(state.get("stale_after_minutes", 120)) * 60
    return (now - locked_at).total_seconds() > max_age


def next_agent(state: Dict[str, Any]) -> str | None:
    if state.get("status") == DONE_STATUS:
        return None
    chain = state.get("review_chain") or DEFAULT_REVIEW_CHAIN
    step = int(state.get("current_step", 0))
    if step >= len(chain):
        return None
    return chain[step]


def claim_task(data_dir: str | Path, agent: str, task_id: str | None = None,
               steal_stale: bool = False) -> Dict[str, Any]:
    if agent not in VALID_AGENTS:
        raise ValueError(f"unknown agent: {agent}")
    ensure_collab_dirs(data_dir)
    state = read_state(data_dir)
    current_status = state.get("status", IDLE_STATUS)

    if current_status.startswith("CLAIMED_"):
        if not (steal_stale and is_stale(state)):
            raise RuntimeError(f"task is already claimed by {state.get('locked_by')}")

    expected = next_agent(state)
    if expected and agent != expected:
        raise RuntimeError(f"next agent is {expected}, not {agent}")

    if task_id is not None:
        state["task_id"] = task_id
    elif not state.get("task_id"):
        raise ValueError("task_id is required before the first claim")

    state["status"] = _claimed_status(agent)
    state["locked_by"] = agent
    state["locked_at"] = utc_now()
    write_state(data_dir, state)
    return read_state(data_dir)


def release_task(data_dir: str | Path, agent: str) -> Dict[str, Any]:
    state = read_state(data_dir)
    if state.get("locked_by") != agent:
        raise RuntimeError(f"task is locked by {state.get('locked_by')}")
    step = int(state.get("current_step", 0))
    state["status"] = IDLE_STATUS if step == 0 else _handoff_status(state["review_chain"][step - 1])
    state["locked_by"] = None
    state["locked_at"] = None
    write_state(data_dir, state)
    return read_state(data_dir)


def _git_value(args: List[str], cwd: str | Path | None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_context(cwd: str | Path | None = None) -> Dict[str, Any]:
    branch = _git_value(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    commit = _git_value(["rev-parse", "HEAD"], cwd)
    files = _git_value(["diff", "--name-only"], cwd)
    return {
        "available": bool(branch and commit),
        "branch": branch,
        "commit": commit,
        "files_changed": files.splitlines() if files else [],
    }


def create_handoff(data_dir: str | Path, agent: str, status: str,
                   task_summary: str, changes: str, verification: str,
                   concerns: str = "None", next_focus: str = "None",
                   repo_dir: str | Path | None = None) -> Tuple[Path, Dict[str, Any]]:
    if status not in VALID_HANDOFF_STATUS:
        raise ValueError(f"invalid handoff status: {status}")
    state = read_state(data_dir)
    if state.get("locked_by") != agent:
        raise RuntimeError(f"task is locked by {state.get('locked_by')}")

    ensure_collab_dirs(data_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    path = handoffs_dir(data_dir) / f"{ts}-{agent}.md"
    git = git_context(repo_dir)
    metadata = {
        "handoff_version": HANDOFF_VERSION,
        "agent": agent,
        "cycle_id": state.get("cycle_id"),
        "task_id": state.get("task_id"),
        "status": status,
        "timestamp": utc_now(),
        "git_available": git["available"],
        "branch": git["branch"],
        "commit": git["commit"],
        "files_changed": git["files_changed"],
    }
    body = [
        "---",
        *frontmatter_lines(metadata),
        "---",
        "",
        "## Task",
        task_summary.strip() or "Not provided.",
        "",
        "## Changes",
        changes.strip() or "Not provided.",
        "",
        "## Verification",
        verification.strip() or "Not provided.",
        "",
        "## Concerns",
        concerns.strip() or "None.",
        "",
        "## Next Agent Focus",
        next_focus.strip() or "None.",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")

    validation = validate_handoff(path)
    if validation["valid"]:
        step = int(state.get("current_step", 0)) + 1
        state["current_step"] = step
        state["locked_by"] = None
        state["locked_at"] = None
        state["last_handoff"] = str(path)
        if step >= len(state.get("review_chain", [])):
            state["status"] = DONE_STATUS
        else:
            state["status"] = _handoff_status(agent)
        write_state(data_dir, state)
    return path, validation


def frontmatter_lines(metadata: Dict[str, Any]) -> List[str]:
    lines = []
    for key, value in metadata.items():
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                lines.extend(f"  - {json.dumps(item)}" for item in value)
            else:
                lines.append(f"{key}: []")
        elif value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    return lines


def parse_frontmatter(text: str) -> Dict[str, Any]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise ValueError("unterminated frontmatter")
    block = text[4:end].strip("\n")
    data: Dict[str, Any] = {}
    current_list: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  - "):
            if current_list is None:
                raise ValueError("list item without key")
            data[current_list].append(json.loads(line[4:]))
            continue
        current_list = None
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line}")
        key, value = line.split(":", 1)
        value = value.strip()
        if value == "":
            data[key] = []
            current_list = key
        elif value == "[]":
            data[key] = []
        elif value == "null":
            data[key] = None
        elif value == "true":
            data[key] = True
        elif value == "false":
            data[key] = False
        else:
            data[key] = json.loads(value)
    return data


def validate_handoff(path: str | Path) -> Dict[str, Any]:
    errors = []
    handoff_path = Path(path)
    try:
        metadata = parse_frontmatter(handoff_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}

    required = [
        "handoff_version", "agent", "cycle_id", "task_id", "status",
        "timestamp", "git_available", "files_changed",
    ]
    for key in required:
        if key not in metadata:
            errors.append(f"missing required field: {key}")
    if metadata.get("handoff_version") != HANDOFF_VERSION:
        errors.append(f"unsupported handoff_version: {metadata.get('handoff_version')}")
    if metadata.get("agent") not in VALID_AGENTS:
        errors.append(f"invalid agent: {metadata.get('agent')}")
    if metadata.get("status") not in VALID_HANDOFF_STATUS:
        errors.append(f"invalid status: {metadata.get('status')}")
    if not isinstance(metadata.get("files_changed", []), list):
        errors.append("files_changed must be a list")
    if metadata.get("timestamp") and _parse_time(metadata["timestamp"]) is None:
        errors.append("timestamp must be ISO8601")
    return {"valid": not errors, "errors": errors, "metadata": metadata}


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(state)
    summary["next_agent"] = next_agent(state)
    summary["lock_stale"] = is_stale(state)
    return summary


def sanitize_review_chain(value: str | None) -> List[str]:
    if not value:
        return DEFAULT_REVIEW_CHAIN
    chain = [part.strip() for part in re.split(r"[,>]", value) if part.strip()]
    unknown = [agent for agent in chain if agent not in VALID_AGENTS]
    if unknown:
        raise ValueError(f"unknown agents in review chain: {', '.join(unknown)}")
    if len(set(chain)) != len(chain):
        raise ValueError("review chain contains duplicate agents")
    return chain


def list_handoffs(data_dir: str | Path, limit: int = 10) -> List[Dict[str, Any]]:
    """Return metadata from the most recent handoff files, newest first."""
    hdir = handoffs_dir(data_dir)
    if not hdir.exists():
        return []
    files = sorted(hdir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]
    results = []
    for f in files:
        try:
            meta = parse_frontmatter(f.read_text(encoding="utf-8"))
            meta["file"] = str(f)
            results.append(meta)
        except Exception:
            results.append({"file": str(f), "error": "could not parse frontmatter"})
    return results


def agent_health() -> Dict[str, Any]:
    """Check binary presence and auth status for each agent in DEFAULT_COMMANDS."""
    from arn_v9.collab_runner import DEFAULT_COMMANDS, kimi_auth_status

    health: Dict[str, Any] = {}
    for agent_name, cmd in DEFAULT_COMMANDS.items():
        binary = cmd[0]
        exists = Path(binary).exists()
        entry: Dict[str, Any] = {
            "binary": binary,
            "exists": exists,
            "status": "ready" if exists else "missing",
        }
        if agent_name == "kimi" and exists:
            auth = kimi_auth_status()
            entry["auth_ok"] = auth["ok"]
            if auth["ok"]:
                entry["status"] = "auth_ok"
                entry["expires_in_seconds"] = round(auth["seconds_left"])
            else:
                entry["status"] = "auth_expired"
                entry["auth_reason"] = auth.get("reason", "unknown")
        health[agent_name] = entry
    return health


def write_feed(data_dir: str | Path, message: str, target: str = "all",
               source: str = "human") -> Dict[str, Any]:
    """Append a human feed message to today's JSONL feed file."""
    if target not in set(DEFAULT_REVIEW_CHAIN) | {"all"}:
        raise ValueError(f"invalid target agent: {target}")
    ensure_collab_dirs(data_dir)
    feeds_dir(data_dir).mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    feed_file = feeds_dir(data_dir) / f"{today}.jsonl"
    entry: Dict[str, Any] = {
        "feed_version": "1.0",
        "timestamp": utc_now(),
        "from": source,
        "to": target,
        "message": message,
    }
    with feed_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_feeds(data_dir: str | Path, agent: str = "all", limit: int = 5) -> List[Dict[str, Any]]:
    """Return the last N feed entries relevant to the given agent."""
    fdir = feeds_dir(data_dir)
    if not fdir.exists():
        return []
    files = sorted(fdir.glob("*.jsonl"), reverse=True)
    entries: List[Dict[str, Any]] = []
    for f in files:
        for line in reversed(f.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            target = entry.get("to", "all")
            if target in ("all", agent):
                entries.append(entry)
            if len(entries) >= limit:
                break
        if len(entries) >= limit:
            break
    return entries
