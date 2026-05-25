"""
Runner for serial Codex -> Claude -> Kimi collaboration cycles.

The runner is deliberately separate from the `arn collab` state primitives so
the protocol can be tested without invoking external model CLIs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from arn_v9 import collab


DEFAULT_COMMANDS = {
    # danger-full-access is required: workspace-write uses macOS sandbox-exec which
    # blocks writes outside the -C dir even with --add-dir. Agents must write
    # handoffs to {data_dir}/collab/handoffs/ so sandbox restriction is lifted.
    # --add-dir {repo_dir} + --add-dir {data_dir} document the intended scope.
    "codex": [
        "/Users/hustle/.nvm/versions/node/v25.2.1/bin/codex",
        "exec",
        "-C",
        "{repo_dir}",
        "-s",
        "danger-full-access",
        "--add-dir",
        "{repo_dir}",
        "--add-dir",
        "{data_dir}",
        "--skip-git-repo-check",
        "{prompt}",
    ],
    # bypassPermissions: collab agent needs to run tests, write handoffs, edit
    # files, etc. — all within repo_dir and data_dir. acceptEdits blocks bash
    # tool calls which makes the agent useless. bypassPermissions is safe here
    # because the task is local dev work with no network or secrets exposure.
    # Use 2.1.143 (current): the older .app bundle (2.1.138) hangs when stdout
    # is a file/pipe rather than a TTY — fixed by capture_output=True in runner.
    "claude": [
        "/Users/hustle/.local/share/claude/versions/2.1.143",
        "--print",
        "--permission-mode",
        "bypassPermissions",
        "--add-dir",
        "{repo_dir}",
        "--add-dir",
        "{data_dir}",
        "{prompt}",
    ],
    # --add-dir grants Kimi read/write to repo and data dirs.
    "kimi": [
        "/Users/hustle/.local/bin/kimi",
        "--work-dir",
        "{repo_dir}",
        "--add-dir",
        "{repo_dir}",
        "--add-dir",
        "{data_dir}",
        "--print",
        "--afk",
        "--prompt",
        "{prompt}",
    ],
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def command_exists(command: List[str]) -> bool:
    return Path(command[0]).exists()


def kimi_auth_status() -> Dict[str, object]:
    cred_path = Path.home() / ".kimi" / "credentials" / "kimi-code.json"
    if not cred_path.exists():
        return {
            "ok": False,
            "reason": f"missing credentials file: {cred_path}",
            "fix": "Run `/Users/hustle/.local/bin/kimi login`.",
        }
    try:
        data = json.loads(cred_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"could not read Kimi credentials metadata: {exc}",
            "fix": "Run `/Users/hustle/.local/bin/kimi logout` then `/Users/hustle/.local/bin/kimi login`.",
        }
    expires_at = float(data.get("expires_at") or 0)
    seconds_left = expires_at - time.time()
    if seconds_left <= 0:
        return {
            "ok": False,
            "reason": "Kimi OAuth token is expired",
            "expires_at": expires_at,
            "seconds_left": round(seconds_left, 1),
            "fix": "Run `/Users/hustle/.local/bin/kimi login` to refresh credentials.",
        }
    return {
        "ok": True,
        "expires_at": expires_at,
        "seconds_left": round(seconds_left, 1),
    }


def latest_handoff(data_dir: Path) -> str:
    state = collab.read_state(data_dir)
    handoff = state.get("last_handoff")
    if handoff:
        return handoff
    files = sorted(collab.handoffs_dir(data_dir).glob("*.md"))
    return str(files[-1]) if files else "None"


def load_task_brief(repo_dir: Path, task_id: str | None) -> str:
    if not task_id:
        return "No task_id was provided. Use COLLAB.md and the latest handoff to choose the smallest evidence-backed next step."
    candidates = [
        repo_dir / "tasks" / f"{task_id}.md",
        repo_dir / ".agent" / "tasks" / f"{task_id}.md",
    ]
    brief = None
    for path in candidates:
        if path.exists():
            brief = path.read_text(encoding="utf-8")
            break
    if brief is None:
        brief = (
            f"No task brief file found for `{task_id}`. Read COLLAB.md, "
            "docs/collab-protocol.md, research/*.md, and the latest handoff; "
            "then choose the smallest concrete change that advances this task_id."
        )

    # Append previous run result if it exists — prevents agents repeating done work
    result_path = repo_dir / "tasks" / f"{task_id}-result.md"
    if result_path.exists():
        result_content = result_path.read_text(encoding="utf-8")
        brief += (
            "\n\n---\n"
            "## Previous Run Result (IMPORTANT: do not repeat completed work)\n\n"
            + result_content
        )

    return brief


def build_feed_section(data_dir: Path, agent: str, limit: int = 5) -> str:
    """Return a human context section if there are recent feed messages for this agent."""
    try:
        feeds = collab.read_feeds(data_dir, agent=agent, limit=limit)
    except Exception:
        return ""
    if not feeds:
        return ""
    lines = ["\nHuman Context (recent operator messages):"]
    for entry in feeds:
        ts = entry.get("timestamp", "")[:16]
        to = entry.get("to", "all")
        msg = entry.get("message", "")
        lines.append(f"  [{ts}] to={to}: {msg}")
    return "\n".join(lines) + "\n"


def build_prompt(agent: str, repo_dir: Path, data_dir: Path) -> str:
    state = collab.read_state(data_dir)
    previous = latest_handoff(data_dir)
    task_brief = load_task_brief(repo_dir, state.get("task_id"))
    feed_section = build_feed_section(data_dir, agent)
    return f"""You are {agent} in the ARN serial collaboration loop.{feed_section}

Goal:
Work only on the currently claimed collaboration step. Do not start unrelated work.

Repository:
{repo_dir}

Runtime collaboration state:
{data_dir}/collab

Current state:
{json.dumps(collab.summarize_state(state), indent=2)}

Previous handoff:
{previous}

Current task brief:
{task_brief}

Required workflow:
1. Read COLLAB.md and docs/collab-protocol.md.
2. Read the previous handoff if it exists.
3. Inspect only files relevant to the current task and prior handoff.
4. If you are implementing, make the minimal correct change.
5. If you are reviewing and find concrete issues, fix them.
6. If you are reviewing and find no issues, do not edit code.
7. Run focused verification.
8. End by running `python3 arn_v9/scripts/arn_cli.py collab handoff ...` with your agent name.

Task discovery:
- Use ARN's goal in COLLAB.md as the north star.
- Prefer reliability and data integrity before new intelligence features.
- If you discover necessary follow-up work, record a proposed task in your handoff with problem, evidence, files, success criteria, verification, and suggested owner.
- Do not silently expand scope beyond the claimed task.

Handoff requirements:
- status must be complete, blocked, needs_review, or no_issues
- include task summary, changes/review result, verification, concerns, and next-agent focus
- do not store secrets, API keys, or credentials

Do not leave the task claimed without writing a handoff. If blocked, write a blocked handoff explaining why.
"""


def render_command(agent: str, repo_dir: Path, data_dir: Path, prompt: str) -> List[str]:
    template = DEFAULT_COMMANDS[agent]
    return [
        part.format(repo_dir=str(repo_dir), data_dir=str(data_dir), prompt=prompt)
        for part in template
    ]


def _stream_to_file(pipe, path: Path) -> None:
    """Read chunks from *pipe* and append to *path* so logs are available live."""
    with open(path, "ab") as f:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            f.write(chunk)
            f.flush()


def run_agent(agent: str, repo_dir: Path, data_dir: Path, execute: bool,
              timeout: int) -> Dict[str, object]:
    prompt = build_prompt(agent, repo_dir, data_dir)
    log_dir = collab.logs_dir(data_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = log_dir / f"{utc_stamp()}-{agent}-prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    command = render_command(agent, repo_dir, data_dir, prompt)
    result = {
        "agent": agent,
        "execute": execute,
        "prompt_file": str(prompt_path),
        "command": command,
    }
    if not command_exists(command):
        result["returncode"] = 127
        result["error"] = f"command not found: {command[0]}"
        return result
    if agent == "kimi":
        auth = kimi_auth_status()
        result["auth"] = auth
        if execute and not auth["ok"]:
            result["returncode"] = 401
            result["error"] = auth["reason"]
            return result

    if not execute:
        result["returncode"] = 0
        result["dry_run"] = True
        return result

    env = os.environ.copy()
    env["ARN_DATA_DIR"] = str(data_dir)

    # Strip Claude Code / Anthropic session vars before spawning any agent.
    # When the runner executes inside a Claude Code conversation:
    #   - CLAUDECODE=1 / CLAUDE_CODE_SESSION_ID signal child Claude CLI that it
    #     is inside an active session, causing it to reroute output to a
    #     background task file and exit immediately (zero bytes in our log).
    #   - ANTHROPIC_API_KEY="" (empty) overrides the child's own auth,
    #     causing silent auth failure with exit code 0.
    # Stripping these lets each agent CLI start as a standalone process.
    _STRIP_PREFIXES = ("CLAUDE_CODE_", "ANTHROPIC_", "CLAUDE_SESSION", "CLAUDE_EFFORT")
    _STRIP_EXACT    = {"CLAUDECODE", "AI_AGENT"}
    for key in list(env.keys()):
        if key in _STRIP_EXACT or key.startswith(_STRIP_PREFIXES):
            del env[key]

    # Use Popen with PIPEs and background threads so logs are streamed
    # incrementally. Direct file-handle stdout causes older Claude Code builds
    # to hang when stdout is not a TTY; PIPE avoids this while threads drain
    # the buffers and write to disk so watchers can tail -f live.
    stamp    = utc_stamp()
    out_path = log_dir / f"{stamp}-{agent}-stdout.log"
    err_path = log_dir / f"{stamp}-{agent}-stderr.log"
    out_path.write_text("", encoding="utf-8")   # touch so watchers can tail -f now
    err_path.write_text("", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(repo_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        result["returncode"] = 126
        result["error"] = f"failed to start agent: {exc}"
        return result

    out_thread = threading.Thread(target=_stream_to_file, args=(proc.stdout, out_path))
    err_thread = threading.Thread(target=_stream_to_file, args=(proc.stderr, err_path))
    out_thread.start()
    err_thread.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        result["error"] = f"Agent timed out after {timeout}s"

    out_thread.join()
    err_thread.join()

    result.update({
        "returncode": returncode,
        "stdout_log": str(out_path),
        "stderr_log": str(err_path),
    })
    return result


def read_log_excerpt(path: str | None, limit: int = 12000) -> str:
    if not path:
        return ""
    log_path = Path(path)
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def fallback_handoff(data_dir: Path, repo_dir: Path, agent: str,
                     result: Dict[str, object], current_step: int) -> Path:
    stdout = read_log_excerpt(result.get("stdout_log"))
    stderr = read_log_excerpt(result.get("stderr_log"), limit=4000)
    status = "complete" if current_step == 0 else "needs_review"
    changes = stdout or "Agent exited successfully but produced no stdout."
    concerns = (
        "Runner-created fallback handoff because the agent exited with code 0 "
        "without advancing collaboration state. Review the stdout/stderr logs."
    )
    if stderr:
        concerns += f"\n\nstderr excerpt:\n{stderr}"
    path, validation = collab.create_handoff(
        data_dir=data_dir,
        agent=agent,
        status=status,
        task_summary=f"Fallback handoff for {agent}",
        changes=changes,
        verification=(
            f"Agent process returned 0. stdout log: {result.get('stdout_log')}; "
            f"stderr log: {result.get('stderr_log')}"
        ),
        concerns=concerns,
        next_focus="Verify this fallback handoff carefully before trusting the changes.",
        repo_dir=repo_dir,
    )
    if not validation["valid"]:
        raise RuntimeError(f"fallback handoff failed validation: {validation['errors']}")
    return path


def ensure_started(data_dir: Path, task_id: str | None, review_chain: str | None,
                   force: bool) -> Dict[str, object]:
    path = collab.state_path(data_dir)
    if path.exists() and not force:
        state = collab.read_state(data_dir)
        if task_id and not state.get("task_id"):
            state["task_id"] = task_id
            collab.write_state(data_dir, state)
        return collab.read_state(data_dir)
    chain = collab.sanitize_review_chain(review_chain)
    return collab.init_collab(data_dir, task_id=task_id, review_chain=chain, force=force)


def run_cycle(repo_dir: Path, data_dir: Path, task_id: str | None,
              review_chain: str | None, execute: bool, force: bool,
              timeout: int) -> Dict[str, object]:
    state = ensure_started(data_dir, task_id, review_chain, force)
    events = []

    # Recover from stale CLAIMED_X states (runner was killed mid-agent run).
    # If we start and the state is already claimed but stale, write a fallback
    # handoff so the cycle can continue rather than being stuck forever.
    if str(state.get("status", "")).startswith("CLAIMED_") and collab.is_stale(state):
        stale_agent = state.get("locked_by") or (
            collab.next_agent(state) or "unknown"
        )
        dummy_result = {
            "returncode": 0,
            "stdout_log": None,
            "stderr_log": None,
        }
        try:
            handoff_path = fallback_handoff(
                data_dir, repo_dir, stale_agent, dummy_result,
                int(state.get("current_step", 0))
            )
            events.append({
                "event": "stale_claim_recovered",
                "agent": stale_agent,
                "handoff": str(handoff_path),
                "message": f"stale CLAIMED_{stale_agent.upper()} detected on startup; wrote recovery fallback",
            })
            state = collab.read_state(data_dir)
        except Exception as exc:
            events.append({
                "event": "stale_claim_recovery_failed",
                "agent": stale_agent,
                "message": str(exc),
            })

    while state.get("status") != collab.DONE_STATUS:
        agent = collab.next_agent(state)
        if not agent:
            break
        if not str(state.get("status", "")).startswith("CLAIMED_"):
            state = collab.claim_task(data_dir, agent, task_id=state.get("task_id") or task_id)
            events.append({"event": "claimed", "agent": agent, "state": state})

        before = collab.read_state(data_dir)
        result = run_agent(agent, repo_dir, data_dir, execute, timeout)
        events.append({"event": "ran_agent", "result": result})
        after = collab.read_state(data_dir)

        if not execute:
            break
        if result.get("returncode") != 0:
            # Agent failed (non-zero exit). Write a fallback so the cycle
            # doesn't get stuck in CLAIMED_X forever, then stop — don't
            # continue running subsequent agents after a hard failure.
            try:
                handoff_path = fallback_handoff(
                    data_dir, repo_dir, agent, result, int(before.get("current_step", 0))
                )
                events.append({
                    "event": "fallback_handoff",
                    "agent": agent,
                    "handoff": str(handoff_path),
                    "message": f"agent exited with code {result.get('returncode')}; runner wrote fallback handoff",
                })
            except Exception as exc:
                events.append({
                    "event": "error_fallback_failed",
                    "agent": agent,
                    "message": str(exc),
                })
            break
        if after == before or after.get("locked_by") == agent:
            try:
                handoff_path = fallback_handoff(
                    data_dir, repo_dir, agent, result, int(before.get("current_step", 0))
                )
                events.append({
                    "event": "fallback_handoff",
                    "agent": agent,
                    "handoff": str(handoff_path),
                    "message": "agent exited without advancing state; runner wrote fallback handoff",
                })
                state = collab.read_state(data_dir)
                continue
            except Exception as exc:
                events.append({
                    "event": "missing_handoff",
                    "agent": agent,
                    "message": f"agent exited without advancing collaboration state; fallback failed: {exc}",
                })
                break
        state = after

    final_state = collab.read_state(data_dir)
    final_report = write_final_report(data_dir, events, final_state)
    result: Dict[str, object] = {
        "state": final_state, "events": events, "final_report": str(final_report)
    }
    if final_state.get("status") == collab.DONE_STATUS:
        result_report = write_cycle_result_report(data_dir, repo_dir, final_state)
        if result_report:
            result["result_report"] = str(result_report)
    return result


def _parse_handoff_file(path: Path) -> Dict[str, object]:
    """Parse a handoff .md file into a structured dict (no external deps)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    result: Dict[str, object] = {"path": str(path)}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_lines = parts[1].splitlines()
            body = parts[2]
            fc_mode = False
            fc_list: List[str] = []
            for line in fm_lines:
                stripped = line.strip()
                if stripped.startswith("files_changed:"):
                    fc_mode = True
                    fc_list = []
                elif fc_mode and stripped.startswith("- "):
                    fc_list.append(stripped[2:].strip().strip('"'))
                elif stripped and ":" in stripped and not line.startswith(" ") and not stripped.startswith("-"):
                    fc_mode = False
                    k, _, v = stripped.partition(":")
                    result[k.strip()] = v.strip().strip('"')
            result["files_changed_list"] = fc_list

    sections: Dict[str, str] = {}
    current = None
    buf: List[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    result["sections"] = sections

    def pick(*keys: str) -> str:
        for k in keys:
            if k in sections and sections[k]:
                return str(sections[k])
        return ""

    result["s_changes"]      = pick("Changes", "Review Summary", "Critical Fix Applied")
    result["s_verification"] = pick("Verification")
    result["s_concerns"]     = pick("Concerns", "Concerns / Architectural Notes")
    return result


def write_cycle_result_report(data_dir: Path, repo_dir: Path,
                               final_state: Dict[str, object]) -> "Path | None":
    """
    Write a structured result report to tasks/{task_id}-result.md.
    This file is included in the next run's agent prompts so agents
    know what was already done and don't repeat completed work.
    """
    if final_state.get("status") != collab.DONE_STATUS:
        return None

    task_id  = final_state.get("task_id") or "unknown"
    cycle_id = final_state.get("cycle_id") or "?"
    done_at  = final_state.get("updated_at") or utc_stamp()

    # Load all handoffs for this cycle
    handoffs_dir = collab.handoffs_dir(data_dir)
    all_files    = sorted(handoffs_dir.glob("*.md"), key=lambda f: f.stat().st_mtime)
    handoffs     = [_parse_handoff_file(f) for f in all_files
                    if _parse_handoff_file(f).get("cycle_id") == cycle_id]

    lines: List[str] = [
        f"# Task Result: {task_id}",
        "",
        f"**Completed:** {done_at}",
        f"**Cycle:** {cycle_id}",
        f"**Status:** DONE",
        "",
        "## Agent Contributions",
        "",
    ]

    STATUS_ICON = {
        "complete":     "✅",
        "no_issues":    "✅",
        "needs_review": "⚠️",
        "blocked":      "🔴",
    }

    for h in handoffs:
        agent   = str(h.get("agent", "?")).upper()
        status  = str(h.get("status", "?"))
        icon    = STATUS_ICON.get(status, "❓")
        ts      = str(h.get("timestamp", ""))[:16]
        files   = h.get("files_changed_list", [])

        lines += [f"### {agent} — {icon} {status}  ({ts})", ""]

        changes = str(h.get("s_changes", "")).strip()
        if changes:
            lines.append("**Changes:**")
            # First 800 chars to keep it concise
            lines.append(changes[:800] + ("…" if len(changes) > 800 else ""))
            lines.append("")

        if files:
            lines.append(f"**Files changed:** {', '.join(str(f) for f in files)}")
            lines.append("")

        verif = str(h.get("s_verification", "")).strip()
        if verif:
            first_v = verif.splitlines()[0]
            lines.append(f"**Verification:** {first_v}")
            lines.append("")

        concerns = str(h.get("s_concerns", "")).strip()
        if concerns:
            lines.append("**Concerns:**")
            lines.append(concerns[:500] + ("…" if len(concerns) > 500 else ""))
            lines.append("")

    lines += [
        "---",
        "",
        "## Do Not Repeat",
        "",
        "The work above was completed in the previous run.",
        "On a new run of this task, focus only on unresolved concerns or new follow-up work.",
        "Do not re-implement, re-verify, or re-review anything already marked ✅ above.",
        "",
    ]

    out_path = repo_dir / "tasks" / f"{task_id}-result.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_final_report(data_dir: Path, events: List[Dict[str, object]],
                       state: Dict[str, object]) -> Path:
    report_dir = collab.reports_dir(data_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{utc_stamp()}-cycle-report.md"
    lines = [
        "# Collaboration Cycle Report",
        "",
        f"Status: {state.get('status')}",
        f"Task: {state.get('task_id')}",
        f"Next agent: {collab.next_agent(state)}",
        f"Last handoff: {state.get('last_handoff')}",
        "",
        "## Events",
        "",
    ]
    for event in events:
        lines.append(f"- `{event.get('event')}`")
        if "agent" in event:
            lines.append(f"  Agent: {event['agent']}")
        if "result" in event:
            result = event["result"]
            lines.append(f"  Return code: {result.get('returncode')}")
            lines.append(f"  Prompt: {result.get('prompt_file')}")
            if result.get("stdout_log"):
                lines.append(f"  Stdout: {result.get('stdout_log')}")
            if result.get("stderr_log"):
                lines.append(f"  Stderr: {result.get('stderr_log')}")
        if "message" in event:
            lines.append(f"  Message: {event['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARN serial agent collaboration")
    parser.add_argument("--repo-dir", default=str(Path.cwd()))
    parser.add_argument("--data-dir", default=os.environ.get("ARN_DATA_DIR", str(Path.home() / ".arn_data")))
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--review-chain", default=None)
    parser.add_argument("--force", action="store_true", help="Start a fresh cycle")
    parser.add_argument("--execute", action="store_true", help="Actually launch agent CLIs")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-agent timeout in seconds")
    args = parser.parse_args()

    result = run_cycle(
        repo_dir=Path(args.repo_dir).resolve(),
        data_dir=Path(args.data_dir).expanduser(),
        task_id=args.task_id,
        review_chain=args.review_chain,
        execute=args.execute,
        force=args.force,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, default=str))
    if result["state"].get("status") != collab.DONE_STATUS:
        sys.exit(2 if args.execute else 0)


if __name__ == "__main__":
    main()
