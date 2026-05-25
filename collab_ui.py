#!/usr/bin/env python3
"""
ARN Collaboration Terminal UI
Run: python3 collab_ui.py
"""
import os, sys, json, re, shutil, subprocess, time, itertools
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).parent
DATA_DIR = Path.home() / ".arn_data"
COLLAB   = DATA_DIR / "collab"
TASKS    = REPO / "tasks"
CLI      = REPO / "arn_v9/scripts/arn_cli.py"
PY       = sys.executable
AGENTS   = ["kimi", "claude", "codex"]

# ── Terminal helpers ───────────────────────────────────────────────────────────
def W():   return shutil.get_terminal_size().columns
def H():   return shutil.get_terminal_size().lines
def clr(): os.system("clear")

def _c(t, code): return f"\033[{code}m{t}\033[0m"
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")
def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def italic(t): return _c(t, "3")
def ansi_len(s): return len(re.sub(r'\033\[[0-9;]*m', '', str(s)))

def hr(ch="─"):    return dim(ch * W())
def hr_s(ch="╌"): return dim(ch * W())

def trunc(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n-1] + "…"

def rjust(label, value, w):
    """Print label on left, value right-aligned to w."""
    raw = ansi_len(label) + ansi_len(value)
    return label + " " * max(1, w - raw) + value

def wrap_text(text, width):
    """Simple word wrapper — returns list of plain-text lines <= width chars."""
    words = str(text).split()
    lines, cur = [], ""
    for word in words:
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]

# ── State helpers ──────────────────────────────────────────────────────────────
def read_state():
    f = COLLAB / "state.json"
    try:   return json.loads(f.read_text()) if f.exists() else {}
    except: return {}

def elapsed_secs(ts):
    if not ts: return 0
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - dt).total_seconds())
    except: return 0

def fmt_elapsed(s):
    if s < 60:   return "just now"
    if s < 3600: return f"{s // 60}m"
    return f"{s // 3600}h {(s % 3600) // 60}m"

def status_badge(s):
    if not s:           return dim("idle")
    if "DONE"    in s:  return green(s)
    if "CLAIMED" in s:  return yellow(s)
    if "HANDOFF" in s:  return cyan(s)
    return s

def agent_status_icon(status):
    s = (status or "").lower()
    if s == "complete":     return green("✅ complete")
    if s == "no_issues":    return green("✅ no issues")
    if s == "needs_review": return yellow("⚠️  needs review")
    if s == "blocked":      return red("🔴 blocked")
    return dim(status or "unknown")

def runner_alive():
    try:
        r = subprocess.run(["pgrep", "-f", "collab_runner"],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())
    except: return False

def run_cli(*args):
    return subprocess.run([PY, str(CLI)] + list(args),
                          cwd=REPO, capture_output=True, text=True)

def last_handoff_lines(path, n=3):
    if not path: return []
    try:
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        return lines[:n]
    except: return []

# ── Handoff parsing ────────────────────────────────────────────────────────────
def parse_handoff(path):
    """Parse a handoff .md file into a structured dict."""
    if not path: return {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except: return {}

    result = {"path": str(path)}
    body = text

    # Parse YAML frontmatter between first two ---
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_lines = parts[1].splitlines()
            body = parts[2]
            fc_mode = False
            fc_list = []
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

    # Parse markdown ## sections
    sections = {}
    current = None
    buf = []
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

    # Convenience aliases — agents use different section names
    def pick(*keys):
        for k in keys:
            if k in sections and sections[k]:
                return sections[k]
        return ""

    result["s_changes"]      = pick("Changes", "Review Summary", "Critical Fix Applied")
    result["s_verification"] = pick("Verification")
    result["s_concerns"]     = pick("Concerns", "Concerns / Architectural Notes")
    result["s_next"]         = pick("Next Agent Focus", "Next Agent Focus (Codex)")
    result["s_task"]         = pick("Task")

    return result


def load_cycle_handoffs(cycle_id=None):
    """Return parsed handoffs for a cycle ordered by mtime."""
    d = COLLAB / "handoffs"
    if not d.exists(): return []
    files = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime)
    handoffs = [parse_handoff(f) for f in files]
    if cycle_id:
        handoffs = [h for h in handoffs if h.get("cycle_id") == cycle_id]
    return handoffs

# ── Process detection ──────────────────────────────────────────────────────────
def detect_agent_process(agent):
    """
    Return dict with process info if the agent CLI is running, else None.
    Matches the COMM field only — not the full command line — so Python runner
    processes that mention the agent name in --review-chain are not matched.
    """
    cli_aliases = {
        "kimi":   ["kimi", "kimi code", "kimi-code"],
        "claude": ["claude", "claude code", "claude-code"],
        "codex":  ["codex", "codex cli"],
    }
    aliases = [a.lower() for a in cli_aliases.get(agent.lower(), [agent.lower()])]

    try:
        r = subprocess.run(
            ["ps", "-eo", "pid,etime,comm"],
            capture_output=True, text=True
        )
        for line in r.stdout.splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) < 3: continue
            pid, etime, comm = parts[0], parts[1], parts[2].strip()
            if any(a in comm.lower() for a in aliases):
                return {
                    "pid":       pid,
                    "cli_name":  comm,
                    "elapsed_s": _etime_to_secs(etime),
                    "cmd_short": comm,
                }
    except: pass
    return None

def _etime_to_secs(etime):
    try:
        parts = etime.replace("-", ":").split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 2:  return parts[0]*60 + parts[1]
        if len(parts) == 3:  return parts[0]*3600 + parts[1]*60 + parts[2]
        if len(parts) == 4:  return parts[0]*86400 + parts[1]*3600 + parts[2]*60 + parts[3]
    except: pass
    return 0

# ── Log discovery ──────────────────────────────────────────────────────────────
def session_start_time():
    """Timestamp of the most recent prompt file — marks when this session began."""
    d = COLLAB / "logs"
    if not d.exists(): return None
    files = sorted(d.glob("*-prompt.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0].stat().st_mtime if files else None

def find_current_log(agent):
    """Return the stdout log for THIS session (newer than the session prompt)."""
    d = COLLAB / "logs"
    if not d.exists(): return None
    since = session_start_time() or 0
    candidates = []
    for pat in (f"*{agent}*stdout*", f"*{agent}*.log"):
        for f in d.glob(pat):
            if f.stat().st_mtime > since:
                candidates.append(f)
    if not candidates: return None
    return max(candidates, key=lambda f: f.stat().st_mtime)

def find_any_recent_log(agent):
    """Fallback: most recent log for this agent regardless of session."""
    d = COLLAB / "logs"
    if not d.exists(): return None
    for pat in (f"*{agent}*stdout*", f"*{agent}*.log"):
        files = sorted(d.glob(pat), key=lambda f: f.stat().st_mtime, reverse=True)
        if files: return files[0]
    return None

# ── Header ─────────────────────────────────────────────────────────────────────
def print_header(title="ARN Collaboration Terminal"):
    w = W()
    inner = w - 4
    raw   = f"  {title}"
    spaces = " " * max(0, inner - len(raw))
    print(dim("╔" + "═" * (w-2) + "╗"))
    print(dim("║") + bold(raw) + spaces + "  " + dim("║"))
    print(dim("╚" + "═" * (w-2) + "╝"))

# ── Dashboard ──────────────────────────────────────────────────────────────────
def screen_dashboard():
    clr()
    w = W()
    print_header()
    print()

    st    = read_state()
    alive = runner_alive()

    if not st:
        print(dim("  No collaboration state found. Press [n] to start a task."))
        print()
    else:
        status    = st.get("status", "?")
        task_id   = st.get("task_id", "—")
        chain     = st.get("review_chain", [])
        step      = st.get("current_step", 0)
        locked    = st.get("locked_by") or "—"
        locked_at = st.get("locked_at")
        hoff_path = st.get("last_handoff")
        step_lbl  = f"{step+1}/{len(chain)}" if chain else "—"
        age       = fmt_elapsed(elapsed_secs(locked_at)) if locked_at else "—"

        chain_str = " → ".join(
            bold(yellow(a)) if a == locked else dim(a) for a in chain
        )

        # Row 1: status + runner indicator
        lft = f"  {bold('Status')}  {status_badge(status)}"
        rgt = f"{'🟢' if alive else '⚫'} runner"
        print(rjust(lft, rgt, w - 2))

        # Row 2: task + locked agent
        lft = f"  {bold('Task')}    {cyan(trunc(task_id, w//2 - 4))}"
        rgt = f"{yellow(locked)}  {dim(age)}"
        print(rjust(lft, rgt, w - 2))

        # Row 3: chain + step
        lft = f"  {bold('Chain')}   {chain_str}"
        rgt = dim(f"step {step_lbl}")
        print(rjust(lft, rgt, w - 2))
        print()

        # Result report shortcut if DONE
        result_file = TASKS / f"{task_id}-result.md"
        if status == "DONE" and result_file.exists():
            age_r = fmt_elapsed(int(time.time() - result_file.stat().st_mtime))
            print(f"  {green('📄')} Result report saved  {dim(f'({age_r} ago)')}  — press {bold('[r]')} to view")
            print()

        # Last handoff preview
        lines = last_handoff_lines(hoff_path)
        if lines:
            print(dim("  ┌── last handoff " + "─" * max(0, w - 20)))
            for line in lines:
                print(dim("  │ ") + italic(trunc(line, w - 6)))
            print(dim("  └" + "─" * (w - 4)))
            print()

    print(hr())
    lft = f"  {bold('[n]')} New task   {bold('[f]')} Feed   {bold('[h]')} History   {bold('[r]')} Results"
    rgt = f"{bold('[l]')} Live log   {bold('[a]')} Agents   {bold('[q]')} Quit  "
    print(rjust(lft, rgt, w))
    print(hr())

    try:    return input("  › ").strip().lower()
    except (KeyboardInterrupt, EOFError): return "q"

# ── Results screen ─────────────────────────────────────────────────────────────
def screen_results(cycle_id=None):
    """Per-agent contribution cards for the current or selected cycle."""
    clr()
    w  = W()
    st = read_state()
    if not cycle_id:
        cycle_id = st.get("cycle_id")

    handoffs = load_cycle_handoffs(cycle_id)
    task_id  = st.get("task_id") or (handoffs[0].get("task_id") if handoffs else "—")
    cyc_status = st.get("status", "—")

    print_header(f"Results  ›  {task_id}")
    print()

    if not handoffs:
        print(dim("  No handoffs found for this cycle."))
        print()
        try: input("  Press Enter.")
        except: pass
        return

    # Cycle summary
    done_at = st.get("updated_at", "")[:16].replace("T", " ")
    n = len(handoffs)
    print(f"  {status_badge(cyc_status)}  ·  {bold(str(n))} agent{'s' if n!=1 else ''}  ·  {dim(cycle_id or '?')}  ·  {dim(done_at)}")
    print()

    # Saved report path
    result_file = TASKS / f"{task_id}-result.md"
    if result_file.exists():
        age_r = fmt_elapsed(int(time.time() - result_file.stat().st_mtime))
        print(f"  {green('📄')} Saved: {dim(str(result_file))}  {dim(f'({age_r} ago)')}")
        print()

    # Card inner width: total - 2 indent - 2 border chars - 2 padding
    inner = w - 6

    def card_row(text):
        """Print one row inside a card box, truncated to inner width."""
        raw_len = ansi_len(text)
        pad = max(0, inner - raw_len)
        print(dim("  │ ") + text + " " * pad + dim("│"))

    for h in handoffs:
        agent   = (h.get("agent") or "?").upper()
        hstatus = h.get("status", "?")
        ts      = h.get("timestamp", "")[:16].replace("T", " ")
        files   = h.get("files_changed_list", [])
        icon    = agent_status_icon(hstatus)

        # ── Card header line ──────────────────────────────────────────────────
        title_text = f" {bold(agent)}  {icon}  {dim(ts)} "
        title_raw  = ansi_len(title_text)
        dash_fill  = max(0, inner - title_raw)
        print(dim("  ┌─") + title_text + dim("─" * dash_fill + "┐"))

        # ── Changes ──────────────────────────────────────────────────────────
        changes = h.get("s_changes", "").strip()
        if changes:
            # First paragraph, collapsed to one line
            first = changes.split("\n\n")[0].replace("\n", " ").strip()
            limit = inner - 12
            for seg in wrap_text(first, limit)[:3]:
                card_row(bold("Changes: ") + trunc(seg, limit))

        # ── Files ─────────────────────────────────────────────────────────────
        if files:
            fstr = "  ".join(files[:5]) + ("  …" if len(files) > 5 else "")
            card_row(dim("Files:   ") + cyan(trunc(fstr, inner - 10)))
        else:
            card_row(dim("Files:   ") + dim("(none recorded)"))

        # ── Verification ──────────────────────────────────────────────────────
        verif = h.get("s_verification", "").strip()
        if verif:
            first_v = verif.splitlines()[0].strip()
            card_row(dim("Verified:") + " " + trunc(first_v, inner - 11))

        # ── Concerns ──────────────────────────────────────────────────────────
        concerns = h.get("s_concerns", "").strip()
        if concerns:
            for i, line in enumerate(concerns.splitlines()[:3]):
                stripped = line.strip()
                if stripped:
                    prefix = yellow("⚠ ") if i == 0 else "  "
                    card_row(prefix + trunc(stripped, inner - 4))

        print(dim("  └" + "─" * (inner + 2) + "┘"))
        print()

    # Footer
    print(hr_s())
    print(f"  {bold('[v]')} view full handoff   {bold('[f]')} feed agents   {bold('[Enter]')} back")
    print(hr_s())

    try:   cmd = input("  › ").strip().lower()
    except (KeyboardInterrupt, EOFError): return

    if cmd == "v":
        print()
        for i, h in enumerate(handoffs, 1):
            agent = (h.get("agent") or "?").upper()
            print(f"  {bold(str(i))})  {agent}")
        try:
            sel = input("  View #: ").strip()
            idx = int(sel) - 1
            if 0 <= idx < len(handoffs):
                clr()
                print_header(Path(handoffs[idx]["path"]).name)
                print()
                for line in Path(handoffs[idx]["path"]).read_text().splitlines():
                    print("  " + trunc(line, w - 4))
                print()
                try: input("  Press Enter.")
                except: pass
        except (ValueError, KeyboardInterrupt): pass
        screen_results(cycle_id)

    elif cmd == "f":
        flow_feed()
        screen_results(cycle_id)

# ── Live watch ─────────────────────────────────────────────────────────────────
def _status_bar(agent, task_id, log_path=None):
    """Pinned status bar at the top of the live view."""
    w  = W()
    st = read_state()
    chain  = st.get("review_chain", [])
    step   = st.get("current_step", 0)
    locked = st.get("locked_by", agent)

    proc  = detect_agent_process(agent)
    cli   = proc["cli_name"] if proc else agent
    pid   = proc["pid"]      if proc else "—"
    esec  = proc["elapsed_s"] if proc else elapsed_secs(st.get("locked_at"))

    chain_fmt = " › ".join(
        bold(yellow(a)) if a == locked else dim(a) for a in chain
    )

    lft = f" {bold(green(cli))}  {dim(f'pid {pid}')}"
    rgt = f"{chain_fmt}  {dim(f'step {step+1}/{len(chain)}' if chain else '')} "
    print(dim("┌" + "─"*(w-2) + "┐"))
    print(dim("│") + rjust(lft, rgt, w-2) + dim("│"))

    lft2 = f" {dim('task')} {cyan(trunc(task_id or '?', w//2 - 8))}"
    rgt2 = f"{dim('running')} {bold(fmt_elapsed(esec))} "
    print(dim("│") + rjust(lft2, rgt2, w-2) + dim("│"))

    if log_path:
        sz   = log_path.stat().st_size
        lft3 = f" {dim('log')} {dim(log_path.name)}"
        rgt3 = f"{dim(f'{sz:,} bytes')} "
        print(dim("│") + rjust(lft3, rgt3, w-2) + dim("│"))

    print(dim("└" + "─"*(w-2) + "┘"))


def flow_watch(agent=None, task_id=None):
    """
    Live watcher — attaches to the agent's stdout the moment it exists.
    Runner streams stdout to file in real-time so tail -f works from line 1.
    """
    spin = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    waited = 0
    log    = None
    while waited < 120:
        log = find_current_log(agent)
        if log:
            break

        clr()
        w      = W()
        proc   = detect_agent_process(agent)
        is_run = runner_alive()
        s      = next(spin)

        print_header(f"Live  ›  {agent or '?'}  ›  {task_id or '?'}")
        print()

        if proc:
            cli  = proc["cli_name"]
            pid  = proc["pid"]
            esec = proc["elapsed_s"]
            print(f"  {bold(green(cli))}  {dim(f'pid {pid}')}")
            print(f"  {bold(s)} Starting up — log will appear any moment…")
            print()
            bar_w = w - 22
            fill  = int((esec % 40) / 40 * bar_w)
            bar   = green("█" * fill) + dim("░" * (bar_w - fill))
            print(f"  {dim('[')}{bar}{dim(']')}  {dim(fmt_elapsed(esec))}")
        elif is_run:
            print(f"  {bold(yellow(s))}  Runner active — waiting for {bold(agent)} to launch…")
        else:
            print(f"  {red('⚫')}  Runner is not running.")
            print(f"  Check {bold('[h]')} History or {bold('[r]')} Results.")

        print()
        print(dim("  Ctrl+C → dashboard"))

        try:
            time.sleep(1.5)
            waited += 2
        except KeyboardInterrupt:
            return

        if not is_run and not find_current_log(agent):
            st = read_state()
            new_agent = st.get("locked_by")
            if new_agent and new_agent != agent:
                agent   = new_agent
                task_id = st.get("task_id", task_id)
                continue
            try: input("\n  Runner stopped. Press Enter.")
            except: pass
            return

    if not log:
        print(red(f"\n  Timed out waiting for {agent} log. Try [r] Results."))
        try: input("  Press Enter.")
        except: pass
        return

    # Log exists — stream it live
    clr()
    _status_bar(agent, task_id, log)
    print(dim(f"  ↓  streaming live  —  Ctrl+C to return to dashboard"))
    print()

    try:
        subprocess.call(["tail", "-n", "200", "-f", str(log)])
    except KeyboardInterrupt:
        pass

    # After Ctrl+C — check what happened
    clr()
    st = read_state()
    if st.get("status") == "DONE":
        print(green(f"\n  ✓ Cycle DONE."))
    elif st.get("locked_by") != agent:
        new = st.get("locked_by", "?")
        print(yellow(f"\n  {agent} handed off → {bold(new)} is next."))
        try:
            ans = input(f"  Watch {new}? [y/n]: ").strip().lower()
            if ans == "y":
                flow_watch(agent=new, task_id=task_id)
                return
        except (KeyboardInterrupt, EOFError): pass
    else:
        print(dim(f"\n  Left live view. {agent} may still be running."))
    try: input("  Press Enter to return to dashboard.")
    except: pass

# ── New task ───────────────────────────────────────────────────────────────────
def flow_new_task():
    clr()
    w = W()
    print_header("New Task")
    print()

    # 1 — Name
    while True:
        try: raw = input("  Task name  (e.g. fix-recall-bug): ").strip()
        except (KeyboardInterrupt, EOFError): return
        slug = re.sub(r"[^a-zA-Z0-9\-]", "-", raw).strip("-")
        if slug: break
        print(red("  Name cannot be empty."))

    task_id = f"ARN-{slug}"
    print(f"  → ID: {cyan(task_id)}\n")

    # 2 — Description
    print(dim("  Describe the task. Blank line to finish."))
    lines = []
    while True:
        try:    line = input("  > ")
        except (EOFError, KeyboardInterrupt): break
        if not line and lines: break
        lines.append(line)
    desc = "\n".join(lines).strip()
    if not desc:
        print(red("\n  No description — cancelled.")); time.sleep(1); return

    # 3 — Agent chain
    print()
    print(hr_s())
    print(f"  {'#':<4} Agent")
    print(hr_s())
    for i, a in enumerate(AGENTS, 1):
        print(f"  {bold(str(i)):<4} {a}")
    print(hr_s())
    print(dim("  Enter numbers in run order, e.g.  1 3  or  1 2 3"))
    print()

    chain = []
    while not chain:
        try: sel = input("  Agents: ").strip()
        except (KeyboardInterrupt, EOFError): return
        seen = set()
        for tok in sel.split():
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(AGENTS) and AGENTS[idx] not in seen:
                    chain.append(AGENTS[idx]); seen.add(AGENTS[idx])
        if not chain: print(red("  Pick at least one (1 / 2 / 3)."))

    chain_str = " → ".join(chain)
    print(f"\n  Chain: {bold(chain_str)}\n")

    # 4 — Optional hint
    try: note = input("  Hint for agents? (Enter to skip): ").strip()
    except (KeyboardInterrupt, EOFError): note = ""

    # 5 — Confirm
    print()
    print(hr())
    print(f"  Task   {cyan(task_id)}")
    print(f"  Chain  {chain_str}")
    if note: print(f"  Hint   {trunc(note, w - 12)}")
    print(hr())
    try: go = input("  Launch? [y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError): go = "n"
    if go != "y":
        print(dim("  Cancelled.")); time.sleep(1); return

    # Write task file
    TASKS.mkdir(exist_ok=True)
    (TASKS / f"{task_id}.md").write_text(
        f"# ARN Task: {raw}\n\n## Task ID\n`{task_id}`\n\n"
        f"## Review Chain\n```\n{chain_str}\n```\n\n"
        f"## Description\n\n{desc}\n\n"
        "## Agent Instructions\n\n"
        "- Read `COLLAB.md` and `docs/collab-protocol.md` first.\n"
        "- Claim your step, do minimal correct work, write a handoff.\n"
        "- Run `python3 -m py_compile` on every Python file you change.\n\n"
        "## Verification\n```bash\n"
        "python3 -m pytest arn_v9/tests/ -x -q 2>&1 | tail -10\n```\n"
    )
    print(green("  ✓ Task file saved"))

    if note:
        run_cli("collab", "feed", "--agent", "all", "-m", note)
        print(green("  ✓ Hint sent to all agents"))

    print("  Launching runner…")
    subprocess.Popen(
        [PY, "-m", "arn_v9.collab_runner",
         "--repo-dir",     str(REPO),
         "--data-dir",     str(DATA_DIR),
         "--task-id",      task_id,
         "--review-chain", ",".join(chain),
         "--force", "--execute"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    print(green(f"  ✓ Runner started — {chain_str}\n"))
    print(dim("  Entering live view — Ctrl+C to return to dashboard"))
    time.sleep(1)
    flow_watch(agent=chain[0], task_id=task_id)

    # After watching — auto-offer results if cycle finished
    st_after = read_state()
    if st_after.get("status") == "DONE":
        try:
            ans = input(f"\n  {green('✓ Cycle complete!')} View results? [y/n]: ").strip().lower()
            if ans == "y":
                screen_results()
        except (KeyboardInterrupt, EOFError): pass

# ── Feed ───────────────────────────────────────────────────────────────────────
def flow_feed():
    clr()
    print_header("Feed Message")
    print()
    print(dim("  Inject a note into an agent's next prompt.\n"))

    targets = AGENTS + ["all"]
    for i, t in enumerate(targets, 1):
        print(f"  {bold(str(i))})  {t}")
    print()

    try: sel = input("  Target: ").strip()
    except (KeyboardInterrupt, EOFError): return
    try:   target = targets[int(sel) - 1]
    except (ValueError, IndexError):
        print(red("  Invalid.")); time.sleep(1); return

    try: msg = input(f"\n  Message → {bold(target)}: ").strip()
    except (KeyboardInterrupt, EOFError): return
    if not msg:
        print(dim("  Cancelled.")); time.sleep(1); return

    run_cli("collab", "feed", "--agent", target, "-m", msg)
    print(green(f"\n  ✓ Sent."))
    try: input("  Press Enter.")
    except: pass

# ── History ────────────────────────────────────────────────────────────────────
def screen_history():
    clr()
    w = W()
    print_header("Task History")
    print()

    d = COLLAB / "handoffs"
    if not d.exists() or not list(d.glob("*.md")):
        print(dim("  No handoffs yet.")); time.sleep(1); return

    # Group handoffs by cycle_id, most recent first
    all_files = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    cycles = {}   # cycle_id -> list of parsed handoffs
    order  = []   # cycle_ids in recency order
    for f in all_files:
        h   = parse_handoff(f)
        cid = h.get("cycle_id", "unknown")
        if cid not in cycles:
            cycles[cid] = []
            order.append(cid)
        cycles[cid].append(h)

    # Render table
    print(hr_s())
    hdr = f"  {'#':<4} {'Task':<32} {'Status':<16} {'Agents':<22} When"
    print(hdr)
    print(hr_s())

    cycle_list = []
    for cid in order[:12]:
        hs = sorted(cycles[cid], key=lambda h: h.get("timestamp", ""))
        tid = hs[0].get("task_id", "?") if hs else "?"

        # Status summary
        statuses = [h.get("status", "") for h in hs]
        if all(s in ("complete", "no_issues") for s in statuses):
            st_str = green("✅ DONE")
        elif any(s == "blocked" for s in statuses):
            st_str = red("🔴 BLOCKED")
        elif any(s == "needs_review" for s in statuses):
            st_str = yellow("⚠️  REVIEW")
        else:
            st_str = dim(", ".join(set(s for s in statuses if s))[:14])

        agents_str = " → ".join(h.get("agent", "?")[:5] for h in hs)
        last_ts    = max((h.get("timestamp", "") for h in hs), default="")
        age        = fmt_elapsed(elapsed_secs(last_ts)) if last_ts else "?"

        i = len(cycle_list) + 1
        cycle_list.append(cid)
        print(f"  {bold(str(i)):<4} {trunc(tid, 32):<32} {st_str:<16} {dim(agents_str):<22} {dim(age)}")

    print(hr_s())
    print()

    try: sel = input("  View cycle # for results (Enter to go back): ").strip()
    except (KeyboardInterrupt, EOFError): return

    try:
        idx = int(sel) - 1
        if 0 <= idx < len(cycle_list):
            screen_results(cycle_list[idx])
    except ValueError: pass

# ── Live log shortcut ──────────────────────────────────────────────────────────
def flow_live_log():
    st    = read_state()
    agent = st.get("locked_by")
    task  = st.get("task_id")
    flow_watch(agent=agent, task_id=task)

# ── Agents health ──────────────────────────────────────────────────────────────
def screen_agents():
    clr()
    print_header("Agent Health")
    print()
    r = run_cli("collab", "agents")
    out = (r.stdout or r.stderr or "(no output)").strip()
    for line in out.splitlines():
        print("  " + trunc(line, W() - 4))
    print()
    try: input("  Press Enter.")
    except: pass

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    dispatch = {
        "n": flow_new_task,
        "f": flow_feed,
        "h": screen_history,
        "r": screen_results,
        "l": flow_live_log,
        "a": screen_agents,
    }
    while True:
        choice = screen_dashboard()
        if choice == "q":
            clr(); print(dim("  Goodbye.\n")); break
        fn = dispatch.get(choice)
        if fn: fn()

if __name__ == "__main__":
    main()
