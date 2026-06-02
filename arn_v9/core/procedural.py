"""
ARN v9 Procedural Memory System.

Synthesizes session activity into reusable procedure memories that surface
automatically via the same retrieval pipeline as episodic memories.
No LLM calls — entirely algorithmic.

A procedure captures the GOAL → STEPS → FAILURES → VERIFICATION pattern
from a session complex enough to be worth remembering. Complexity is scored
by tool diversity and error corrections (pivoting after failure is the
strongest signal of a procedure worth keeping).
"""

import re
import time
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("arn.procedural")

# Error indicators in tool_result content.
# No leading \b so patterns match inside compound words like "ImportError",
# "FileNotFoundError", "CalledProcessError", etc.
_ERROR_RE = re.compile(
    r'(?:error|exception|traceback|failed?|not found|permission denied|'
    r'cannot|unable|undefined|invalid|timeout|no such file|does not exist|'
    r'refused|denied|killed|aborted|crash|exit code [1-9]|returncode=[1-9])',
    re.IGNORECASE,
)

COMPLEXITY_THRESHOLD = 8.0  # minimum score to trigger extraction (tunable)

# Environment hint patterns for CONTEXT section
_ENV_HINTS = [
    (r'\b(?:python3?|pip3?|pytest)\b', 'Python'),
    (r'\b(?:npm|node|typescript|tsx?)\b', 'Node.js/TypeScript'),
    (r'\b(?:git|github|pr|pull request)\b', 'Git'),
    (r'\b(?:docker|compose|container)\b', 'Docker'),
    (r'\b(?:bash|shell|zsh|sh)\b', 'Shell'),
    (r'\b(?:sqlite|postgres|mysql|redis)\b', 'Database'),
    (r'\b(?:curl|http|api|rest|endpoint)\b', 'HTTP/API'),
    (r'\b(?:rust|cargo)\b', 'Rust'),
]


# =========================================================
# A1 — COMPLEXITY SCORING
# =========================================================

def compute_task_complexity(session_episodes: List[Dict[str, Any]]) -> float:
    """
    Score session complexity to decide if a procedure is worth extracting.

    Composite formula:
        complexity = (tool_calls × 0.3) + (tool_diversity × 2.0)
                   + (error_corrections × 3.0) + (turns × 0.1)

    Error corrections (pivot after failure) carry the highest weight because
    they capture hard-won knowledge about what doesn't work.

    Returns 0.0 if no tool calls exist in the session.
    """
    episodes = sorted(
        [e for e in session_episodes
         if e.get('role') in ('tool_call', 'tool_result', 'user', 'assistant')],
        key=lambda e: e.get('created_at', 0),
    )

    tool_calls = [e for e in episodes if e.get('role') == 'tool_call']
    if not tool_calls:
        return 0.0

    tool_call_count = len(tool_calls)

    # Unique tool names: parse "Tool call: exec(...)" or "exec(...)" or bare "exec"
    tools_used: set = set()
    for tc in tool_calls:
        m = re.match(r'(?:Tool call:\s*)?([A-Za-z_][A-Za-z0-9_]*)', tc.get('content', ''))
        if m:
            tools_used.add(m.group(1).lower())
    tool_diversity = len(tools_used)

    # Error corrections: tool_result with error → subsequent tool_call with different
    # tool OR different parameters. Different parameters covers the common case where
    # the agent retries the same tool (e.g. exec) with a different command.
    error_corrections = 0
    prev_tool: Optional[str] = None
    prev_call_content: Optional[str] = None
    prev_was_error = False

    for ep in episodes:
        role = ep.get('role')
        if role == 'tool_call':
            m = re.match(r'(?:Tool call:\s*)?([A-Za-z_][A-Za-z0-9_]*)', ep.get('content', ''))
            cur_tool = m.group(1).lower() if m else None
            cur_content = ep.get('content', '')
            if prev_was_error and cur_tool:
                if cur_tool != prev_tool:
                    error_corrections += 1  # different tool
                elif cur_content != prev_call_content:
                    error_corrections += 1  # same tool, different args
            prev_tool = cur_tool
            prev_call_content = cur_content
            prev_was_error = False
        elif role == 'tool_result':
            prev_was_error = bool(_ERROR_RE.search(ep.get('content', '')))

    turns = sum(1 for e in episodes if e.get('role') == 'user')

    complexity = (
        tool_call_count * 0.3
        + tool_diversity * 2.0
        + error_corrections * 3.0
        + turns * 0.1
    )
    return round(complexity, 3)


# =========================================================
# A2 — PROCEDURE EXTRACTION
# =========================================================

def extract_procedure(
    storage,
    embedder,
    session_episodes: List[Dict[str, Any]],
    session_id: str,
    complexity_threshold: float = COMPLEXITY_THRESHOLD,
) -> Optional[int]:
    """
    Synthesize a procedural memory from session episodes if the session is
    complex enough to be worth capturing.

    Builds structured content from episode data (no LLM call). The procedure
    is stored as a regular episode with role='procedural' so it surfaces
    through the normal hybrid retrieval pipeline.

    Returns the new episode ID, or None if complexity is below threshold.
    """
    episodes = sorted(session_episodes, key=lambda e: e.get('created_at', 0))
    complexity = compute_task_complexity(episodes)

    if complexity < complexity_threshold:
        logger.debug(
            f"Session {session_id} complexity {complexity:.1f} < "
            f"threshold {complexity_threshold} — skipping extraction"
        )
        return None

    # GOAL: first user message in session
    goal = next(
        (e['content'] for e in episodes if e.get('role') == 'user'),
        'Unknown task',
    )[:500]

    # Build (tool_call, tool_result) pairs in order
    pairs: List[Tuple[Dict, Dict]] = []
    pending_call: Optional[Dict] = None
    for ep in episodes:
        role = ep.get('role')
        if role == 'tool_call':
            pending_call = ep
        elif role == 'tool_result' and pending_call is not None:
            pairs.append((pending_call, ep))
            pending_call = None

    # Classify pairs as failures or successful steps
    last_error_idx = -1
    for i, (_, tr) in enumerate(pairs):
        if _ERROR_RE.search(tr.get('content', '')):
            last_error_idx = i

    failures: List[Tuple[str, str]] = []
    for i, (tc, tr) in enumerate(pairs):
        if _ERROR_RE.search(tr.get('content', '')):
            failures.append((tc['content'][:200], tr['content'][:200]))

    # Successful path = pairs after the last error
    successful_steps = [
        tc['content'][:200]
        for i, (tc, _) in enumerate(pairs)
        if i > last_error_idx
    ]
    if not successful_steps and pairs:
        # All steps may have been clean — use all of them
        successful_steps = [tc['content'][:200] for tc, _ in pairs]

    # Overall session success: last tool_result was not an error
    session_succeeded = True
    if pairs:
        _, last_tr = pairs[-1]
        session_succeeded = not bool(_ERROR_RE.search(last_tr.get('content', '')))

    # VERIFICATION: last non-error tool_result
    verification = ''
    for tc, tr in reversed(pairs):
        if not _ERROR_RE.search(tr.get('content', '')):
            verification = tr['content'][:200]
            break

    # CONTEXT: environment hints from all episode content
    all_text = ' '.join(e.get('content', '') for e in episodes)
    env_hints = [
        label
        for pattern, label in _ENV_HINTS
        if re.search(pattern, all_text, re.IGNORECASE)
    ]

    # Ordered unique tool chain
    tool_chain: List[str] = []
    seen: set = set()
    for tc, _ in pairs:
        m = re.match(r'(?:Tool call:\s*)?([A-Za-z_][A-Za-z0-9_]*)', tc.get('content', ''))
        if m:
            t = m.group(1).lower()
            if t not in seen:
                tool_chain.append(t)
                seen.add(t)

    # Assemble structured content
    lines = [f"GOAL: {goal}", '']

    lines.append('STEPS:')
    if successful_steps:
        for idx, step in enumerate(successful_steps, 1):
            lines.append(f'  {idx}. {step}')
    else:
        lines.append('  (no clear successful path identified)')
    lines.append('')

    if failures:
        lines.append('FAILURES:')
        for call_c, result_c in failures[:5]:
            lines.append(f'  - Tried: {call_c}')
            lines.append(f'    Result: {result_c[:120]}')
        lines.append('')

    if verification:
        lines.append(f'VERIFICATION: {verification}')
        lines.append('')

    if env_hints:
        lines.append(f'CONTEXT: {", ".join(env_hints)}')

    content = '\n'.join(lines).strip()

    vec = embedder.encode(content, mode='passage')
    source_ids = [e['id'] for e in session_episodes if e.get('id')]

    ep_id = storage.store_episode(
        content=content,
        vector=vec,
        importance=0.85,
        source='system',
        role='procedural',
        session_id=session_id,
        metadata={
            'source_session': session_id,
            'source_episode_ids': source_ids[:50],
            'complexity_score': complexity,
            'tool_chain': tool_chain,
            'success': session_succeeded,
            'effectiveness_score': 1.0,
        },
    )

    logger.info(
        f"Extracted procedure {ep_id} from session {session_id} "
        f"(complexity={complexity:.1f}, tools={tool_chain}, success={session_succeeded})"
    )
    return ep_id


# =========================================================
# A3 — PROCEDURE SELF-IMPROVEMENT VIA SUPERSEDES
# =========================================================

def find_similar_procedures(
    storage,
    embedder,
    content: str,
    threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """
    Find existing active procedural memories with embedding similarity
    above threshold. Returns dicts with an extra '_similarity' key.
    """
    vec = embedder.encode(content, mode='query')
    candidates = storage.knn_search(vec, top_k=20)
    if not candidates:
        return []

    ids = [eid for eid, _ in candidates]
    episodes = storage.get_episodes_by_ids(ids)

    result = []
    ep_vecs, ep_ids = storage.get_episode_vectors(ids)
    id_to_vec = {eid: ep_vecs[i] for i, eid in enumerate(ep_ids)}

    for ep in episodes:
        if ep.get('role') != 'procedural':
            continue
        if ep.get('invalidated_at') is not None:
            continue
        if ep.get('superseded_by') is not None:
            continue
        ep_vec = id_to_vec.get(ep['id'])
        if ep_vec is None:
            continue
        sim = float(ep_vec @ vec)
        if sim >= threshold:
            result.append({**ep, '_similarity': sim})

    result.sort(key=lambda x: x['_similarity'], reverse=True)
    return result


# =========================================================
# A4 — EFFECTIVENESS TRACKING
# =========================================================

def compute_session_error_rate(session_episodes: List[Dict[str, Any]]) -> float:
    """Return the ratio of failing tool_results to total tool_results (0.0–1.0)."""
    results = [e for e in session_episodes if e.get('role') == 'tool_result']
    if not results:
        return 0.0
    errors = sum(1 for e in results if _ERROR_RE.search(e.get('content', '')))
    return errors / len(results)


def compute_effectiveness_deltas(
    injected_procedure_ids: List[int],
    error_rate: float,
) -> Dict[int, float]:
    """
    Derive effectiveness delta for each injected procedure based on the
    session's tool error rate.

    Rules:
    - error_rate < 0.20  → +0.1 boost (session went well)
    - error_rate > 0.50  → -0.2 reduction (session struggled)
    - 0.20–0.50          → no change
    """
    if not injected_procedure_ids:
        return {}
    deltas: Dict[int, float] = {}
    for proc_id in injected_procedure_ids:
        if error_rate < 0.20:
            deltas[proc_id] = 0.1
        elif error_rate > 0.50:
            deltas[proc_id] = -0.2
    return deltas


def apply_effectiveness_updates(
    storage,
    deltas: Dict[int, float],
    review_threshold: float = 0.3,
) -> List[int]:
    """
    Write updated effectiveness_score to each procedure's metadata.
    Returns IDs of procedures that crossed below review_threshold and
    should be flagged in the review queue.
    """
    flagged: List[int] = []
    for ep_id, delta in deltas.items():
        ep = storage.get_episode(ep_id)
        if ep is None:
            continue
        meta = ep.get('metadata') or {}
        current = float(meta.get('effectiveness_score', 1.0))
        new_score = max(0.1, min(2.0, current + delta))
        meta['effectiveness_score'] = round(new_score, 3)
        storage.update_episode(ep_id, {'metadata': json.dumps(meta)})
        logger.debug(f"Procedure {ep_id}: effectiveness {current:.2f} → {new_score:.2f}")
        if new_score < review_threshold and current >= review_threshold:
            flagged.append(ep_id)
    return flagged


# =========================================================
# A5 — DEEP REFLECT (CURATOR)
# =========================================================

def deep_reflect_procedures(
    storage,
    embedder,
    stale_days: int = 30,
    archive_days: int = 60,
    archive_importance: float = 0.15,
    dup_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Periodic curator pass for procedural memories. Designed to run every
    N sessions (caller controls frequency) or on demand.

    Steps:
    1. Stale detection   — zero access after stale_days → importance → 0.1
    2. Duplicate merging — sim > dup_threshold pairs → keep better, supersede other
    3. Archival          — importance < archive_importance + age > archive_days → set valid_until

    Returns a stats dict.
    """
    now = time.time()
    stats = {
        'total_procedures': 0,
        'active_procedures': 0,
        'stale_marked': 0,
        'duplicates_merged': 0,
        'archived': 0,
        'avg_effectiveness': 0.0,
        'most_used': None,
        'least_effective': None,
    }

    # Fetch all procedural memories
    all_procs = [
        ep for ep in storage.get_all_episodes(consolidated=None)
        if ep.get('role') == 'procedural'
    ]
    stats['total_procedures'] = len(all_procs)

    active = [
        ep for ep in all_procs
        if ep.get('invalidated_at') is None
        and ep.get('superseded_by') is None
        and (ep.get('valid_until') is None or ep['valid_until'] > now)
    ]
    stats['active_procedures'] = len(active)

    if not active:
        return stats

    # --- Step 1: Stale detection ---
    stale_cutoff = now - stale_days * 86400
    for ep in active:
        if ep.get('access_count', 0) == 0 and ep['created_at'] < stale_cutoff:
            current_imp = ep['importance']
            if current_imp > 0.1:
                storage.update_episode(ep['id'], {'importance': 0.1})
                stats['stale_marked'] += 1

    # --- Step 2: Duplicate merging ---
    if len(active) >= 2:
        vecs, vec_ids = storage.get_episode_vectors([ep['id'] for ep in active])
        id_to_idx = {eid: i for i, eid in enumerate(vec_ids)}
        id_to_ep = {ep['id']: ep for ep in active}

        merged: set = set()
        for i in range(len(vec_ids)):
            if vec_ids[i] in merged:
                continue
            for j in range(i + 1, len(vec_ids)):
                if vec_ids[j] in merged:
                    continue
                sim = float(vecs[i] @ vecs[j])
                if sim < dup_threshold:
                    continue
                ep_i = id_to_ep[vec_ids[i]]
                ep_j = id_to_ep[vec_ids[j]]
                # Keep the one with higher effectiveness_score, then higher access_count
                eff_i = float((ep_i.get('metadata') or {}).get('effectiveness_score', 1.0))
                eff_j = float((ep_j.get('metadata') or {}).get('effectiveness_score', 1.0))
                if eff_i >= eff_j:
                    keep, discard = ep_i, ep_j
                else:
                    keep, discard = ep_j, ep_i
                storage.supersede_episode(discard['id'], keep['id'])
                merged.add(discard['id'])
                stats['duplicates_merged'] += 1
                logger.debug(
                    f"Merged duplicate procedures: {discard['id']} → {keep['id']} "
                    f"(sim={sim:.3f})"
                )

    # Refresh active list after merges
    active = [
        ep for ep in active
        if ep.get('invalidated_at') is None
        and ep.get('superseded_by') is None
        and ep['id'] not in merged  # type: ignore[possibly-undefined]
    ]

    # --- Step 3: Archival ---
    archive_cutoff = now - archive_days * 86400
    for ep in active:
        if ep['importance'] < archive_importance and ep['created_at'] < archive_cutoff:
            storage.update_episode(ep['id'], {'valid_until': now})
            stats['archived'] += 1

    # --- Stats summary ---
    eff_scores = []
    most_used = None
    least_eff = None
    max_access = -1
    min_eff = float('inf')

    for ep in active:
        meta = ep.get('metadata') or {}
        eff = float(meta.get('effectiveness_score', 1.0))
        eff_scores.append(eff)
        ac = ep.get('access_count', 0)
        if ac > max_access:
            max_access = ac
            most_used = {
                'content_preview': ep['content'][:80],
                'access_count': ac,
            }
        if eff < min_eff:
            min_eff = eff
            least_eff = {
                'content_preview': ep['content'][:80],
                'effectiveness_score': eff,
            }

    stats['avg_effectiveness'] = round(sum(eff_scores) / len(eff_scores), 3) if eff_scores else 0.0
    stats['most_used'] = most_used
    stats['least_effective'] = least_eff

    return stats
