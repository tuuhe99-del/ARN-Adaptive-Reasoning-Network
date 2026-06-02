# ARN v0.10.0 — Full Build Report

This document covers everything built across the full overhaul session: what was removed, what replaced it, how each system works, and why the decisions were made.

---

## Table of Contents

1. [What Was Removed and Why](#1-what-was-removed-and-why)
2. [Storage Layer — sqlite-vec + FTS5](#2-storage-layer--sqlite-vec--fts5)
3. [Retrieval Pipeline — RRF + MMR + Score-Gap](#3-retrieval-pipeline--rrf--mmr--score-gap)
4. [Temporal Intelligence — Bi-temporal + Supersedes](#4-temporal-intelligence--bi-temporal--supersedes)
5. [Entity Extraction](#5-entity-extraction)
6. [Post-Session Reflection](#6-post-session-reflection)
7. [OpenClaw Integration](#7-openclaw-integration)
8. [Schema v7 — Sessions + Role-Aware Episodes](#8-schema-v7--sessions--role-aware-episodes)
9. [Plugin API + Daemon Mode](#9-plugin-api--daemon-mode)
10. [TypeScript Plugin](#10-typescript-plugin)
11. [CLI Extensions](#11-cli-extensions)
12. [Integration Tests](#12-integration-tests)
13. [README Overhaul](#13-readme-overhaul)
14. [Bug Fixes](#14-bug-fixes)
15. [Commit History](#15-commit-history)

---

## 1. What Was Removed and Why

### Cortical Columns (removed in Phase 0)

**What they were:** 8 domain-specific "columns" (coding, security, system, data, language, tool_use, general, meta) that each maintained their own prototype embeddings and tried to route memories to the most appropriate domain.

**Why removed:** The domain-routing required calibrating 8 × N parameters across domains that no one was actually tuning. In practice, every episode ended up in `general` or was duplicated across domains. The prototype vectors drifted randomly and provided no measurable benefit to recall quality. The complexity was real; the benefit was not.

**What replaced it:** A single flat embedding space. The retrieval pipeline's diversity (RRF + MMR) handles topical separation without explicit domain labeling.

---

### Embedding Tiers (removed in Phase 0)

**What they were:** 4 selectable model tiers (nano/base/large/xl) with different model sizes, embedding dimensions, and calibration constants. An env var `ARN_EMBEDDING_TIER` selected which one to use.

**Why removed:** All 4 tiers scored identically (7/7) on the internal benchmark suite. The code paths were 4× the complexity for zero measurable quality difference at the recall task ARN actually does (similarity-based retrieval from a small-to-medium personal memory store). Tier switching also created fingerprint mismatches when the env var changed after data was stored.

**What replaced it:** Single fixed model — `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, ~22MB download, Pi-friendly. If a different model is needed, pass `embedding_fn` to `EmbeddingEngine` or `ARNv9`.

---

### Word-Overlap Contradiction Detection (removed in Phase 0)

**What it was:** A module (`contradictions.py`) that compared new episodes against existing ones. When cosine similarity exceeded 0.85 AND word-overlap (Jaccard on `.split()` tokens) was below 50%, it fired a "contradiction" supersession automatically — without user review.

**Why removed:** This produced false positives on paraphrases and style variants. "The server uses Redis" and "Redis is used as the cache layer" would be flagged as contradictory because high semantic similarity + different words. The supersession chain fired mid-`perceive()`, invalidating memories the user never consented to remove.

**What replaced it:** Supersessions still happen, but they are detected in `reflect()` after a session ends, queued for review, and only applied when the user resolves the review item explicitly. The heuristic (sim > 0.85 + overlap < 40%) is the same — the difference is human-in-the-loop before action is taken.

---

### Manual `time_context` (removed in Phase 0)

**What it was:** A `time_context` parameter on `store()` and `recall()` that accepted strings like `"past"`, `"current"`, `"future"` and re-ranked results to surface temporally-tagged memories. There were keyword lists for each category (`_PAST_KEYWORDS`, `_CURRENT_KEYWORDS`, `_FUTURE_KEYWORDS`).

**Why removed:** The burden was on callers to know what temporal category their memory belonged to. In practice no callers used it. The bi-temporal columns (`valid_from`, `valid_until`) now handle temporality structurally — superseded facts are marked with `valid_until = NOW` and excluded from default recall automatically. No manual tagging needed.

---

### Auto-Consolidation at 256 Episodes (removed in Phase 0)

**What it was:** `perceive()` checked the count of unconsolidated episodes after every store. When it hit 256, it triggered a consolidation run inline — clustering similar episodes and merging clusters into semantic nodes.

**Why removed:** This ran mid-session, mid-conversation, with no user awareness. Lossy merging mid-session corrupted memories that were still actively being referenced. The threshold was also arbitrary.

**What replaced it:** Consolidation is entirely opt-in. Call `arn consolidate` from the CLI, `arn.consolidate()` in Python, or it runs at the end of `arn reflect`. It also has tighter criteria: only merges episodes with sim > 0.90 (near-identical), and skips pinned episodes, episodes in the review queue, and episodes created in the last 7 days.

---

## 2. Storage Layer — sqlite-vec + FTS5

### Before: Memmap Files

The old system stored embedding vectors in `.npy` memory-mapped files alongside the SQLite database. This meant two failure modes: the SQLite file and the memmap file could desync (e.g. a crash mid-write). It also required a `vec_index` integer column in every episode row to track the row's position in the memmap array. Expanding the memmap required careful atomic writes.

### After: Everything in SQLite

All vectors are stored inside SQLite using the `sqlite-vec` extension, which adds a `vec0` virtual table type. The schema:

```sql
CREATE VIRTUAL TABLE episode_embeddings
USING vec0(embedding float[384]);
```

A `vec0` table is indexed — queries use approximate KNN, not a full scan. Every time an episode is stored:

```python
conn.execute(
    "INSERT OR REPLACE INTO episode_embeddings(rowid, embedding) VALUES (?, ?)",
    (ep_id, vector.astype(np.float32).tobytes())
)
```

The `rowid` matches the episode `id`, so a join is a single integer lookup. KNN search:

```python
rows = conn.execute(
    "SELECT rowid, distance FROM episode_embeddings "
    "WHERE embedding MATCH ? AND k=?",
    (query_vec_bytes, top_k)
).fetchall()
# distance is L2; invert to a similarity-like score
results = [(r[0], 1.0 / (1.0 + r[1])) for r in rows]
```

### FTS5 Full-Text Index

The episodes table has a companion FTS5 virtual table:

```sql
CREATE VIRTUAL TABLE episodes_fts
USING fts5(
    content,
    content='episodes',
    content_rowid='id',
    tokenize='porter ascii'
);
```

`content='episodes'` makes this a "content table" — FTS5 stores only the index, not the actual text. `content_rowid='id'` links it back to the episodes row. `tokenize='porter ascii'` applies Porter stemming so "running", "runs", "ran" all match the same token.

Three triggers keep the FTS index in sync with the episodes table automatically — insert, delete, update. FTS search returns BM25 rank (negative float, lower = more relevant):

```python
rows = conn.execute(
    "SELECT rowid, rank FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?",
    (query, top_k)
).fetchall()
# Flip negative rank, normalize to [0, 1]
results = [(r[0], float(-r[1])) for r in rows]
```

### Why Both?

Vector search finds semantically similar content — good for conceptual queries, bad for exact terms. FTS5 finds keyword matches — good for exact terms, proper nouns, version strings, identifiers, bad for synonyms. Neither alone is sufficient. The fusion layer (see Section 3) combines them.

---

## 3. Retrieval Pipeline — RRF + MMR + Score-Gap

### Step 1: Three Ranked Lists

Every `recall()` call produces three independent ranked lists:

- **Vector KNN** — top K×4 episodes by L2 distance in the embedding space
- **FTS5** — top K×4 episodes by BM25 text rank, filtered to active episode IDs (not invalidated, not expired)
- **Entity matching** — top K×4 episodes containing entities extracted from the query (proper nouns, identifiers, etc.)

### Step 2: Reciprocal Rank Fusion

RRF combines any number of ranked lists without knowing their score scales or requiring weight tuning. For each episode, its RRF score is the sum of its reciprocal ranks across all lists:

```python
def fuse_rrf(vec_results, fts_results, entity_results=None, k=60):
    # k=60 is a standard smoothing constant from the RRF paper
    vec_ranks  = {eid: i+1 for i, (eid, _) in enumerate(vec_results)}
    fts_ranks  = {eid: i+1 for i, (eid, _) in enumerate(fts_results)}
    ent_ranks  = {eid: i+1 for i, (eid, _) in enumerate(entity_results or [])}
    all_ids    = set(vec_ranks) | set(fts_ranks) | set(ent_ranks)
    fallback   = len(all_ids) + 1  # rank if not present in that list

    return {
        eid: sum(1.0 / (k + ranks.get(eid, fallback))
                 for ranks in [vec_ranks, fts_ranks, ent_ranks])
        for eid in all_ids
    }
```

An episode that ranks #1 in all three lists gets the maximum possible RRF score. An episode that only appears in one list gets a lower score — but still appears, which is the point: each retrieval method catches things the others miss.

### Step 3: Composite Score

After RRF, each episode is scored with additional signals:

```
score = rrf_score
      + recency_decay  × 0.3     (14-day half-life exponential decay)
      + importance     × 0.15    (stored 0–1 value from perceive())
      + freq_boost               (log(1 + access_count) × 0.05)
      + pin_boost                (0.15 if pinned; also bypasses recency decay)
```

**Recency decay** — `exp(-log(2) / 14 * days_old)`. An episode stored today scores 1.0; one stored 14 days ago scores 0.5; 28 days ago scores 0.25. Pinned episodes always use recency = 1.0.

**Frequency boost** — episodes that have been recalled many times are likely useful. The `access_count` column increments every time an episode appears in a recall result.

### Step 4: MMR Reranking

After sorting by composite score, the top K×2 results go through Maximal Marginal Relevance reranking. MMR alternates between relevance and diversity:

```python
def mmr_rerank(query_emb, results, result_vecs, lambda_param=0.7, top_k):
    query_sims = result_vecs @ query_emb   # relevance to query
    selected, remaining = [], list(range(len(results)))

    while len(selected) < top_k and remaining:
        best, best_score = None, float('-inf')
        for i in remaining:
            # redundancy = max similarity to already-selected results
            redundancy = max(result_vecs[selected] @ result_vecs[i]) if selected else 0.0
            # MMR score balances relevance vs redundancy
            score = lambda_param * query_sims[i] - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score, best = score, i
        selected.append(best)
        remaining.remove(best)

    return [results[i] for i in selected]
```

`lambda_param=0.7` weights relevance 70% and diversity 30%. This means if you have 5 near-identical results, MMR will pick the most relevant one and then look for something different for slot #2.

### Step 5: Score-Gap Cutoff (Adaptive Threshold)

Instead of a fixed similarity threshold (e.g. "only return results with score > 0.3"), the cutoff finds the largest relative gap in the score distribution:

```python
def score_gap_cutoff(results, top_k=5, min_gap_ratio=0.15):
    sorted_r = sorted(results, key=lambda r: r['score'], reverse=True)
    scores = [r['score'] for r in sorted_r]
    score_range = scores[0] - scores[-1]

    best_gap, best_cut = 0.0, top_k
    for i in range(1, min(len(scores), top_k + 5)):
        gap = (scores[i-1] - scores[i]) / score_range
        if gap > best_gap and gap >= min_gap_ratio:
            best_gap, best_cut = gap, i

    return sorted_r[:max(1, best_cut)]
```

If the score distribution is `[0.9, 0.85, 0.82, 0.3, 0.28]`, the gap between 0.82 and 0.3 (relative gap = 0.52 / 0.62 = 84%) is much larger than any other gap. The cutoff fires there, returning 3 results rather than all 5.

Fixed thresholds fail in two directions: too tight cuts relevant results when the query is about a niche topic where even the best match has moderate similarity; too loose includes noise when results drop off a cliff after rank 2. The gap method adapts to each query's actual distribution.

---

## 4. Temporal Intelligence — Bi-temporal + Supersedes

### Bi-temporal Columns

Every episode has two time columns beyond the standard `created_at`:

```sql
valid_from   REAL    -- when this fact became true (defaults to created_at)
valid_until  REAL    -- when this fact stopped being true (NULL = still valid)
```

Default recall only returns episodes where `valid_until IS NULL OR valid_until > now`. Historical recall (`include_historical=True`) returns everything.

This means the database is a full history, not a mutable store. No fact is ever truly deleted — only marked as no longer valid.

### Supersedes Chains

When a new fact contradicts an old one, `supersede_episode(old_id, new_id)` links them:

```python
def supersede_episode(self, old_id: int, new_id: int):
    now = time.time()
    # Mark old episode as no longer valid
    conn.execute(
        "UPDATE episodes SET superseded_by=?, valid_until=?, invalidated_at=? WHERE id=?",
        (new_id, now, now, old_id)
    )
    # Link new episode back to what it replaced
    conn.execute("UPDATE episodes SET supersedes=? WHERE id=?", (old_id, new_id))
```

Walking the chain with `get_history(episode_id)` follows both the `supersedes` pointer (backwards) and `superseded_by` pointer (forwards), returning all versions of a fact sorted by time.

### Pinned Episodes

Pinned episodes have `pinned = 1` in the episodes table. The retrieval pipeline checks this in three places:
1. **Scoring** — recency score forced to 1.0, plus a flat +0.15 boost
2. **Consolidation** — filtered out before clustering; never merged into semantic nodes
3. **Supersession detection** — skipped when scanning for contradictions in `reflect()`

```python
arn.pin(episode_id)    # sets pinned=1
arn.unpin(episode_id)  # sets pinned=0
```

---

## 5. Entity Extraction

`arn_v9/core/entities.py` extracts structured named entities from episode text using regex patterns:

| Entity type | Pattern targets |
|-------------|----------------|
| `proper_noun` | Capitalized words / multi-word names (excluding sentence-start stopwords) |
| `quoted_string` | Text inside double quotes, 2–60 chars |
| `code_identifier` | `dotted.paths.like.this`, `snake_case_identifiers` |
| `url` | `https?://...` |
| `file_path` | `/absolute/paths` and `~/home/relative/paths` |
| `number_unit` | `42ms`, `8GB`, `3.14rad` — numbers followed by unit abbreviations |

Entities are stored in a separate `entities` table linked to the episode. At recall time, the query is also entity-extracted and those entities are used to boost scores for episodes sharing the same named entities.

This matters for recall quality because: if you ask "what did we decide about Redis?", the entity "Redis" should strongly boost results mentioning Redis even if the vector similarity is moderate (vector models compress everything into 384 dimensions — specific proper nouns can get lost).

---

## 6. Post-Session Reflection

`reflect()` is the post-session cleanup method. It runs three analysis passes then consolidation.

### Pass 1: Contradiction Scan

Takes the top 200 active, unpinned, non-invalidated episodes ordered by `importance × recency_score`. Computes pairwise cosine similarities. For any pair where:
- cosine similarity > 0.85 (semantically very similar)
- AND word-level Jaccard overlap < 40% (but the words are different)

...it queues a review item of type `'contradiction'` with priority equal to the similarity score.

The pair that triggered this has near-identical meaning expressed with different words — which is a strong signal that one supersedes the other, but the system doesn't know which is newer/correct without human judgment.

### Pass 2: Importance Recalibration

Finds all episodes with `access_count >= 5`. For each:
```
suggested_importance = current_importance + (access_count // 5) × 0.05
```
Capped at 0.95. Only applies changes where the delta is > 0.04 (to avoid noise).

Applied immediately (not queued) because this is an objective signal: if something is retrieved repeatedly, it's undervalued.

### Pass 3: Ambiguity Detection

Flags episodes where `access_count > 3 AND importance < 0.2 AND invalidated_at IS NULL`. These are memories that are clearly being used but were stored with very low importance. Queued for review as `'ambiguous'` with priority 0.3.

### Review Queue

All flagged items go into the `memory_review_queue` table:

```sql
memory_review_queue (
    id, episode_id, review_type, reason, priority,
    created_at, resolved_at, resolution
)
```

`get_pending_reviews()` returns items where `resolved_at IS NULL`, ordered by priority descending.

### Resolving Reviews

Five resolution actions:

| Action | What happens |
|--------|-------------|
| `update` | Re-embed with new content/importance |
| `delete` | `invalidate_episode()` — sets `invalidated_at = now` |
| `pin` | `set_pinned(episode_id, True)` |
| `keep_both` | No structural change — marks resolved |
| `defer` | No structural change — marks resolved (review again next session) |

All actions call `resolve_review(review_id, action)` which sets `resolved_at` and `resolution`.

---

## 7. OpenClaw Integration

### The Core Problem It Solves

OpenClaw (an AI agent framework) has built-in memory that uses markdown files — `USER.md`, `MEMORY.md`, `IDENTITY.md`. These are manually maintained, size-limited, and do not support semantic search. The ARN plugin replaces this entirely with live semantic memory.

### Integration Architecture

```
OpenClaw                                ARN Server (port 7900)
──────────────────────────────────────────────────────────────
user sends message
  │
  ├──[message_received hook]──────────► POST /perceive
  │                                     {content, role: "user", session_id}
  │
  ├──[before_prompt_build hook]─────►  POST /recall
  │   priority 40 (runs before LLM)    {query, top_k: 8}
  │   ◄────────── recall results ──────
  │   format as markdown, append
  │   to system prompt
  │
  │   LLM sees:
  │   [original system prompt]
  │   ## Recalled Memories
  │   - [2 days ago] User prefers dark mode
  │   - [1 week ago] Main project is ARN
  │
  ├──[llm_output hook]─────────────►  POST /perceive
  │                                    {content, role: "assistant"}
  │
  ├──[before_tool_call]────────────►  POST /perceive
  │                                    {content, role: "tool_call"}
  ├──[after_tool_call]─────────────►  POST /perceive
  │                                    {content, role: "tool_result"}
  │
session ends
  └──[session_end hook]────────────►  POST /session/end
                                       → triggers reflect()
```

### Why `before_prompt_build` and Not a Tool?

Tools require the agent to decide when to search memory. This creates a chicken-and-egg problem: the agent needs memory to know when it needs memory. By injecting via `before_prompt_build`, every single LLM call automatically gets relevant context prepended to the system prompt — the agent never has to ask for it.

Priority 40 puts ARN before most other hooks. The memories appear naturally in the system prompt, the same way a human recalls relevant context without consciously "doing a memory lookup."

### Fire-and-Forget Hooks

The perceive hooks (message_received, llm_output, before_tool_call, after_tool_call, before_compaction) are fire-and-forget — they don't await the HTTP response. This means:
- Storage never blocks the LLM call
- Latency for the user is unaffected
- If the ARN server is temporarily unavailable, the hook fails silently (memory capture drops, but conversation continues)

`before_prompt_build` is awaited because it must return memories before the LLM call proceeds.

---

## 8. Schema v7 — Sessions + Role-Aware Episodes

### New `sessions` Table

```sql
CREATE TABLE sessions (
    id           TEXT PRIMARY KEY,    -- caller-defined (e.g. "sess-20260602-abc")
    started_at   REAL NOT NULL,        -- unix timestamp
    ended_at     REAL,                 -- NULL until session.end() called
    reason_start TEXT,                 -- optional label ("user opened chat")
    reason_end   TEXT,                 -- optional label ("user closed chat")
    episode_count INTEGER DEFAULT 0,  -- updated at end_session()
    metadata     TEXT DEFAULT '{}'    -- JSON for arbitrary data
);
```

Sessions are designed to be idempotent on create — calling `create_session(id, ...)` twice with the same id returns the existing record rather than erroring. This handles the case where a plugin reconnects mid-session.

### New Columns on Episodes

```sql
ALTER TABLE episodes ADD COLUMN role       TEXT DEFAULT 'user';
ALTER TABLE episodes ADD COLUMN metadata   TEXT DEFAULT '{}';
ALTER TABLE episodes ADD COLUMN session_id TEXT;
```

`role` allows post-hoc filtering by message type. If you want to search only through things the user said (not tool outputs), that's a single SQL condition: `WHERE role = 'user'`.

`metadata` is JSON — arbitrary key-value attached to an episode at store time. The TypeScript plugin uses it to store the OpenClaw session ID, tool name for tool calls, etc.

`session_id` links episodes to their session. `get_session_episodes(session_id)` returns all episodes stored during that session, in order.

### Migration Strategy

Each schema version has its own migration function:

```python
def _migrate_schema(self, conn, from_version):
    if from_version < 5:
        self._migrate_v4_to_v5(conn)   # vec0 tables, FTS5 index
    if from_version < 6:
        self._migrate_v5_to_v6(conn)   # bi-temporal, entities, review queue
    if from_version < 7:
        self._migrate_v6_to_v7(conn)   # sessions table, role/metadata/session_id
```

Each migration uses `ALTER TABLE ... ADD COLUMN` with `try/except` for idempotency — if the column already exists, `ALTER TABLE` raises an error and the `except` passes silently. This means migrations can be re-run without corruption.

---

## 9. Plugin API + Daemon Mode

### Endpoint Design Decisions

The plugin API lives on port 7900 (separate from the main REST API on port 8742) and has no `agent_id` parameter on any endpoint — it always uses `DEFAULT_AGENT_ID = os.environ.get("ARN_AGENT_ID", "default")`. This simplifies the TypeScript plugin: it makes simple POST requests without needing to track agent identity.

The main REST API on 8742 is for direct programmatic use where multi-agent isolation matters. The plugin API on 7900 is for single-agent setups like OpenClaw where there's one agent per ARN instance.

### `_age_label()` Helper

Recall results include a human-readable age label. The formula:

```python
def _age_label(created_at: float) -> str:
    delta = time.time() - created_at
    if delta < 90:           return "just now"
    if delta < 3600:         return f"{int(delta/60)} minutes ago"
    if delta < 86400:        return f"{int(delta/3600)} hours ago"
    if delta < 86400 * 14:   return f"{int(delta/86400)} days ago"
    return f"{int(delta/86400/7)} weeks ago"
```

This appears in recall results as a contextual hint — "I remembered this 3 days ago vs. 6 months ago" affects how much trust to place in it.

### Daemon Mode

The server supports background operation via `os.fork()`:

```python
def _start_daemon(host, port):
    _ARN_DIR.mkdir(exist_ok=True)
    pid = os.fork()
    if pid > 0:
        # Parent: write PID file and exit
        _PID_FILE.write_text(str(pid))
        return
    # Child: redirect stdout/stderr to log file, run server
    with open(_LOG_FILE, 'a') as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    uvicorn.run(app, host=host, port=port)
```

PID file at `~/.arn/arn.pid`. Log file at `~/.arn/arn.log`. Stopping the daemon:

```python
def _stop_daemon():
    pid = int(_PID_FILE.read_text())
    os.kill(pid, signal.SIGTERM)
    _PID_FILE.unlink()
```

Status check combines PID validity with a health endpoint call to report live stats:

```bash
arn status
# → ARN daemon running (PID 1234) — 847 episodes, 12 sessions, 2.4MB
```

---

## 10. TypeScript Plugin

### `ArnClient` Class

Typed HTTP client that handles all communication with the ARN server:

```typescript
class ArnClient {
    constructor(private baseUrl: string) {}

    async post<T>(path: string, body: object): Promise<T> {
        const res = await fetch(`${this.baseUrl}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return res.json();
    }

    // fire-and-forget: doesn't await, errors silently
    perceive(content: string, role: string, sessionId?: string): void {
        this.post('/perceive', { content, role, session_id: sessionId }).catch(() => {});
    }

    async recall(query: string, topK = 8): Promise<RecallResult[]> {
        const result = await this.post<{results: RecallResult[]}>('/recall', { query, top_k: topK });
        return result.results ?? [];
    }
    // ... sessionStart, sessionEnd, pin, forget, recentSessions, pendingReviews, resolveReview
}
```

### `formatMemories()` Function

Converts recall results to a markdown string that gets appended to the system prompt:

```typescript
function formatMemories(results: RecallResult[]): string {
    if (!results.length) return '';
    const lines = results.map(r => `- [${r.age_label}] ${r.content}`);
    return `\n\n## Recalled Memories\n${lines.join('\n')}\n`;
}
```

Simple and readable. The agent sees this as part of its system context, not as a structured object. The `age_label` field (e.g. "3 days ago") gives temporal grounding.

### Hook Priority System

OpenClaw hooks have integer priorities — higher priority hooks run first. The plugin registers `before_prompt_build` at priority 40. The default memory hook (being disabled) typically runs at priority 50. This ensures ARN's memories are injected before any other context injection, and before the LLM call.

### Agent Tools

The 5 tools provide explicit memory control beyond the automatic injection:

```typescript
// arn_recall: targeted search
{ name: "arn_recall", parameters: { query: string, top_k?: number, role_filter?: string } }

// arn_pin: permanent facts
{ name: "arn_pin", parameters: { episode_id: number } }

// arn_forget: remove stale info
{ name: "arn_forget", parameters: { episode_id: number } }

// arn_sessions: list past sessions
{ name: "arn_sessions", parameters: { limit?: number } }

// arn_review: check flagged contradictions
{ name: "arn_review", parameters: {} }
```

These are registered in `openclaw.plugin.json` via the `contracts.tools` array. The plugin manifest also defines a config schema so OpenClaw validates config values before the plugin loads.

### SKILL.md

The SKILL.md file is copied to the OpenClaw workspace by `arn connect`. It tells the agent how to behave with ARN in plain English:

```markdown
You have persistent memory that works automatically.

Rules:
- Never mention that memories are being injected into your context
- Trust what the current user says over recalled memories if they conflict
- Do NOT write to MEMORY.md, USER.md, or IDENTITY.md — ARN handles this
- Do NOT call memory_search or memory_get — those are the old system
```

This matters because without guidance, agents often try to use both the old and new memory systems simultaneously, or announce "I'm now searching my memory..." on every message.

---

## 11. CLI Extensions

### `arn connect`

Performs the full OpenClaw wiring in one command:

1. Locates the plugin source at `integrations/openclaw/` relative to the ARN repo
2. Copies it to `~/.openclaw/plugins/arn-memory/`
3. Runs `npm install` in the plugin directory
4. Calls `openclaw plugins register arn-memory`
5. Calls `openclaw memory disable` (disables the built-in markdown memory)
6. Copies `SKILL.md` to `~/.openclaw/workspace/skills/`
7. Starts the ARN daemon on port 7900 if not already running

Each step is logged so the user can see what's happening and where it fails if something goes wrong.

### `arn disconnect`

Reverses everything:

1. `openclaw plugins unregister arn-memory`
2. `openclaw memory enable` (restores built-in memory)
3. Stops the daemon
4. Removes the plugin files from `~/.openclaw/plugins/arn-memory/`

ARN data is preserved — nothing in `~/.arn_data/` is touched.

### `arn status`

Calls `_daemon_status(port)` which:
1. Reads the PID file, checks the process is alive
2. Makes a GET request to `/v1/health`
3. Formats the response: episode count, session count, DB size

---

## 12. Integration Tests

`tests/test_openclaw_integration.py` tests the full pipeline from Python, bypassing HTTP entirely. It uses shared module-scoped fixtures (one `ARNv9` instance, one `storage` reference) to avoid the overhead of creating a new database per test class.

### Why No HTTP in Integration Tests?

The FastAPI server uses module-level singletons (`pool`, `DATA_ROOT`, `DEFAULT_AGENT_ID`) set at import time. Resetting them after import is unreliable. The storage and cognitive layers implement the full feature set — the HTTP endpoints are thin wrappers. Testing at the layer below HTTP gives better coverage of the actual logic with less setup complexity.

### Degraded Mode Handling

The test environment blocks outbound network access — `all-MiniLM-L6-v2` is not cached and cannot be downloaded. `ARNv9(use_embeddings=True)` falls back to the lexical hash embedding engine automatically.

The `TestThresholdValidation` class tests semantic accuracy (topic-appropriate recall), which requires real embeddings. It uses a module-level flag:

```python
def _has_real_embeddings() -> bool:
    try:
        e = EmbeddingEngine(use_model=True)
        return not e.is_degraded
    except Exception:
        return False

REAL_EMBEDDINGS = _has_real_embeddings()

@pytest.mark.skipif(not REAL_EMBEDDINGS, reason="requires real sentence-transformers model")
class TestThresholdValidation:
    ...
```

All other test classes (schema, sessions, role-aware perceive/recall, pin/forget, reflect, age labels) pass in degraded mode because they test structural correctness, not semantic quality.

### Test Coverage

| Class | Tests | Focus |
|-------|-------|-------|
| `TestSchemaV7` | 3 | `sessions` table exists, `role`/`metadata`/`session_id` on episodes, `SCHEMA_VERSION == 7` |
| `TestSessionManagement` | 7 | create, get, idempotent create, end (with episode_count update), recent list, count, get_episodes |
| `TestRoleAwarePerceive` | 7 | store with each role (user, assistant, tool_call, tool_result), metadata, session_id roundtrip |
| `TestRoleAwareRecall` | 4 | basic recall, SQL role filter, SQL session filter |
| `TestPinAndForget` | 5 | pin, pinned state in DB, unpin, invalidate, invalidated excluded from recall |
| `TestReflectAndReview` | 4 | `reflect()` returns stats dict, pending reviews accessible, enqueue+resolve cycle, session end updates count |
| `TestAgeLabel` | 5 | just now (< 90s), minutes, hours, days, weeks |
| `TestThresholdValidation` | 3 | topic isolation (cooking vs programming), cross-topic non-contamination |

---

## 13. README Overhaul

The README was rewritten to:

1. **Remove all personal name references** — "Mohamed (MrKali)", "Hi, I'm...", "My name is...", "I built...", "— Mohamed", "Mohamed prefers Python" in examples
2. **Add ASCII diagrams:**
   - Architecture diagram (agent → ARN core → SQLite tables)
   - Retrieval pipeline (all steps with formula)
   - Memory type hierarchy (working memory, long-term, bi-temporal, supersedes chain)
   - Reflect workflow (3 passes + consolidation rules)
3. **Fix stale info:**
   - Schema version 6 → 7
   - `contrib/openclaw/` → `integrations/openclaw/`
   - Port 7900 for plugin API throughout
   - All new CLI subcommands listed
4. **Improve how-to sections:**
   - Step-by-step explanations for each operation (store, recall, pin, reflect, review)
   - Session lifecycle flow
   - OpenClaw hook event flow
5. **Add role values table** for the plugin API
6. **Third-person voice** throughout (no first-person "I/my/me")

---

## 14. Bug Fixes

### `AttributeError: 'EmbeddingEngine' object has no attribute '_config'`

**File:** `arn_v9/core/embeddings.py`

**Root cause:** Phase 0 removed the tier system (which stored `self._config` and `self._tier`) but left two references to those attributes in `_load_model()`:

```python
# Success path — crashed on successful model load:
logger.info(f"Loaded {self._tier} model — dim={self._config['dim']}, ~{self._config['approx_ram_mb']}MB RAM")

# Error path — crashed in the except block (obscuring the original error):
logger.critical(f"Failed to load embedding model '{self._config['name']}': {e}. ...")
```

The error path was particularly nasty: when a model fails to load (e.g. network blocked in CI), Python executes the `except` block, hits `self._config`, and raises a new `AttributeError`. The original exception is lost and the `except` block propagates the wrong error.

**Fix:**
```python
logger.info(f"Loaded {MODEL_NAME} (dim={_EMBEDDING_DIM})")
# ...
logger.critical(f"Failed to load embedding model '{MODEL_NAME}': {e}. ...")
```

**Impact:** This was causing `ARNv9(use_embeddings=True)` to raise `AttributeError` instead of gracefully falling back to degraded mode. The integration test fixture (`arn` fixture in `test_openclaw_integration.py`) was failing before any test ran.

---

## 15. Commit History

All work was done on branch `claude/arn-v9-overhaul-LDe1Y`.

```
9f0597c  Rewrite README — diagrams, anonymous voice, v0.10.0 accuracy
95c2675  Add OpenClaw integration — sessions, role-aware perceive/recall, TS plugin, daemon
8ddf7bb  Phase 4: Update README for v0.10.0 — hybrid retrieval, new CLI, no tiers/columns
02a4e6b  Phase 4: Cleanup — new CLI subcommands, Phase 1-3 tests, remove stale references
24c33c6  Phase 3: Post-session reflection — reflect(), review queue, reconciliation
0f05e9a  Phase 2: Temporal intelligence — bi-temporal, supersedes, pins, self-editing, entities
[...]    Phase 1: Storage + retrieval overhaul — sqlite-vec, FTS5, RRF, recency, MMR, threshold
[...]    Phase 0: Simplify — remove columns, tiers, temporal tags, word-overlap contradiction
```

---

## Summary Table

| System | File(s) | What it does |
|--------|---------|-------------|
| Vector KNN | `storage/persistence.py` | Approximate nearest-neighbor via sqlite-vec `vec0` table |
| FTS5 index | `storage/persistence.py` | BM25 full-text search with Porter stemming, auto-maintained via triggers |
| Entity extraction | `core/entities.py` | Proper nouns, code identifiers, paths, URLs, number+unit combos |
| RRF fusion | `core/retrieval.py` | Combine 3 ranked lists without weight tuning |
| MMR reranking | `core/retrieval.py` | Eliminate near-duplicate results for diversity |
| Score-gap cutoff | `core/retrieval.py` | Adaptive threshold based on score distribution shape |
| Bi-temporal facts | `storage/persistence.py` | `valid_from`/`valid_until` per episode; historical recall opt-in |
| Supersedes chains | `storage/persistence.py` | Link old→new facts; old marked with `valid_until = NOW` |
| Pinning | `storage/persistence.py` | Pin = bypass decay + consolidation + supersession |
| Reflection | `core/reflect.py` | Post-session scan: contradictions, importance recalibration, ambiguity |
| Review queue | `storage/persistence.py` | Queue flagged episodes; 5 resolution actions |
| Sessions table | `storage/persistence.py` | Session lifecycle with episode_count and timestamps |
| Role-aware episodes | `storage/persistence.py` | `role`/`metadata`/`session_id` columns on every episode |
| Plugin API | `api/server.py` | 11 endpoints on port 7900, no agent_id, for OpenClaw plugin |
| Daemon mode | `api/server.py` | `os.fork()` background process with PID file + log |
| TS plugin | `integrations/openclaw/index.ts` | ArnClient + 8 hooks + 5 agent tools |
| CLI connect/disconnect | `scripts/arn_cli.py` | Wire/unwire OpenClaw in one command |
| Integration tests | `tests/test_openclaw_integration.py` | 38 tests covering schema v7, sessions, roles, reflect, age labels |
