# ARN Task: Fix ARN Plugin api.config Mismatch (CRITICAL — ARN Not Functional)

## Task ID
`ARN-redteam-gateway-arn-401`

## Priority
**CRITICAL** — ARN memory is NOT used by the agent at all. All agent memory
currently comes from markdown files only (USER.md, MEMORY.md, memory/*.md).
Until this is fixed, ARN provides zero value to live agent sessions.

## Review Chain
```
codex → claude → kimi
```

## Root Cause — CONFIRMED

The `arn-memory` plugin's `register(api)` function receives `api.config`
containing the **entire root openclaw.json** instead of just the plugin's
config block. This is proven by this gateway log entry:

```
[ARN] register: api.config = {"meta":{"lastTouchedVersion":"2026.4.26",
"lastTouchedAt":"2026-05-19T12:08:11.845Z"},"wizard":{"lastRunAt":...}}
```

The plugin code at line 364 of `/Users/hustle/arn-v9-repo/openclaw-arn-plugin/index.js`:
```js
apiKey: api.config.apiKey || "",
```

Because `api.config` is the root config object, `api.config.apiKey` is
`undefined` → defaults to `""` → line 43's guard `if (config.apiKey)` is
`false` → the `X-API-Key` header is never sent → **401 on every ARN call**.

The plugin correctly logs `resolved apiKey present=true` in sessions where
`api.config` is properly scoped to the plugin config block — but the broken
sessions (which appear to be embedded-fallback CLI runs) get the full root
config instead.

## What Was Observed

Live test battery with blank MD files (USER.md, MEMORY.md all zeroed out):
- Agent said: "I don't know your real name" — could not recall Alex
- Agent said: "No stored note about your colleague's name" — could not recall Jordan
- Gateway log confirmed: `[ARN] recall failed: 401` on every `before_prompt_build`
- Direct ARN API call with the same key **works** — the server and key are fine
- The earlier "passing" tests (T1–T8) were ALL reading from MD files, not ARN

## What To Fix

### 1. Find where api.config is scoped incorrectly (primary fix)

The plugin receives the wrong `api.config` when running in **embedded fallback
mode** (the CLI falls back to running the agent locally when the gateway WS
times out). In embedded mode, openclaw may pass the root config object to
`api.config` instead of the plugin-specific config block.

Investigate:
- `openclaw agent` CLI embedded fallback path in the openclaw source
- How `api.config` is constructed for externally-loaded plugins vs built-in plugins
- Whether the fix is in the plugin (read from `api.config.plugins?.entries?.["arn-memory"]?.config?.apiKey`)
  or in the gateway/runner (fix the scoping before passing to `register()`)

### 2. Plugin-side defensive fix (apply regardless)

In `/Users/hustle/arn-v9-repo/openclaw-arn-plugin/index.js`, update the
`register(api)` function to resolve the apiKey defensively:

```js
register(api) {
  // api.config may be the full root config in embedded mode — extract plugin config if so
  const pluginCfg = (api.config?.plugins?.entries?.["arn-memory"]?.config) || api.config || {};
  const config = {
    arnEndpoint: pluginCfg.arnEndpoint || DEFAULT_ARN_ENDPOINT,
    apiKey: pluginCfg.apiKey || "",
    topK: pluginCfg.topK || DEFAULT_TOP_K,
    minScore: pluginCfg.minScore || DEFAULT_MIN_SCORE,
    tokenBudget: pluginCfg.tokenBudget || DEFAULT_TOKEN_BUDGET,
    storeMessages: pluginCfg.storeMessages !== false,
    storeTools: pluginCfg.storeTools !== false,
    storeCompaction: pluginCfg.storeCompaction !== false,
  };
  // Log what we resolved so failures are diagnosable
  api.logger?.info?.(`[ARN] register: resolved apiKey present=${!!config.apiKey} endpoint=${config.arnEndpoint}`);
```

### 3. Fix embedded fallback config scoping (secondary fix)

Find the openclaw embedded runner that passes `api.config` to plugins and
ensure it passes `pluginEntry.config` (the specific plugin config block) rather
than the root config. Look in:
- `~/.nvm/versions/node/v25.2.1/lib/node_modules/openclaw/dist/`
- Search for `register(` or `plugin.register` near config passing

### 4. Install sqlite-vec for proper vector indexing

The ARN server currently runs without `sqlite-vec`, degrading vector recall
(falls back to numpy memmap). Gateway logs confirm:
`chunks_vec not updated — sqlite-vec unavailable`

```bash
# In the ARN virtual environment (or system Python if no venv)
cd /Users/hustle/arn-v9-repo
pip install sqlite-vec
# Then restart the ARN server
```

Verify after install:
```bash
python3 -c "import sqlite_vec; print('sqlite-vec ok:', sqlite_vec.__version__)"
```

### 5. Upgrade embedding tier from nano → base

The machine has 8GB RAM. ARN is running `nano` tier (all-MiniLM-L6-v2,
384-dim, MTEB 56.3) which is the Pi-optimised low-RAM tier. The `base` tier
(all-mpnet-base-v2, 768-dim) is far better for recall quality on this hardware.

Set `ARN_EMBEDDING_TIER=base` in the ARN server startup:

Option A — environment variable before starting:
```bash
export ARN_EMBEDDING_TIER=base
```

Option B — add to any `.env` or startup script that launches the ARN server.

After changing the tier, restart the ARN server. **Note**: existing embeddings
were computed with the nano model. After upgrading, re-store key facts so they
are indexed with base-tier vectors for consistent recall.

## Files To Change
- `/Users/hustle/arn-v9-repo/openclaw-arn-plugin/index.js` — defensive fix (safe to change)
- `/Users/hustle/arn-v9-repo/arn_v9/core/embeddings.py` — verify `DEFAULT_TIER` env var wiring
- Possibly openclaw dist (investigate only, may not be safe to patch)

## Agent Instructions

- Read `COLLAB.md` and `docs/collab-protocol.md` first.
- Apply the defensive plugin fix first (it's safe and fixes the symptom).
- Then investigate the embedded runner to understand the source.
- Run `python3 -m py_compile` on any Python file touched.
- Do NOT change ARN server code or API keys.

## Verification
```bash
# Zero out MDs
> ~/.openclaw/workspace/USER.md
> ~/.openclaw/workspace/MEMORY.md

# Run identity test — should recall Alex from ARN, not say "I don't know"
openclaw agent --agent main --session-id arn-fix-verify \
  --message "Who am I? What do you know about me?" \
  --json 2>&1 | grep -i "alex\|don't know\|no information"

# Restore MDs after test
cp /tmp/md-backup-main/USER.md ~/.openclaw/workspace/USER.md
cp /tmp/md-backup-main/MEMORY.md ~/.openclaw/workspace/MEMORY.md

# Confirm no 401s in gateway log during the test
grep "ARN.*401" /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log | tail -5
# Expected: 0 lines from the test session
```
