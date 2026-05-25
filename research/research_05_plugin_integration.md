# Research Report 5 -- Plugin Integration Testing

**Agent:** Plugin Integration Test Engineer
**Area:** OpenClaw plugin hooks (most critical gap)
**Risk Level:** CRITICAL
**Effort to Fix:** 12-20 hours

---

## Executive Summary

The 7 OpenClaw hooks are completely untested by automation. The plugin uses a weak 32-bit hash for deduplication (collision probability ~1 in 2^32). All API calls are fire-and-forget with silent error swallowing. If the ARN API is unreachable, the plugin continues without memory — no retries, no circuit breaker, no alerting. This is the gap most likely to hide a silent failure in production.

---

## Hook-by-Hook Analysis

| Hook | Trigger | API Calls | Risk |
|------|---------|-----------|------|
| session_start | New session | 3x POST /v1/memory/recall | Medium — persona preload fails silently |
| message_received | User message | POST /v1/memory/store | Low — fire-and-forget |
| message_sent | Agent reply | POST /v1/memory/store | Low — fire-and-forget |
| before_tool_call | Tool invoked | POST /v1/memory/store | Low — fire-and-forget |
| after_tool_call | Tool returns | POST /v1/memory/store | Low — fire-and-forget |
| before_prompt_build | Prompt assembled | POST /v1/memory/store + POST /v1/memory/recall | **High** — fallback store + context injection |
| before_compaction | Turn ends | POST /v1/memory/store | Low — fire-and-forget |

### Critical Path: before_prompt_build

This hook has TWO responsibilities:
1. **Fallback auto-store:** If message_received/message_sent missed messages, it stores them from the prompt build event
2. **Auto-inject:** Queries ARN and injects relevant memories into the prompt

If this hook fails:
- Messages may be lost (fallback store fails)
- Agent sees NO context (injection fails)
- OpenClaw has no way to know ARN is down

---

## Deduplication Audit

```javascript
function hashContent(text) {
  const payload = String(text || "");
  let h = 0;
  for (let i = 0; i < payload.length; i++) {
    h = ((h << 5) - h + payload.charCodeAt(i)) | 0;
  }
  return String(h);
}
```

- **Algorithm:** DJB2-like 32-bit hash
- **Collision probability:** ~1 in 2^32 for random inputs (~1 in 4 billion)
- **Risk:** Very low for normal use, but possible with adversarial input
- **Session scope:** Deduplication is per-session, not global

**Recommendation:** Acceptable for now, but consider SHA-256 for critical deployments.

---

## Error Handling Audit

Every API call follows this pattern:
```javascript
try {
  await arnStore(agentId, content, source, memoryType, context, importance, config);
} catch (e) {
  console.warn(`[ARN] message_received store failed: ${e.message}`);
}
```

**Problems:**
1. Errors are logged to console but not surfaced to OpenClaw's error handling
2. No retry logic — transient network failures are permanent data loss
3. No circuit breaker — a down ARN API will be hammered on every message
4. No offline buffering — messages during outage are lost forever

---

## Test Harness Design

### Mock ARN Server (Node.js)
```javascript
// mock-arn-server.js
const mockArn = {
  stores: [],
  memories: [],
  store(body) {
    this.stores.push(body);
    return { stored: true, episode_id: this.stores.length };
  },
  recall(body) {
    return { results: this.memories.filter(m => m.content.includes(body.query)) };
  }
};
```

### Jest Test Suite
```javascript
// plugin.test.js
import plugin from './index.js';

describe('message_received', () => {
  test('stores user message with source=user', async () => {
    const mockApi = createMockApi();
    await plugin.register(mockApi);
    await mockApi.emit('message_received', { content: 'Hello' }, { agentId: 'test' });
    expect(mockArn.stores[0].source).toBe('user');
  });
  
  test('deduplicates identical messages', async () => {
    await mockApi.emit('message_received', { content: 'Hello' }, { agentId: 'test' });
    await mockApi.emit('message_received', { content: 'Hello' }, { agentId: 'test' });
    expect(mockArn.stores.length).toBe(1);
  });
});

describe('before_prompt_build', () => {
  test('injects relevant memories', async () => {
    mockArn.memories = [{ content: 'User likes Python', source: 'user', similarity: 0.9 }];
    const result = await mockApi.emit('before_prompt_build', { prompt: 'What language?' }, { agentId: 'test' });
    expect(result.prependContext).toContain('User likes Python');
  });
  
  test('fallback stores missed messages', async () => {
    await mockApi.emit('before_prompt_build', {
      messages: [{ role: 'user', content: 'Missed message' }]
    }, { agentId: 'test' });
    expect(mockArn.stores.some(s => s.content === 'Missed message')).toBe(true);
  });
});
```

---

## Recommended Fixes

### Fix 1: Add Retry Logic (High Priority)
```javascript
async function arnFetchWithRetry(path, method, body, config, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      return await arnFetch(path, method, body, config);
    } catch (e) {
      if (i === retries - 1) throw e;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
}
```

### Fix 2: Add Circuit Breaker (Medium Priority)
Track failure count per endpoint. After N consecutive failures, skip ARN calls for a cooldown period.

### Fix 3: Plugin Test Suite (High Priority)
Implement the Jest test harness above. Run in CI on every push.

### Fix 4: Error Propagation (Medium Priority)
Optionally propagate ARN errors to OpenClaw so the user knows memory is unavailable.
