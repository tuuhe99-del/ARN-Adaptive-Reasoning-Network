# Research Report 6 -- API Security

**Agent:** API Security Auditor
**Area:** Authentication and API exposure risks
**Risk Level:** MEDIUM
**Effort to Fix:** 4-8 hours

---

## Executive Summary

The API is currently bound to 0.0.0.0:8742, meaning it accepts connections from any network interface. Authentication is optional (only enforced if ARN_API_KEY env var is set). Agent ID validation uses a regex pattern but filesystem isolation provides actual separation. The rate limiter is per-agent and can be bypassed by creating new agent IDs. No CORS restrictions are configured.

---

## Vulnerability Assessment

| Vulnerability | Severity | Exploitability | Evidence |
|---------------|----------|----------------|----------|
| Binds to 0.0.0.0 by default | High | Trivial | server.py:642 |
| API key comparison not timing-safe | Medium | Low | server.py:355 |
| No request body size limit | Medium | Easy | No max_length on content |
| Rate limiter bypassable | Medium | Easy | Create new agent_id |
| No CORS restrictions | Low | Trivial | No CORSMiddleware |
| Agent ID regex allows traversal | Low | Hard | Pattern allows dots but fs isolates |

---

## Detailed Findings

### 1. Network Binding (server.py:642)

```python
uvicorn.run(app, host="0.0.0.0", port=8742)
```

The server binds to `0.0.0.0` (all interfaces), not `127.0.0.1` (localhost only). This means any device on the network can reach the API. The comments in server.py mention `0.0.0.0` as the deployment option, but this is the hardcoded default.

### 2. Timing-Safe Key Comparison (server.py:352-356)

```python
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY is not None:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

The comparison `x_api_key != API_KEY` is a standard Python string comparison, which is NOT timing-safe. An attacker could use timing analysis to guess the API key character by character.

### 3. Rate Limiter Bypass (server.py:274-289)

```python
class RateLimiter:
    def check(self, agent_id: str) -> bool:
        now = time.time()
        window = self._windows[agent_id]
        self._windows[agent_id] = [t for t in window if now - t < 60]
        if len(self._windows[agent_id]) >= self._rpm:
            return False
        self._windows[agent_id].append(now)
        return True
```

The rate limiter tracks requests per `agent_id`. An attacker can bypass the limit by sending requests with different `agent_id` values. There is no global rate limit.

Additionally, `_windows` grows unbounded — old agent IDs are never purged, causing a slow memory leak.

### 4. Agent ID Validation (server.py:208-234)

```python
class AgentPool:
    def get(self, agent_id: str) -> ARNPlugin:
        if agent_id not in self._plugins:
            # ...
            self._plugins[agent_id] = ARNPlugin(agent_id=agent_id, ...)
```

The `agent_id` is validated by FastAPI's request model using a regex pattern `^[a-zA-Z0-9_\-]+$`. This prevents path traversal characters (`/`, `..`, etc.). Each agent gets its own subdirectory, so filesystem isolation provides defense in depth.

---

## Security Hardening Plan

### For Local-Only Deployment (Current)
1. Bind to `127.0.0.1` instead of `0.0.0.0`
2. Set `ARN_API_KEY` environment variable
3. Firewall port 8742 from external access

### For Network-Exposed Deployment
1. **Bind to specific interface** or use a reverse proxy
2. **Require API key** — set `ARN_API_KEY` and validate all requests
3. **Timing-safe comparison:** Use `hmac.compare_digest()`
4. **Add global rate limit** alongside per-agent limit
5. **Add request size limit:** `max_length=50000` on content fields
6. **Add CORS middleware:** Restrict to known origins
7. **Add request logging:** Audit all store/delete operations
8. **Use HTTPS:** TLS termination at reverse proxy

### Code Changes
```python
# Timing-safe key comparison
import hmac

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY is not None:
        if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Global rate limit + per-agent
class RateLimiter:
    def __init__(self, rpm: int = 300, global_rpm: int = 1000):
        self._rpm = rpm
        self._global_rpm = global_rpm
        self._windows = defaultdict(list)
        self._global_window = []
    
    def check(self, agent_id: str) -> bool:
        now = time.time()
        # Global check
        self._global_window = [t for t in self._global_window if now - t < 60]
        if len(self._global_window) >= self._global_rpm:
            return False
        # Per-agent check
        window = self._windows[agent_id]
        self._windows[agent_id] = [t for t in window if now - t < 60]
        if len(self._windows[agent_id]) >= self._rpm:
            return False
        self._windows[agent_id].append(now)
        self._global_window.append(now)
        return True
```

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Unauthorized network access | High | Medium | Bind to 127.0.0.1 or use firewall |
| API key brute force | Medium | Low | Timing-safe comparison |
| DoS via many agent IDs | Medium | Medium | Global rate limit |
| DoS via large payloads | Medium | Medium | Request size limit |
| CORS abuse | Low | Low | CORS middleware |
