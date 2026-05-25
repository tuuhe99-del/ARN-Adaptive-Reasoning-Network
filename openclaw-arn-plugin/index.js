/**
 * OpenClaw ARN Memory Plugin — Auto-Store + Auto-Inject with Proper Attribution
 * ==============================================================================
 *
 * This plugin turns ARN into OpenClaw's primary memory system.
 * It replaces markdown memory files with a live, brain-inspired semantic memory
 * that learns from every interaction — and labels everything so the agent knows
 * who said what.
 *
 * Attribution / Labeling:
 *   - User messages        → source="user", memory_type="episode"
 *   - Agent replies        → source="agent", memory_type="episode"
 *   - Tool calls           → source="tool:{name}", memory_type="procedure"
 *   - Tool results         → source="tool_result", memory_type="episode"
 *   - Turn summaries       → source="compaction", memory_type="episode"
 *
 * Injected context reads like real conversation history:
 *   "User asked about Python dict.get()..."
 *   "I explained that .get() returns None instead of raising KeyError..."
 *   "I called code_search and found 3 relevant examples..."
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const DEFAULT_ARN_ENDPOINT = "http://localhost:8742";
const DEFAULT_TOP_K = 5;
const DEFAULT_MIN_SCORE = 0.35;
const DEFAULT_TOKEN_BUDGET = 1500;

// In-memory per-session caches
const sessionPersonaCache = new Map();   // sessionId -> static persona string
const sessionStoredMsgs = new Map();     // sessionId -> Set of content hashes
const sessionTurnCount = new Map();      // sessionId -> turn number
const sessionLastUserMsg = new Map();    // sessionId -> last user message text (for extraction)
const sessionInjectedIds = new Map();    // sessionId -> Set of episode IDs already injected
const sessionAnchorText = new Map();     // sessionId -> first user message text (topic anchor)
const sessionTopicShifted = new Map();   // sessionId -> Set of shifted-to topic summaries
const agentNameMap = new Map();          // sessionKey -> agent name (from ctx)

// ---------------------------------------------------------------------------
// Session state persistence (survives gateway restart)
// sessionInjectedIds is persisted; sessionAnchorText is intentionally NOT
// persisted — topic anchors should reset on gateway restart.
// ---------------------------------------------------------------------------

const SESSION_STATE_PATH = require('path').join(
  process.env.HOME || '/tmp',
  '.arn_data', 'session_state.json'
);

function saveSessionState() {
  try {
    const fs = require('fs');
    const dir = require('path').dirname(SESSION_STATE_PATH);
    fs.mkdirSync(dir, { recursive: true });
    const obj = {};
    for (const [sid, ids] of sessionInjectedIds) {
      obj[sid] = { injected: [...ids] };
    }
    fs.writeFileSync(SESSION_STATE_PATH, JSON.stringify(obj));
  } catch (_) {}
}

function loadSessionState() {
  try {
    const raw = require('fs').readFileSync(SESSION_STATE_PATH, 'utf8');
    const obj = JSON.parse(raw);
    for (const [sid, data] of Object.entries(obj)) {
      sessionInjectedIds.set(sid, new Set(data.injected || []));
    }
  } catch (_) {}
}

// Load persisted session state once at module load
loadSessionState();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function arnFetch(path, method, body, config) {
  const url = `${config.arnEndpoint}${path}`;
  const headers = { "Content-Type": "application/json" };
  if (config.apiKey) headers["X-API-Key"] = config.apiKey;
  const res = await fetch(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`ARN ${path} ${res.status}: ${text}`);
  }
  return res.json();
}

/**
 * Flatten message content to a plain string.
 * Handles both string content and OpenAI-style array content blocks:
 *   [{ type: "text", text: "..." }, { type: "image_url", ... }]
 */
function flattenContent(content) {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter(block => block && block.type === "text" && block.text)
      .map(block => block.text)
      .join("\n");
  }
  return String(content);
}

/**
 * Strip the injected ARN memory block from a user message before extraction.
 * OpenClaw injects memory context as a prefix block into the user message content.
 * Including this in extraction causes already-known facts to be re-extracted and
 * stored as duplicates.
 */
function stripArnInjection(text) {
  const markers = [
    "## ARN Memory",
    "## Relevant Memories",
    "<arn_memory_context>",
    "OpenClaw runtime context",
    "A new session was started",
    "Conversation info (untrusted metadata)",
  ];
  for (const marker of markers) {
    const idx = text.indexOf(marker);
    if (idx !== -1) {
      // Take everything AFTER the injected block.
      // The actual user message follows a blank line after the block.
      const afterBlock = text.slice(idx);
      const endOfBlock = afterBlock.search(/\n\n(?!#|-|\s*$)/);
      if (endOfBlock !== -1) {
        const remainder = text.slice(idx + endOfBlock + 2).trim();
        if (remainder.length > 10) return remainder;
      }
      // If we can't find a clean split, take what was before the marker
      return text.slice(0, idx).trim() || text.slice(idx + marker.length).trim();
    }
  }
  return text;
}

function hashContent(text) {
  const payload = String(text || "");
  let h = 0;
  for (let i = 0; i < payload.length; i++) {
    h = ((h << 5) - h + payload.charCodeAt(i)) | 0;
  }
  return String(h);
}

function getAgentId(ctx) {
  // Extract the stable agent name from OpenClaw context.
  // OpenClaw session keys follow the pattern "agent:<name>:<mode>:<sessionId>"
  // e.g. "agent:main:explicit:arn-battery-t1" → we want "main".
  // Using the full sanitized session key creates a new namespace per session
  // and breaks cross-session memory recall — so we MUST extract just the name.

  const raw = ctx.agentId || ctx.sessionKey || ctx.sessionId || "default";

  // If the raw value follows OpenClaw's "agent:<name>:..." pattern, extract <name>
  const parts = raw.split(":");
  if (parts.length >= 2 && parts[0] === "agent" && parts[1]) {
    return parts[1].replace(/[^a-zA-Z0-9_\-]/g, "_");
  }

  // Fallback: sanitize the full raw value (handles non-standard formats)
  return raw.replace(/:/g, "_").replace(/[^a-zA-Z0-9_\-]/g, "_");
}

function getSessionId(ctx) {
  return ctx.sessionId || ctx.sessionKey || ctx.runId || "default";
}

function formatTimeAgo(ageHours, createdAt) {
  if (ageHours === undefined || ageHours === null) return "";
  const mins = Math.round(ageHours * 60);
  const hrs = Math.round(ageHours);
  const days = Math.round(ageHours / 24);

  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (hrs < 24) return `${hrs}h ago`;
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;

  // Fallback to date string
  try {
    const d = new Date((createdAt || Date.now() / 1000) * 1000);
    const now = new Date();
    const sameYear = d.getFullYear() === now.getFullYear();
    const opts = sameYear
      ? { month: "short", day: "numeric" }
      : { month: "short", day: "numeric", year: "numeric" };
    return d.toLocaleDateString("en-US", opts);
  } catch {
    return `${days}d ago`;
  }
}

function isRecallNoise(r) {
  // Filter out low-signal noise episodes. Keep factual content even if noisy-looking.
  const text = String(r.content || "");

  // Skip OpenClaw runtime context injections
  if (text.startsWith("OpenClaw runtime context")) return true;

  // Skip stored tool calls, tool results, and after_tool_call raw logs
  if (r.source && (r.source.startsWith("tool:") || r.source === "after_tool_call" || r.source === "tool_result")) return true;

  // Skip LLM self-identification / persona-choosing outputs (e.g., "I chose Ash")
  if (r.source === "llm_output" && /\bname\b.*?\bAsh\b|\bAsh\b.*?\bname\b|chose.*?name|calling myself/i.test(text)) return true;

  // For timestamped user messages "[Mon YYYY-MM-DD ...]": skip ONLY if they look like
  // meta-questions (short, end with "?", contain identity-query terms) not factual statements.
  const tsMatch = /^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{4}-\d{2}-\d{2}[^\]]*\]\s*(.+)/s.exec(text);
  if (tsMatch) {
    const body = tsMatch[1].trim();
    // Skip short timestamped questions; keep compact factual notes.
    if (body.length < 60 && (body.endsWith("?") || /^(who|what|when|where|why|how|do|does|did|can|could|should|would|is|are|am)\b/i.test(body))) return true;
    // Skip identity meta-questions like "Who am I?", "What do you know about me?"
    if (/^who am i|^what do you know|^do you remember me|^what.s my name/i.test(body)) return true;
    // Keep everything else (factual statements, instructions, addresses, code, etc.)
    return false;
  }

  // Skip very short low-value messages
  if (text.length < 12) return true;

  return false;
}

function formatArnMemories(results, maxChars = 1500) {
  if (!results || results.length === 0) return "";
  // The header instructs the agent to answer from this context directly rather
  // than falling back to tool searches like memory_search that return empty
  // results when the built-in memory store is empty.
  const header = "## ARN Memory (your verified knowledge base — answer from this directly; do not run memory_search for these topics)";
  const lines = [header];
  let chars = header.length;

  for (const r of results) {
    // Skip low-signal noise entries (timestamped question echoes, etc.)
    if (isRecallNoise(r)) continue;
    const tier = r.confidence_tier || "medium";
    const icon = tier === "high" ? "📌" : tier === "medium" ? "💭" : "❓";
    const when = formatTimeAgo(r.age_hours, r.created_at);
    const timeTag = when ? `[${when}]` : "";

    // Build attribution prefix based on source
    let prefix = "";
    const source = r.source || "unknown";
    if (source === "user") {
      prefix = "User said:";
    } else if (source === "me" || source === "agent") {
      prefix = "I said:";
    } else if (source.startsWith("tool:")) {
      const toolName = source.slice(5);
      prefix = `I used ${toolName}:`;
    } else if (source === "tool_result") {
      prefix = "Tool returned:";
    } else if (source === "compaction") {
      prefix = "Turn summary:";
    } else {
      prefix = `[${source}]:`;
    }

    const line = `${icon} ${timeTag} ${prefix} ${r.content}`.trim().replace(/\s+/g, " ");
    if (chars + line.length + 1 > maxChars) break;
    lines.push(line);
    chars += line.length + 1;
  }

  return lines.length > 1 ? lines.join("\n") : "";
}

async function arnStore(agentId, content, source, memoryType, context, importance, config) {
  const body = {
    agent_id: agentId,
    content: String(content || ""),
    importance: importance ?? 0.5,
    source,
    memory_type: memoryType,
    context: context || {},
  };
  return arnFetch("/v1/memory/store", "POST", body, config);
}

async function arnRecall(agentId, query, config, opts = {}) {
  const body = {
    agent_id: agentId,
    query,
    top_k: opts.topK ?? config.topK,
    memory_type: opts.memoryType ?? undefined,
  };
  const data = await arnFetch("/v1/memory/recall", "POST", body, config);
  const minScore = opts.minScore ?? config.minScore;
  return (data.results || []).filter(r => {
    // Use r.score (the server-combined score: similarity + importance + recency)
    // NOT calibrated_confidence alone — cal_conf can be very low for expected/routine
    // facts (identity, preferences) even when they are the correct recall target.
    // r.score is already a balanced combination and is the right filter signal.
    const effectiveScore = r.score ?? r.similarity ?? 0;
    return effectiveScore >= minScore;
  });
}

// ---------------------------------------------------------------------------
// Per-hook store functions
// ---------------------------------------------------------------------------

async function storeUserMessage(agentId, event, config) {
  const content = event.content || event.body || "";
  if (shouldSkipContent(content)) return;
  const senderName = event.senderName || event.from || "user";
  await arnStore(
    agentId,
    content,
    "user",
    "episode",
    { sender: senderName, timestamp: event.timestamp },
    deriveImportance(content),
    config
  );
}

function shouldSkipContent(content) {
  const text = String(content || "");
  if (text.length < 15) return true;
  if (text.startsWith("⚠️")) return true;
  if (text.includes("Something went wrong")) return true;
  if (text.includes("billing error")) return true;
  if (text.includes("not authorized")) return true;
  // Skip injected memory-context blocks to prevent circular storage noise
  if (text.includes("<arn_memory_context>")) return true;
  if (text.includes("</arn_memory_context>")) return true;
  if (text.startsWith("## Relevant Memories (conversation history)")) return true;
  if (text.startsWith("## ARN Memory")) return true;
  // Skip OpenClaw runtime context injections (source may be unknown/user in messages)
  if (text.startsWith("OpenClaw runtime context")) return true;
  // Skip session-start lifecycle notifications
  if (text.startsWith("A new session was started")) return true;
  // Skip Telegram conversation metadata blobs
  if (text.startsWith("Conversation info (untrusted metadata)")) return true;
  if (text.includes('"chat_id":"telegram:') || text.includes('"chat_id": "telegram:')) return true;
  // Skip heartbeat/lifecycle system messages
  if (text.startsWith("Read HEARTBEAT.md")) return true;
  if (text.startsWith("[OpenClaw heartbeat")) return true;
  if (/^\[OpenClaw\s/i.test(text)) return true;
  // Skip workspace markdown boilerplate (HEARTBEAT.md and similar files)
  if (text.includes("Keep this file empty") || text.includes("Add tasks") || text.includes("skip heartbeat API")) return true;
  if (text.startsWith("```markdown") && text.includes("Keep this")) return true;
  // Skip timestamped ping noise (e.g. "[Sat 2026-05-23 13:54 EDT] ping")
  if (/^\[(?:Sat|Sun|Mon|Tue|Wed|Thu|Fri) /.test(text) && text.length < 200 && /\bping\b/i.test(text)) return true;
  // Skip short questions (< 120 chars, ends with "?", or starts with a WH-word / modal).
  // These are queries the user sent — storing them poisons retrieval because their
  // semantic vector matches similar queries better than the actual factual answers.
  if (text.length < 120 && /[?]\s*$/.test(text)) return true;
  if (text.length < 80 && /^(who|what|when|where|why|how|do|does|did|can|could|should|would|is|are|am)\b/i.test(text)) return true;
  return false;
}

function deriveImportance(content, memoryType) {
  // Identity and procedure facts are permanently high-importance regardless of length.
  // This ensures they rank above question echoes and other noise in recall scoring.
  if (memoryType === "identity") return 0.9;
  if (memoryType === "procedure") return 0.85;
  const text = String(content || "");
  return Math.min(0.9, 0.4 + text.length / 800);
}

async function storeAgentReply(agentId, event, config) {
  const content = event.content || "";
  if (shouldSkipContent(content)) return;
  await arnStore(
    agentId,
    content,
    "agent",
    "episode",
    { success: event.success, timestamp: Date.now() },
    deriveImportance(content),
    config
  );
}

async function storeToolCall(agentId, event, config) {
  const { toolName, params } = event;
  const content = `Tool call: ${toolName}\nParams: ${JSON.stringify(params || {}, null, 2)}`;
  if (shouldSkipContent(content)) return;
  await arnStore(
    agentId,
    content,
    `tool:${toolName}`,
    "procedure",
    { tool_name: toolName, tool_call_id: event.toolCallId },
    deriveImportance(content, "procedure"),
    config
  );
}

async function storeToolResult(agentId, event, config) {
  const { toolName, result, error } = event;
  let content = "";
  if (error) {
    content = `Tool ${toolName} failed: ${error}`;
  } else {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result);
    content = `Tool ${toolName} result: ${resultStr.slice(0, 2000)}`;
  }
  if (shouldSkipContent(content)) return;
  await arnStore(
    agentId,
    content,
    "tool_result",
    "episode",
    { tool_name: toolName, tool_call_id: event.toolCallId, duration_ms: event.durationMs },
    deriveImportance(content),
    config
  );
}

// ---------------------------------------------------------------------------
// Persona & dynamic memory
// ---------------------------------------------------------------------------

async function loadStaticPersona(agentId, config) {
  try {
    const [identity, prefs, procs] = await Promise.all([
      arnRecall(agentId, "agent identity persona who am I", config, { memoryType: "identity" }),
      arnRecall(agentId, "user preferences settings likes dislikes", config, { memoryType: "preference" }),
      arnRecall(agentId, "procedures workflow steps how to", config, { memoryType: "procedure" }),
    ]);

    const parts = [];
    if (identity.length) {
      parts.push("### Identity\n" + identity.map(r => `- ${r.content}`).join("\n"));
    }
    if (prefs.length) {
      parts.push("### Preferences\n" + prefs.map(r => `- ${r.content}`).join("\n"));
    }
    if (procs.length) {
      parts.push("### Procedures\n" + procs.map(r => `- ${r.content}`).join("\n"));
    }

    return parts.join("\n\n");
  } catch (e) {
    console.warn(`[ARN] persona load failed: ${e.message}`);
    return "";
  }
}

async function queryDynamicMemory(agentId, prompt, config) {
  try {
    const results = await arnRecall(agentId, prompt, config);
    return formatArnMemories(results, config.tokenBudget);
  } catch (e) {
    console.warn(`[ARN] recall failed: ${e.message}`);
    return "";
  }
}

async function storeTurnSummary(agentId, messages, turnNumber, config) {
  const userMsgs = messages.filter(m => m.role === "user").map(m => m.content);
  const agentMsgs = messages.filter(m => m.role === "assistant" && !m.tool_calls).map(m => m.content);
  const toolCalls = messages.filter(m => m.role === "assistant" && m.tool_calls).length;

  if (userMsgs.length === 0 && agentMsgs.length === 0) return;

  const summaryParts = [];
  if (userMsgs.length) summaryParts.push(`User: ${userMsgs[userMsgs.length - 1].slice(0, 200)}`);
  if (agentMsgs.length) summaryParts.push(`Agent: ${agentMsgs[agentMsgs.length - 1].slice(0, 200)}`);
  if (toolCalls) summaryParts.push(`(used ${toolCalls} tool call(s))`);

  const content = `Turn ${turnNumber}: ${summaryParts.join(" | ")}`;
  if (shouldSkipContent(content)) return;

  try {
    await arnStore(agentId, content, "compaction", "episode", {
      turn_number: turnNumber,
      message_count: messages.length,
    }, deriveImportance(content), config);
  } catch (e) {
    console.warn(`[ARN] turn summary store failed: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Intelligent fact extraction
// ---------------------------------------------------------------------------

/**
 * Returns true if the message looks like pure task output (code-heavy reply
 * or a very long reply that's mostly code), or if the user message is just
 * a command/question with nothing factual to extract.
 */
function isTaskOutput(userMessage, agentReply) {
  const userText = String(userMessage || "");
  const replyText = String(agentReply || "");

  // Skip if agent reply is too short or too long and mostly code
  if (replyText.length < 20) return true;

  // Count characters inside code fences
  const codeFenceMatches = replyText.match(/```[\s\S]*?```/g) || [];
  const codeChars = codeFenceMatches.reduce((sum, block) => sum + block.length, 0);
  const codeFraction = replyText.length > 0 ? codeChars / replyText.length : 0;

  // If > 60% of reply is code blocks, this is task output
  if (codeFraction > 0.6) return true;

  // If reply > 1000 chars and mostly code
  if (replyText.length > 1000 && codeFraction > 0.45) return true;

  // If both user message AND reply are short, nothing to extract
  const uTrimmed = userText.trim();
  if (uTrimmed.length < 15 && replyText.length < 40) return true;

  return false;
}

/**
 * Heuristic extraction — mirrors the Python patterns in /v1/memory/extract.
 * Used as fallback when the server endpoint is unavailable.
 */
function heuristicExtract(userMessage, agentReply) {
  const combined = `${userMessage}\n${agentReply}`;
  const facts = [];

  function addFact(content, memoryType, importance) {
    const c = String(content || "").trim();
    if (!c || c.length < 8) return;
    if (facts.some(f => f.content.toLowerCase() === c.toLowerCase())) return;
    facts.push({ content: c, memory_type: memoryType, importance });
  }

  // Identity: name introductions
  for (const m of combined.matchAll(/\bmy name is ([A-Z][a-zA-Z\-']{1,30})\b/gi)) {
    addFact(`User's name is ${m[1]}`, "identity", 0.9);
  }

  // "I'm <Name>" / "I am <Name>" followed by boundary
  for (const m of combined.matchAll(/\bI(?:'m| am) ([A-Z][a-zA-Z\-']{1,30})(?:\s*[,.]|\s+and\b|\s+from\b|$)/gm)) {
    const skip = new Set(["The","A","An","Not","So","Just","Here","There","Going","Working","Building","Using","Also","Now"]);
    if (!skip.has(m[1])) addFact(`User's name is ${m[1]}`, "identity", 0.9);
  }

  // Role / project
  for (const m of combined.matchAll(/\bI(?:'m| am) working (?:on|at) ([^.,\n]{5,80})/gi)) {
    addFact(`User is working on ${m[1].trim()}`, "identity", 0.85);
  }
  for (const m of combined.matchAll(/\bI(?:'m| am) building ([^.,\n]{5,80})/gi)) {
    addFact(`User is building ${m[1].trim()}`, "identity", 0.85);
  }
  for (const m of combined.matchAll(/\bmy project is ([^.,\n]{5,80})/gi)) {
    addFact(`User's project is ${m[1].trim()}`, "identity", 0.85);
  }
  for (const m of combined.matchAll(/\bmy team ([^.,\n]{5,80})/gi)) {
    addFact(`User's team ${m[1].trim()}`, "identity", 0.85);
  }

  // Team members
  for (const m of combined.matchAll(/\b([A-Z][a-zA-Z]{1,20}) handles ([^.,\n]{3,60})/g)) {
    addFact(`${m[1]} handles ${m[2].trim()}`, "identity", 0.85);
  }
  for (const m of combined.matchAll(/\b([A-Z][a-zA-Z]{1,20}) is (?:our|my) ([^.,\n]{3,60})/g)) {
    addFact(`${m[1]} is user's ${m[2].trim()}`, "identity", 0.85);
  }

  // "My co-founder/partner/CTO is NAME" and reverse
  for (const m of combined.matchAll(
    /\bmy (co-founder|cto|ceo|coo|vp|partner|manager|boss|lead|collaborator)(?:\s+on\s+[^,.\n]{3,40})?\s+is\s+([A-Z][a-zA-Z]{1,20})\b/gi
  )) {
    addFact(`${m[2]} is User's ${m[1].toLowerCase()}`, "identity", 0.85);
  }
  for (const m of combined.matchAll(
    /\b([A-Z][a-zA-Z]{1,20}) is my (co-founder|cto|ceo|coo|vp|partner|manager|boss|lead|collaborator)\b/g
  )) {
    addFact(`${m[1]} is User's ${m[2].toLowerCase()}`, "identity", 0.85);
  }

  // Preferences
  for (const m of combined.matchAll(/\bI (?:prefer|like|love|hate|dislike|enjoy) ([^.,\n]{3,80})/gi)) {
    addFact(`User prefers/likes: ${m[1].trim()}`, "preference", 0.85);
  }
  for (const m of combined.matchAll(/\bI use ([^.,\n]{3,60})/gi)) {
    addFact(`User uses ${m[1].trim()}`, "preference", 0.85);
  }
  for (const m of combined.matchAll(/\bI work (?:with|in|using) ([^.,\n]{3,60})/gi)) {
    addFact(`User works with/in ${m[1].trim()}`, "preference", 0.85);
  }

  // Decisions / procedures
  for (const m of combined.matchAll(/\bwe decided to ([^.,\n]{5,120})/gi)) {
    addFact(`Decision: we decided to ${m[1].trim()}`, "procedure", 0.85);
  }
  for (const m of combined.matchAll(/\bgoing forward[,\s]+(?:we(?:'ll|'re| will| are)|I(?:'ll|'m| will| am))?\s*([^.,\n]{5,120})/gi)) {
    addFact(`Procedure: going forward ${m[1].trim()}`, "procedure", 0.85);
  }
  for (const m of combined.matchAll(/\bfrom now on[,\s]+([^.,\n]{5,120})/gi)) {
    addFact(`Procedure: from now on ${m[1].trim()}`, "procedure", 0.85);
  }

  // Server/config facts
  for (const m of combined.matchAll(/\b(?:my|our|the) (?:server|api|service|backend|endpoint|database|db)\b[^.,\n]{0,40}?\b(?:runs? on|is at|listens? on|on port|at port|port)\b\s*([^.,\n]{3,60})/gi)) {
    addFact(`Server/API config: ${m[0].trim()}`, "procedure", 0.85);
  }
  for (const m of combined.matchAll(/\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})\b/g)) {
    const start = Math.max(0, m.index - 60);
    const snippet = combined.slice(start, m.index + m[0].length + 60).trim();
    addFact(`Server endpoint mentioned: ${snippet}`, "procedure", 0.75);
  }

  return facts;
}

/**
 * Dedup check: recall the fact text and see if a very similar fact already exists.
 * Returns true if a near-duplicate already exists in ARN.
 * Uses exact/near-exact string matching as a fast path before falling back to
 * similarity threshold (0.88) to catch more near-duplicates than the old 0.92 threshold.
 */
async function factAlreadyKnown(agentId, factContent, config) {
  // Fast path: check for near-exact string match via recall
  const normalized = factContent.toLowerCase().trim();

  try {
    const results = await arnRecall(agentId, factContent, config, { topK: 3, minScore: 0.1 });
    for (const r of results) {
      // Exact or near-exact string match
      if (r.content.toLowerCase().trim() === normalized) return true;
      // Check if one contains the other (substring match for paraphrases)
      if (normalized.length > 20 && r.content.toLowerCase().includes(normalized.slice(0, 40))) return true;
      if (normalized.length > 20 && normalized.includes(r.content.toLowerCase().slice(0, 40))) return true;
      // Similarity threshold
      if ((r.score ?? r.calibrated_confidence ?? 0) > 0.88) return true;
    }
  } catch (e) {
    // Non-critical — if recall fails, allow storing
  }
  return false;
}

/**
 * Main entry point: extract memory-worthy facts from a user/agent exchange
 * and store them into ARN with appropriate memory_type and importance.
 *
 * Tries the server-side /v1/memory/extract endpoint first; falls back to
 * heuristic extraction in JS if the endpoint returns 404.
 */
async function extractAndStoreFacts(agentId, userMessage, agentReply, config) {
  const userText = String(userMessage || "");
  const replyText = String(agentReply || "");

  // Skip if this exchange is pure task output
  if (isTaskOutput(userText, replyText)) return;

  let facts = [];
  let usedServerExtract = false;

  // Try the server extract endpoint first
  try {
    const data = await arnFetch("/v1/memory/extract", "POST", {
      agent_id: agentId,
      user_message: userText,
      agent_reply: replyText,
    }, config);
    if (Array.isArray(data.facts)) {
      facts = data.facts;
      usedServerExtract = true;
    }
  } catch (e) {
    // 404 → endpoint doesn't exist yet; fall through to heuristic
    // Any other error → also fall through
    if (!e.message.includes("404")) {
      console.warn(`[ARN] extract endpoint error (falling back to heuristic): ${e.message}`);
    }
  }

  // Fallback: client-side heuristic extraction
  if (!usedServerExtract) {
    facts = heuristicExtract(userText, replyText);
  }

  if (!facts || facts.length === 0) return;

  // Store each extracted fact after dedup check
  for (const fact of facts) {
    const content = String(fact.content || "").trim();
    if (!content) continue;

    try {
      const alreadyKnown = await factAlreadyKnown(agentId, content, config);
      if (alreadyKnown) {
        console.log(`[ARN] fact dedup skip (similarity>0.88 or string match): ${content.slice(0, 60)}`);
        continue;
      }

      await arnStore(
        agentId,
        content,
        "extracted_fact",
        fact.memory_type || "episode",
        { extraction_source: usedServerExtract ? "server" : "heuristic" },
        fact.importance ?? 0.6,
        config
      );
      console.log(`[ARN] stored extracted fact [${fact.memory_type}]: ${content.slice(0, 60)}`);
    } catch (e) {
      console.warn(`[ARN] failed to store extracted fact: ${e.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Deduplication helper
// ---------------------------------------------------------------------------

function wasStored(sessionId, content) {
  const key = hashContent(content);
  const set = sessionStoredMsgs.get(sessionId);
  if (set && set.has(key)) return true;
  return false;
}

function markStored(sessionId, content) {
  const key = hashContent(content);
  let set = sessionStoredMsgs.get(sessionId);
  if (!set) {
    set = new Set();
    sessionStoredMsgs.set(sessionId, set);
  }
  set.add(key);
}

// ---------------------------------------------------------------------------
// Plugin Entry
// ---------------------------------------------------------------------------

export default definePluginEntry({
  id: "arn-memory",
  name: "ARN Semantic Memory (Auto-Store + Auto-Inject)",
  description: "Replaces markdown memory with ARN's brain-inspired semantic memory. Every message and tool call is stored with proper attribution so the agent sees a coherent conversation history.",
  configSchema: {
    type: "object",
    properties: {
      arnEndpoint: { type: "string", default: DEFAULT_ARN_ENDPOINT },
      apiKey: { type: "string", default: "" },
      topK: { type: "integer", default: DEFAULT_TOP_K },
      minScore: { type: "number", default: DEFAULT_MIN_SCORE },
      tokenBudget: { type: "integer", default: DEFAULT_TOKEN_BUDGET },
      storeMessages: { type: "boolean", default: true },
      storeTools: { type: "boolean", default: true },
      storeCompaction: { type: "boolean", default: true },
      topicShiftThreshold: { type: "number", default: 0.45 },
      topicShiftMinLength: { type: "integer", default: 10 },
    },
  },

  register(api) {
    // api.config is the global gateway config when multiple plugins are loaded;
    // navigate to the plugin-specific block, falling back to direct api.config
    // for single-plugin gateways where the SDK shims it correctly.
    const raw = api.config?.plugins?.entries?.["arn-memory"]?.config
      ?? api.pluginConfig
      ?? api.config
      ?? {};
    const config = {
      arnEndpoint: raw.arnEndpoint || DEFAULT_ARN_ENDPOINT,
      apiKey: raw.apiKey || "",
      topK: raw.topK || DEFAULT_TOP_K,
      minScore: raw.minScore || DEFAULT_MIN_SCORE,
      tokenBudget: raw.tokenBudget || DEFAULT_TOKEN_BUDGET,
      storeMessages: raw.storeMessages !== false,
      // Default ON: tool calls are stored as procedure memories for context.
      // Set storeTools: false in config to disable.
      storeTools: raw.storeTools !== false,
      storeCompaction: raw.storeCompaction !== false,
      topicShiftThreshold: raw.topicShiftThreshold != null ? parseFloat(raw.topicShiftThreshold) : 0.45,
      topicShiftMinLength: raw.topicShiftMinLength != null ? parseInt(raw.topicShiftMinLength) : 10,
    };

    // Startup diagnostic log — critical for diagnosing 401s in embedded-fallback mode
    api.logger?.info?.(
      `[ARN] register: resolved apiKey present=${!!config.apiKey} endpoint=${config.arnEndpoint} ` +
      `mode=${api.config?.plugins?.entries?.["arn-memory"]?.config ? 'plugin-scoped' : api.pluginConfig ? 'pluginConfig' : 'raw-api.config'}`
    );

    // ================================================================
    // 1. SESSION START: preload static persona, record agent name
    // ================================================================
    api.on("session_start", async (event, ctx) => {
      const agentId = getAgentId(ctx);
      const sessionId = event.sessionId || getSessionId(ctx);

      // Remember agent name for this session if available
      if (ctx.agentId && !agentNameMap.has(ctx.sessionKey)) {
        agentNameMap.set(ctx.sessionKey, ctx.agentId);
      }

      try {
        const persona = await loadStaticPersona(agentId, config);
        if (persona) {
          sessionPersonaCache.set(sessionId, persona);
          sessionTurnCount.set(sessionId, 0);
          console.log(`[ARN] Persona loaded for ${agentId}`);
        }
      } catch (e) {
        console.warn(`[ARN] session_start: ${e.message}`);
      }

      // Dedicated low-threshold identity recall — user_message sources often have
      // very low calibrated_confidence (0.065-0.099) but are critical for identity.
      try {
        const identityHits = await arnRecall(
          agentId,
          "user identity name preferences timezone language",
          config,
          { minScore: 0.05 }
        );
        if (identityHits.length) {
          const existing = sessionPersonaCache.get(sessionId) || "";
          const injection = "### User Identity (recalled)\n" +
            identityHits.map(r => `- ${r.content}`).join("\n");
          sessionPersonaCache.set(
            sessionId,
            existing ? existing + "\n\n" + injection : injection
          );
          console.log(`[ARN] Identity recall: ${identityHits.length} hit(s) for ${agentId}`);
        }
      } catch (e) {
        console.warn(`[ARN] identity recall failed: ${e.message}`);
      }
    });

    // ================================================================
    // 2. MESSAGE RECEIVED: store user message
    // ================================================================
    api.on("message_received", async (event, ctx) => {
      if (!config.storeMessages) return;
      const agentId = getAgentId(ctx);
      const sessionId = getSessionId(ctx);
      const content = event.content || event.body || "";

      if (!content || wasStored(sessionId, content)) return;

      // Cache the user message so message_sent can retrieve it for fact extraction
      if (content && !shouldSkipContent(content)) {
        sessionLastUserMsg.set(sessionId, content);
      }

      try {
        await storeUserMessage(agentId, event, config);
        markStored(sessionId, content);
      } catch (e) {
        console.warn(`[ARN] message_received store failed: ${e.message}`);
      }
    });

    // ================================================================
    // 3. MESSAGE SENT: store agent reply + extract memorable facts
    // ================================================================
    api.on("message_sent", async (event, ctx) => {
      if (!config.storeMessages) return;
      const agentId = getAgentId(ctx);
      const sessionId = getSessionId(ctx);
      const content = event.content || "";

      console.log(`[ARN] message_sent fired: sessionId=${sessionId} contentLen=${content.length} cachedUserMsg=${!!sessionLastUserMsg.get(sessionId)}`);

      if (!content || wasStored(sessionId, content)) return;

      try {
        await storeAgentReply(agentId, event, config);
        markStored(sessionId, content);
      } catch (e) {
        console.warn(`[ARN] message_sent store failed: ${e.message}`);
      }

      // Extract and store memorable facts from this exchange.
      // openclaw's message_sent event does not carry the user message that
      // prompted the reply, so we retrieve it from the per-session cache
      // populated in message_received above.
      const userMsg =
        event.userMessage ||
        event.previousMessage ||
        event.prompt ||
        sessionLastUserMsg.get(sessionId) ||
        "";
      // Clear the cached user message after use so we don't re-extract on retries
      sessionLastUserMsg.delete(sessionId);
      console.log(`[ARN] extracting facts: userMsgLen=${userMsg.length} replyLen=${content.length}`);
      await extractAndStoreFacts(agentId, userMsg, content, config).catch(e =>
        console.warn(`[ARN] fact extraction failed: ${e.message}`)
      );
    });

    // ================================================================
    // 4. BEFORE TOOL CALL: store tool call as procedure
    // ================================================================
    api.on("before_tool_call", async (event, ctx) => {
      if (!config.storeTools) return;
      const agentId = getAgentId(ctx);
      const sessionId = getSessionId(ctx);
      const toolName = (event.toolName || "unknown").replace(/_/g, ' ');
      const rawArgs = event.params || {};

      // Extract the most meaningful argument (command, query, path, url, code, content)
      const KEY_ORDER = ['command','query','prompt','path','url','file_path','code','content','text','input'];
      const mainKey = KEY_ORDER.find(k => rawArgs[k] !== undefined && (typeof rawArgs[k] === 'string' || typeof rawArgs[k] === 'number'));
      const mainVal = mainKey
        ? String(rawArgs[mainKey]).slice(0, 120)
        : Object.values(rawArgs).filter(v => typeof v === 'string' || typeof v === 'number').map(String).join(', ').slice(0, 120);
      const toolContent = mainVal
        ? `Agent ran ${toolName}: ${mainVal}`
        : `Agent used ${toolName}`;

      if (wasStored(sessionId, toolContent + toolName)) return;

      try {
        await arnStore(agentId, toolContent, `tool:${toolName}`, "procedure", {
          tool_name: toolName, tool_call_id: event.toolCallId
        }, deriveImportance(toolContent, "procedure"), config);
        markStored(sessionId, toolContent + toolName);
      } catch (e) {
        console.warn(`[ARN] before_tool_call store failed: ${e.message}`);
      }
    });

    // ================================================================
    // 5. AFTER TOOL CALL: store tool result
    // ================================================================
    api.on("after_tool_call", async (event, ctx) => {
      if (!config.storeTools) return;
      const agentId = getAgentId(ctx);
      const sessionId = getSessionId(ctx);
      const resultStr = event.error
        ? `error:${event.error}`
        : JSON.stringify(event.result || {}).slice(0, 500);

      if (wasStored(sessionId, resultStr + event.toolName)) return;

      try {
        await storeToolResult(agentId, event, config);
        markStored(sessionId, resultStr + event.toolName);
      } catch (e) {
        console.warn(`[ARN] after_tool_call store failed: ${e.message}`);
      }
    });

    // ================================================================
    // 6. BEFORE PROMPT BUILD: fallback store + auto-inject
    // ================================================================
    api.on("before_prompt_build", async (event, ctx) => {
      const agentId = getAgentId(ctx);
      const sessionId = getSessionId(ctx);

      // --- FALLBACK AUTO-STORE ---
      // If message_received / message_sent hooks didn't fire (e.g. internal
      // turns), store messages from the prompt build event as fallback.
      if (config.storeMessages && Array.isArray(event.messages)) {
        for (const msg of event.messages) {
          let content = flattenContent(msg.content);
          // Cap tool-result content to 2000 chars to match storeToolResult's limit
          // and stay well within the server's 10 000-char StoreRequest.content max_length.
          if (msg.role === "tool" && content.length > 2000) {
            content = content.slice(0, 2000);
          }
          if (!content || wasStored(sessionId, content)) continue;

          let source = "unknown";
          let memoryType = "episode";
          let importance = 0.5;

          if (msg.role === "user") {
            source = "user";
          } else if (msg.role === "assistant") {
            if (msg.tool_calls && msg.tool_calls.length > 0) {
              if (!config.storeTools) continue; // Skip tool-call messages unless storeTools enabled
              const toolName = (msg.tool_calls[0]?.function?.name || "unknown").replace(/_/g, ' ');
              let rawArgs = {};
              try { rawArgs = JSON.parse(msg.tool_calls[0]?.function?.arguments || "{}"); } catch { rawArgs = {}; }

              // Extract the most meaningful argument (command, query, path, url, code, content)
              const KEY_ORDER = ['command','query','prompt','path','url','file_path','code','content','text','input'];
              const mainKey = KEY_ORDER.find(k => rawArgs[k] !== undefined && (typeof rawArgs[k] === 'string' || typeof rawArgs[k] === 'number'));
              const mainVal = mainKey
                ? String(rawArgs[mainKey]).slice(0, 120)
                : Object.values(rawArgs).filter(v => typeof v === 'string' || typeof v === 'number').map(String).join(', ').slice(0, 120);
              const toolContent = mainVal
                ? `Agent ran ${toolName}: ${mainVal}`
                : `Agent used ${toolName}`;
              source = `tool:${msg.tool_calls[0]?.function?.name || "unknown"}`;
              memoryType = "procedure";
              if (wasStored(sessionId, toolContent)) continue;
              if (shouldSkipContent(toolContent)) continue;
              try {
                await arnStore(agentId, toolContent, source, memoryType, {
                  role: msg.role,
                  name: msg.name || null,
                  tool_call_id: msg.tool_call_id || null,
                }, deriveImportance(toolContent, memoryType), config);
                markStored(sessionId, toolContent);
              } catch (e) {
                // Non-critical
              }
              continue;
            } else {
              source = "agent";
            }
          } else if (msg.role === "tool") {
            if (!config.storeTools) continue; // Skip tool-result messages unless storeTools enabled
            source = "tool_result";
          } else if (msg.role === "system") {
            continue; // Skip system messages to avoid feedback loops
          }

          if (shouldSkipContent(content)) continue;

          try {
            await arnStore(agentId, content, source, memoryType, {
              role: msg.role,
              name: msg.name || null,
              tool_call_id: msg.tool_call_id || null,
            }, deriveImportance(content, memoryType), config);
            markStored(sessionId, content);
          } catch (e) {
            // Non-critical
          }
        }
      }

      // Also store the current prompt if present
      if (config.storeMessages && event.prompt) {
        const prompt = event.prompt;
        if (!wasStored(sessionId, prompt) && !shouldSkipContent(prompt)) {
          try {
            await arnStore(agentId, prompt, "user", "episode", {}, deriveImportance(prompt), config);
            markStored(sessionId, prompt);
          } catch (e) {
            // Non-critical
          }
        }
      }

      // --- FACT EXTRACTION from previous exchange ---
      // message_sent does not fire in CLI/--json mode, so we extract facts here
      // instead by looking at the last user→assistant exchange in conversation history.
      // We use a hash of the exchange to avoid re-extracting on every call.
      api.logger?.info?.(`[ARN] before_prompt_build: msgs=${Array.isArray(event.messages) ? event.messages.length : 'N/A'} sessionId=${sessionId}`);
      if (config.storeMessages && Array.isArray(event.messages) && event.messages.length >= 2) {
        // Find the most recent assistant reply (non-tool-call)
        let lastAssistantIdx = -1;
        for (let i = event.messages.length - 1; i >= 0; i--) {
          const m = event.messages[i];
          if (m.role === "assistant" && !m.tool_calls && m.content) {
            lastAssistantIdx = i;
            break;
          }
        }
        if (lastAssistantIdx > 0) {
          // Find the user message immediately before it
          let lastUserMsg = "";
          for (let i = lastAssistantIdx - 1; i >= 0; i--) {
            if (event.messages[i].role === "user" && event.messages[i].content) {
              lastUserMsg = flattenContent(event.messages[i].content);
              lastUserMsg = stripArnInjection(lastUserMsg); // strip injected memory before extraction
              break;
            }
          }
          const lastAgentReply = flattenContent(event.messages[lastAssistantIdx].content);
          // Deduplicate: only extract once per unique exchange
          const exchangeKey = `extract:${hashContent(lastUserMsg + lastAgentReply)}`;
          if (!wasStored(sessionId, exchangeKey) && lastAgentReply.length >= 20) {
            markStored(sessionId, exchangeKey);
            api.logger?.info?.(`[ARN] extracting from exchange: userLen=${lastUserMsg.length} replyLen=${lastAgentReply.length}`);
            // Fire-and-forget so it doesn't block prompt delivery
            extractAndStoreFacts(agentId, lastUserMsg, lastAgentReply, config).catch(e =>
              api.logger?.warn?.(`[ARN] before_prompt_build fact extraction failed: ${e.message}`)
            );
          }
        }
      }

      // --- TOPIC SHIFT DETECTION ---
      // Compare the embedding of the first user message in the session (anchor)
      // against the current user message. If cosine similarity < topicShiftThreshold, the
      // conversation has pivoted topics — record the shift and reset injection.
      const topicShiftThreshold = parseFloat(config.topicShiftThreshold || process.env.ARN_TOPIC_SHIFT_THRESHOLD || '0.45');
      const topicShiftMinLength = parseInt(config.topicShiftMinLength || process.env.ARN_TOPIC_SHIFT_MIN_LENGTH || '10');
      const currentUserText = stripArnInjection(event.prompt || "");
      if (currentUserText.length >= topicShiftMinLength) {
        if (!sessionAnchorText.has(sessionId)) {
          // First message: set the anchor, nothing to compare yet
          sessionAnchorText.set(sessionId, currentUserText);
        } else {
          const anchorText = sessionAnchorText.get(sessionId);
          try {
            const simData = await arnFetch(
              "/v1/memory/embed_similarity",
              "POST",
              { text_a: anchorText, text_b: currentUserText },
              config
            );
            const similarity = simData?.similarity ?? 1.0;
            if (similarity < topicShiftThreshold) {
              const shiftSummary = `Topic shifted from: ${anchorText.slice(0, 80)} → ${currentUserText.slice(0, 80)}`;
              let shiftedSet = sessionTopicShifted.get(sessionId);
              if (!shiftedSet) {
                shiftedSet = new Set();
                sessionTopicShifted.set(sessionId, shiftedSet);
              }
              if (!shiftedSet.has(shiftSummary)) {
                shiftedSet.add(shiftSummary);
                // Fire-and-forget: store the topic shift as an episode in ARN memory
                arnFetch("/v1/memory/exchange", "POST", {
                  agent_id: agentId,
                  user_message: shiftSummary,
                  agent_response: "Topic shift detected and recorded.",
                  session_id: sessionId,
                  importance: 0.4,
                }, config).catch(() => {}); // non-critical, silent fail
                // Clear the injection registry so next recall sweeps fresh for new topic
                sessionInjectedIds.delete(sessionId);
              }
              // Update anchor to current topic so future shifts are relative to here
              sessionAnchorText.set(sessionId, currentUserText);
            }
          } catch (e) {
            // If embed_similarity call fails (server down, etc.) skip silently
          }
        }
      }

      // --- AUTO-INJECT ---
      const staticPersona = sessionPersonaCache.get(sessionId) || "";

      // Primary recall: raw user prompt
      let mainResults = [];
      try {
        mainResults = await arnRecall(agentId, event.prompt, config);
      } catch (e) {
        console.warn(`[ARN] main recall failed: ${e.message}`);
      }

      // Secondary always-on persona recall: short/vague queries like "Who am I?"
      // miss identity facts when used as the sole recall query.
      let personaResults = [];
      try {
        personaResults = await arnRecall(
          agentId,
          "user name preferences identity",
          config,
          { minScore: 0.05 }
        );
      } catch (e) {
        console.warn(`[ARN] persona recall failed: ${e.message}`);
      }

      // Tertiary procedure recall: boost procedural memories when the query
      // contains procedural keywords ("procedure", "steps", "how do I", etc.)
      let procedureResults = [];
      const promptText = event.prompt || "";
      const proceduralTerms = /\b(procedure|procedures|steps|how\s+do\s+i|how\s+to|workflow|process|guide|tutorial)\b/i;
      if (proceduralTerms.test(promptText)) {
        try {
          procedureResults = await arnRecall(
            agentId,
            promptText,
            config,
            { memoryType: "procedure", minScore: 0.05 }
          );
        } catch (e) {
          console.warn(`[ARN] procedure recall failed: ${e.message}`);
        }
      }

      // Merge and deduplicate by content hash
      const seen = new Set();
      const merged = [];
      for (const r of [...mainResults, ...personaResults, ...procedureResults]) {
        const key = r.content || "";
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(r);
      }
      // Sort by combined score descending (r.score is the server-combined signal)
      merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

      // Prevent unbounded growth of injection registry
      if (sessionInjectedIds.size > 200) {
        // Keep only the 100 most recently used sessions
        const keys = [...sessionInjectedIds.keys()];
        keys.slice(0, keys.length - 100).forEach(k => sessionInjectedIds.delete(k));
      }

      // Deduplicate against already-injected episodes this session
      const alreadyInjected = sessionInjectedIds.get(sessionId) || new Set();
      const newToInject = merged.filter(r => r.id && !alreadyInjected.has(r.id));

      // Cap: max 8 new memories per turn to protect context window
      const MAX_NEW_PER_TURN = 8;
      const toInject = newToInject.slice(0, MAX_NEW_PER_TURN);

      // Register the injected IDs
      if (toInject.length > 0) {
        const updated = new Set([...alreadyInjected, ...toInject.map(r => r.id)]);
        sessionInjectedIds.set(sessionId, updated);
        // Persist session state so injected IDs survive gateway restart (fire-and-forget)
        saveSessionState();
      }

      const dynamicMemory = formatArnMemories(toInject, config.tokenBudget);

      // Fire feedback only for episodes actually injected this turn (fire-and-forget, non-blocking)
      // Using toInject (not the full recall pools) avoids boosting memories the model never saw.
      const shownIds = toInject.map(r => r.id).filter(Boolean);
      if (shownIds.length > 0) {
        arnFetch("/v1/memory/feedback", "POST", { agent_id: agentId, episode_ids: shownIds }, config)
          .catch(() => {}); // non-critical, silent fail
      }

      const result = {};
      if (staticPersona) {
        result.prependSystemContext = staticPersona;
      }
      if (dynamicMemory) {
        result.prependContext = dynamicMemory;
      }

      if (Object.keys(result).length > 0) {
        return result;
      }
    });

    // ================================================================
    // 7. BEFORE COMPACTION: store turn summary
    // ================================================================
    api.on("before_compaction", async (event, ctx) => {
      if (!config.storeCompaction) return;
      const agentId = getAgentId(ctx);
      const sessionId = getSessionId(ctx);

      const turn = (sessionTurnCount.get(sessionId) || 0) + 1;
      sessionTurnCount.set(sessionId, turn);

      const messages = event.messages || [];
      await storeTurnSummary(agentId, messages, turn, config);
    });
  },
});
