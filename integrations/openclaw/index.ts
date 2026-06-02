/**
 * ARN Adaptive Memory — OpenClaw Plugin
 *
 * Auto-injects relevant memories before every LLM call via before_prompt_build.
 * Captures all messages, tool calls, and outputs. Manages session lifecycle.
 */

import { definePluginEntry } from "openclaw";
import { Type } from "@sinclair/typebox";

// ─── ARN API client ──────────────────────────────────────────────────────────

interface MemoryResult {
  id: number;
  content: string;
  score: number;
  role: string;
  importance: number;
  pinned: boolean;
  age_label: string;
  session_id: string | null;
}

class ArnClient {
  constructor(private baseUrl: string) {}

  async post(path: string, body: unknown): Promise<unknown> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok) return null;
      return res.json();
    } catch {
      return null;
    }
  }

  async get(path: string): Promise<unknown> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok) return null;
      return res.json();
    } catch {
      return null;
    }
  }

  perceive(
    content: string,
    role: string,
    importance: number,
    metadata: Record<string, unknown>,
    sessionId: string | null
  ): void {
    void this.post("/perceive", { content, role, importance, metadata, session_id: sessionId });
  }

  async recall(
    query: string,
    topK: number,
    roleFilter?: string[],
    sessionId?: string
  ): Promise<MemoryResult[]> {
    const body: Record<string, unknown> = { query, top_k: topK };
    if (roleFilter) body.role_filter = roleFilter;
    if (sessionId) body.session_id = sessionId;
    const res = (await this.post("/recall", body)) as { results?: MemoryResult[] } | null;
    return res?.results ?? [];
  }

  async sessionStart(id: string, reason: string): Promise<void> {
    await this.post("/session/start", { session_id: id, reason });
  }

  async sessionEnd(id: string, reason: string): Promise<void> {
    await this.post("/session/end", { session_id: id, reason });
  }

  pin(id: number): void {
    void this.post("/pin", { episode_id: id });
  }

  forget(id: number): void {
    void this.post("/forget", { episode_id: id });
  }

  async recentSessions(limit = 5): Promise<unknown> {
    return this.get(`/sessions/recent?limit=${limit}`);
  }

  async pendingReviews(max = 3): Promise<unknown> {
    return this.get(`/reviews/pending?max=${max}`);
  }

  async resolveReview(id: number, resolution: string, action: string): Promise<unknown> {
    return this.post("/reviews/resolve", { review_id: id, resolution, action });
  }
}

// ─── Memory formatting ───────────────────────────────────────────────────────

function formatMemories(results: MemoryResult[]): string {
  if (results.length === 0) return "";
  const lines = results.map((r) => {
    const pin = r.pinned ? " [pinned]" : "";
    return `- (${r.role}, ${r.age_label})${pin} ${r.content}`;
  });
  return [
    "## Recalled Memories",
    "The following are relevant memories from past interactions. Use them naturally — never tell the user you are reading from memory or mention this section.",
    ...lines,
  ].join("\n");
}

// ─── Plugin entry ────────────────────────────────────────────────────────────

export default definePluginEntry((pluginDef) => {
  let arn: ArnClient;
  let currentSessionId: string | null = null;
  let lastUserMessage: string | null = null;

  pluginDef.onInit((config) => {
    const url = (config.arnApiUrl as string | undefined) ?? "http://localhost:7900";
    arn = new ArnClient(url);
  });

  // ── Core hook: auto-inject memories before every LLM call ─────────────────
  pluginDef.hook("before_prompt_build", { priority: 40 }, async (event, config) => {
    const messages: Array<{ role: string; content: string }> = event.messages ?? [];
    const userMsg = [...messages].reverse().find((m) => m.role === "user");
    if (!userMsg) return {};

    const query = userMsg.content;
    if (query === lastUserMessage) return {}; // deduplicate
    lastUserMessage = query;

    const topK = (config.maxInjectedMemories as number | undefined) ?? 8;
    const results = await arn.recall(query, topK);
    if (results.length === 0) return {};

    return { appendSystemContext: formatMemories(results) };
  });

  // ── Capture user messages ─────────────────────────────────────────────────
  pluginDef.hook("message_received", {}, (event) => {
    const text: string = event.text ?? "";
    if (!text) return;
    arn.perceive(text, "user", 0.6, { channel: event.channelId }, currentSessionId);
  });

  // ── Capture assistant output ──────────────────────────────────────────────
  pluginDef.hook("llm_output", {}, (event, config) => {
    if (!(config.captureAssistant as boolean | undefined ?? true)) return;
    const text: string = event.text ?? "";
    if (!text) return;
    arn.perceive(
      text,
      "assistant",
      0.5,
      { model: event.model, usage: event.usage },
      currentSessionId
    );
  });

  // ── Capture tool calls ────────────────────────────────────────────────────
  pluginDef.hook("before_tool_call", {}, (event, config) => {
    if (!(config.captureToolCalls as boolean | undefined ?? true)) return;
    const name: string = event.toolName ?? "";
    const params = JSON.stringify(event.params ?? {}).slice(0, 500);
    arn.perceive(
      `Tool call: ${name}(${params})`,
      "tool_call",
      0.4,
      { toolName: name, params: event.params },
      currentSessionId
    );
  });

  // ── Capture tool results ──────────────────────────────────────────────────
  pluginDef.hook("after_tool_call", {}, (event, config) => {
    if (!(config.captureToolCalls as boolean | undefined ?? true)) return;
    const name: string = event.toolName ?? "";
    const result = String(event.result ?? "").slice(0, 2000);
    arn.perceive(
      `Tool result: ${name} → ${result}`,
      "tool_result",
      0.4,
      { toolName: name, success: event.success, durationMs: event.durationMs },
      currentSessionId
    );
  });

  // ── Session lifecycle ─────────────────────────────────────────────────────
  pluginDef.hook("session_start", {}, async (event) => {
    currentSessionId = (event.sessionKey as string | undefined) ?? crypto.randomUUID();
    await arn.sessionStart(currentSessionId, (event.reason as string | undefined) ?? "new");
  });

  pluginDef.hook("session_end", {}, async (event) => {
    if (currentSessionId) {
      await arn.sessionEnd(currentSessionId, (event.reason as string | undefined) ?? "unknown");
    }
    currentSessionId = null;
    lastUserMessage = null;
  });

  // ── Compaction marker ─────────────────────────────────────────────────────
  pluginDef.hook("before_compaction", {}, (event) => {
    arn.perceive(
      `Compaction: ${event.messageCount ?? "?"} messages, ${event.tokenCount ?? "?"} tokens`,
      "compaction_marker",
      0.3,
      { messageCount: event.messageCount, tokenCount: event.tokenCount },
      currentSessionId
    );
  });

  // ─── Agent tools ──────────────────────────────────────────────────────────

  pluginDef.tool(
    "arn_recall",
    "Search long-term memory for specific information. Memories auto-inject each turn — only use this for targeted searches by role or session.",
    Type.Object({
      query: Type.String({ description: "What to search for" }),
      top_k: Type.Optional(Type.Number({ description: "Max results", default: 5 })),
      role_filter: Type.Optional(
        Type.Array(Type.String(), { description: "Filter by role: user, assistant, tool_result, ..." })
      ),
    }),
    async (params) => {
      const results = await arn.recall(
        params.query,
        params.top_k ?? 5,
        params.role_filter as string[] | undefined
      );
      if (results.length === 0) return "No matching memories found.";
      return results
        .map((r) => `[${r.role}, ${r.age_label}${r.pinned ? ", pinned" : ""}] ${r.content}`)
        .join("\n");
    }
  );

  pluginDef.tool(
    "arn_pin",
    "Pin a memory so it persists across consolidation and is not decayed. Use for permanent facts: names, preferences, critical decisions.",
    Type.Object({
      episode_id: Type.Number({ description: "Episode ID to pin" }),
    }),
    (params) => {
      arn.pin(params.episode_id);
      return `Episode ${params.episode_id} pinned.`;
    }
  );

  pluginDef.tool(
    "arn_forget",
    "Soft-delete a memory that is wrong or outdated. The data is preserved in history but excluded from future recall.",
    Type.Object({
      episode_id: Type.Number({ description: "Episode ID to forget" }),
    }),
    (params) => {
      arn.forget(params.episode_id);
      return `Episode ${params.episode_id} removed from active memory.`;
    }
  );

  pluginDef.tool(
    "arn_sessions",
    "List recent sessions to understand past conversation history.",
    Type.Object({
      limit: Type.Optional(Type.Number({ description: "Number of sessions to show", default: 5 })),
    }),
    async (params) => {
      const res = (await arn.recentSessions(params.limit ?? 5)) as {
        sessions?: Array<{ id: string; started_at: number; ended_at?: number; episode_count: number }>;
      } | null;
      const sessions = res?.sessions ?? [];
      if (sessions.length === 0) return "No sessions found.";
      return sessions
        .map((s) => {
          const start = new Date(s.started_at * 1000).toISOString();
          const status = s.ended_at ? "ended" : "active";
          return `${s.id} — ${start} (${status}, ${s.episode_count} episodes)`;
        })
        .join("\n");
    }
  );

  pluginDef.tool(
    "arn_review",
    "Check for flagged memory issues: contradictions, ambiguous facts. Call at session start to stay aware of unresolved conflicts.",
    Type.Object({
      max: Type.Optional(Type.Number({ description: "Max items to show", default: 3 })),
    }),
    async (params) => {
      const res = (await arn.pendingReviews(params.max ?? 3)) as {
        items?: Array<{ id: number; review_type: string; reason: string; content: string }>;
      } | null;
      const items = res?.items ?? [];
      if (items.length === 0) return "No pending memory issues.";
      return items
        .map((i) => `[${i.review_type}] ID ${i.id}: ${i.content.slice(0, 120)}\n  Reason: ${i.reason}`)
        .join("\n\n");
    }
  );
});
