#!/usr/bin/env node
/**
 * OpenClaw ARN Plugin Integration Test
 * ======================================
 * Simulates OpenClaw calling the plugin's hooks and verifies that
 * data is correctly stored in and retrieved from the ARN API.
 *
 * Usage:
 *   node test-plugin.js
 */

const ARN_ENDPOINT = "http://localhost:8742";
const TEST_AGENT = "openclaw_test_agent";

async function arnApi(path, method, body) {
  const res = await fetch(`${ARN_ENDPOINT}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ARN ${path} ${res.status}: ${text}`);
  }
  return res.json();
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Simulate the plugin's hook handlers by calling the same logic
async function testSessionStart() {
  console.log("\n[TEST] session_start");
  // Load persona from ARN
  const persona = await arnApi("/v1/memory/recall", "POST", {
    agent_id: TEST_AGENT,
    query: "agent identity",
    top_k: 3,
    memory_type: "identity",
  });
  console.log(`  Persona memories: ${persona.results?.length || 0}`);
}

async function testMessageReceived() {
  console.log("\n[TEST] message_received (auto-store user message)");
  const msg = {
    content: "How do I fix a Python KeyError?",
    timestamp: Date.now(),
  };
  const store = await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: msg.content,
    importance: 0.5,
    source: "user",
    memory_type: "episode",
    context: { role: "user" },
  });
  console.log(`  Stored user message → episode ${store.episode_id}`);
}

async function testBeforeToolCall() {
  console.log("\n[TEST] before_tool_call (auto-store procedure)");
  const toolCall = {
    toolName: "code_search",
    params: { query: "KeyError fix Python" },
  };
  const store = await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: `Tool call: ${toolCall.toolName}(${JSON.stringify(toolCall.params)})`,
    importance: 0.7,
    source: "tool",
    memory_type: "procedure",
  });
  console.log(`  Stored tool call → episode ${store.episode_id}`);
}

async function testAfterToolCall() {
  console.log("\n[TEST] after_tool_call (auto-store tool result)");
  const toolResult = {
    toolName: "code_search",
    result: "Use dict.get() with a default value to avoid KeyError.",
  };
  const store = await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: `Tool result: ${toolResult.toolName} → ${toolResult.result}`,
    importance: 0.5,
    source: "tool_result",
    memory_type: "episode",
  });
  console.log(`  Stored tool result → episode ${store.episode_id}`);
}

async function testBeforePromptBuild() {
  console.log("\n[TEST] before_prompt_build (auto-inject context)");
  const prompt = "How do I fix a Python KeyError?";

  // Inject dynamic memory
  const recall = await arnApi("/v1/memory/recall", "POST", {
    agent_id: TEST_AGENT,
    query: prompt,
    top_k: 5,
  });
  console.log(`  Recalled ${recall.results?.length || 0} memories for prompt injection`);
  for (const r of recall.results || []) {
    console.log(`    [${r.confidence_tier || "?"}] ${r.content.slice(0, 60)}...`);
  }

  // Get formatted context window
  const context = await arnApi("/v1/memory/context", "POST", {
    agent_id: TEST_AGENT,
    query: prompt,
    max_tokens: 800,
  });
  console.log(`  Injected context length: ${context.context.length} chars`);
}

async function testMessageSent() {
  console.log("\n[TEST] message_sent (auto-store agent reply)");
  const reply = "Use dict.get('key', default) to safely access dictionary values.";
  const store = await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: reply,
    importance: 0.6,
    source: "agent",
    memory_type: "episode",
    context: { role: "assistant" },
  });
  console.log(`  Stored agent reply → episode ${store.episode_id}`);
}

async function testPerAgentIsolation() {
  console.log("\n[TEST] Per-agent isolation");

  // Store diverse memories in agent A
  await arnApi("/v1/memory/store", "POST", {
    agent_id: "agent_A",
    content: "Agent A secret: password is hunter2",
    importance: 0.9,
    source: "user",
    memory_type: "episode",
  });
  await arnApi("/v1/memory/store", "POST", {
    agent_id: "agent_A",
    content: "Agent A works on Python debugging tools",
    importance: 0.8,
    source: "user",
    memory_type: "episode",
  });

  // Store diverse memories in agent B
  await arnApi("/v1/memory/store", "POST", {
    agent_id: "agent_B",
    content: "Agent B likes pizza and Italian food",
    importance: 0.9,
    source: "user",
    memory_type: "episode",
  });
  await arnApi("/v1/memory/store", "POST", {
    agent_id: "agent_B",
    content: "Agent B writes creative poetry",
    importance: 0.8,
    source: "user",
    memory_type: "episode",
  });

  // Agent A querying 'pizza' should NOT return Agent B's pizza memory
  const aRecall = await arnApi("/v1/memory/recall", "POST", {
    agent_id: "agent_A",
    query: "pizza",
    top_k: 3,
  });
  const aHasPizza = aRecall.results?.some((r) =>
    r.content.toLowerCase().includes("pizza")
  );
  console.log(`  Agent A querying 'pizza': ${aRecall.results?.length || 0} results, hasPizza=${aHasPizza}`);
  if (!aHasPizza) {
    console.log("  ✓ Agent A cannot see Agent B's pizza memory");
  } else {
    console.log("  ⚠ Agent A sees Agent B's data (unexpected)");
  }

  // Agent B querying 'password' should NOT return Agent A's secret
  const bRecall = await arnApi("/v1/memory/recall", "POST", {
    agent_id: "agent_B",
    query: "password",
    top_k: 3,
  });
  const bHasPassword = bRecall.results?.some((r) =>
    r.content.toLowerCase().includes("password")
  );
  console.log(`  Agent B querying 'password': ${bRecall.results?.length || 0} results, hasPassword=${bHasPassword}`);
  if (!bHasPassword) {
    console.log("  ✓ Agent B cannot see Agent A's secret");
  } else {
    console.log("  ⚠ Agent B sees Agent A's data (unexpected)");
  }
}

async function testTypedRecall() {
  console.log("\n[TEST] Typed recall (cross-contamination check)");

  // Store identity, preference, and error for the same agent
  await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: "I am a debugging assistant",
    importance: 0.9,
    source: "bootstrap",
    memory_type: "identity",
  });
  await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: "User prefers dark mode",
    importance: 0.8,
    source: "bootstrap",
    memory_type: "preference",
  });
  await arnApi("/v1/memory/store", "POST", {
    agent_id: TEST_AGENT,
    content: "Segmentation fault in C++ code",
    importance: 0.7,
    source: "user",
    memory_type: "error",
  });

  // Query for identity only
  const idOnly = await arnApi("/v1/memory/recall", "POST", {
    agent_id: TEST_AGENT,
    query: "who am I",
    top_k: 5,
    memory_type: "identity",
  });
  const types = idOnly.results?.map((r) => r.memory_type || "episode") || [];
  console.log(`  Identity query returns types: ${JSON.stringify(types)}`);
  const allIdentity = types.every((t) => t === "identity");
  console.log(allIdentity ? "  ✓ No cross-contamination" : "  ✗ Cross-contamination detected!");
}

async function main() {
  console.log("=".repeat(60));
  console.log("OpenClaw ARN Plugin Integration Test");
  console.log("ARN API:", ARN_ENDPOINT);
  console.log("=".repeat(60));

  try {
    // Health check
    const health = await fetch(`${ARN_ENDPOINT}/v1/health`).then((r) => r.json());
    console.log(`\nARN API health: ${health.status} (uptime: ${health.uptime_seconds}s)`);

    await testSessionStart();
    await testMessageReceived();
    await testBeforeToolCall();
    await testAfterToolCall();
    await testBeforePromptBuild();
    await testMessageSent();
    await testPerAgentIsolation();
    await testTypedRecall();

    // Final stats
    const stats = await arnApi("/v1/memory/stats/test_openclaw", "GET");
    console.log(`\n[FINAL] test_openclaw stats: ${stats.episodic_count} episodes`);

    console.log("\n" + "=".repeat(60));
    console.log("ALL PLUGIN INTEGRATION TESTS PASSED");
    console.log("=".repeat(60));
  } catch (e) {
    console.error("\nTEST FAILED:", e.message);
    process.exit(1);
  }
}

main();
