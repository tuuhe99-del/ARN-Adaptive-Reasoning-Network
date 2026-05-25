#!/usr/bin/env python3
"""
ARN Live Integration Test — Multi-session red team simulation
"""
import json, requests, sys

API = "http://127.0.0.1:8742"
AGENT = "Website_Red_Team"

results = {
    "session1_store": [],
    "session1_recall": None,
    "session2_store": [],
    "session2_recall": None,
    "session3_recalls": [],
    "links": None,
    "stats": None,
}

def store(content, source, importance, memory_type="episode"):
    r = requests.post(f"{API}/v1/memory/store", json={
        "agent_id": AGENT,
        "content": content,
        "source": source,
        "importance": importance,
        "memory_type": memory_type,
    })
    r.raise_for_status()
    return r.json()

def recall(query, top_k=5, memory_type=None):
    payload = {"agent_id": AGENT, "query": query, "top_k": top_k}
    if memory_type:
        payload["memory_type"] = memory_type
    r = requests.post(f"{API}/v1/memory/recall", json=payload)
    r.raise_for_status()
    return r.json()

def list_links():
    r = requests.post(f"{API}/v1/memory/links", json={"agent_id": AGENT})
    r.raise_for_status()
    return r.json()

def get_stats():
    r = requests.get(f"{API}/v1/memory/stats/{AGENT}")
    r.raise_for_status()
    return r.json()

def delete_agent():
    r = requests.delete(f"{API}/v1/memory/agent", json={"agent_id": AGENT, "confirm": True})
    return r.json()

def fmt_result(r):
    return {
        "id": r.get("id"),
        "content": r["content"][:80] + "..." if len(r["content"]) > 80 else r["content"],
        "similarity": r.get("similarity"),
        "calibrated_confidence": r.get("calibrated_confidence"),
        "source": r.get("source"),
        "memory_type": r.get("memory_type"),
    }

def main():
    # Optional: clean slate
    print("=== Cleaning slate ===")
    try:
        delete_agent()
        print("Deleted previous agent data")
    except Exception as e:
        print(f"No previous data (or error): {e}")

    # ===== SESSION 1 =====
    print("\n=== SESSION 1: Identity & Setup ===")
    s1_memories = [
        ("My name is Red Team Alpha. I test web applications for security vulnerabilities.", "agent", 0.9, "identity"),
        ("Target system: internal dashboard at 127.0.0.1:8745. Testing for XSS, open redirects, auth bypass.", "user", 0.8, "episode"),
        ("Tool call: curl_http — GET http://127.0.0.1:8745/dashboard", "tool:curl_http", 0.6, "procedure"),
        ("Finding: Dashboard loads without authentication. No login required to view memories.", "agent", 0.85, "episode"),
    ]
    for content, source, importance, mtype in s1_memories:
        resp = store(content, source, importance, mtype)
        results["session1_store"].append({"content": content[:50], "episode_id": resp["episode_id"]})
        print(f"  Stored episode {resp['episode_id']}: {content[:50]}...")

    print("\n--- Session 1 Recall: 'who am I and what am I testing' ---")
    s1_recall = recall("who am I and what am I testing", top_k=5)
    results["session1_recall"] = s1_recall
    for r in s1_recall["results"]:
        print(f"  {fmt_result(r)}")

    # ===== SESSION 2 =====
    print("\n=== SESSION 2: Finding & Fix ===")
    s2_memories = [
        ("Second session. Continuing red team of internal dashboard.", "agent", 0.5, "episode"),
        ("Found: Relations tab SVG neuron graph loads nodes from /v1/memory/recall without authentication. Could expose agent memories to network-adjacent attackers.", "agent", 0.9, "episode"),
        ("Recommended fix: add API key requirement or localhost-only bind for production deployments.", "agent", 0.8, "episode"),
    ]
    for content, source, importance, mtype in s2_memories:
        resp = store(content, source, importance, mtype)
        results["session2_store"].append({"content": content[:50], "episode_id": resp["episode_id"]})
        print(f"  Stored episode {resp['episode_id']}: {content[:50]}...")

    print("\n--- Session 2 Recall: 'what vulnerabilities did I find' ---")
    s2_recall = recall("what vulnerabilities did I find", top_k=5)
    results["session2_recall"] = s2_recall
    for r in s2_recall["results"]:
        print(f"  {fmt_result(r)}")

    # ===== SESSION 3 =====
    print("\n=== SESSION 3: Recall Quality Check (no new stores) ===")
    queries = [
        ("red team identity and name", "identity memory from session 1"),
        ("security vulnerabilities dashboard", "both findings"),
        ("tool calls made during testing", "curl_http procedure memory"),
        ("authentication bypass findings", "auth findings"),
    ]
    for q, expected in queries:
        print(f"\n--- Query: '{q}' ---")
        resp = recall(q, top_k=5)
        results["session3_recalls"].append({"query": q, "response": resp})
        for r in resp["results"]:
            print(f"  {fmt_result(r)}")

    # ===== LINKS CHECK =====
    print("\n=== LINKS CHECK ===")
    links = list_links()
    results["links"] = links
    print(f"Total links: {links['count']}")
    for link in links.get("links", []):
        print(f"  {link}")

    # ===== STATS =====
    print("\n=== STATS ===")
    stats = get_stats()
    results["stats"] = stats
    print(json.dumps(stats, indent=2))

    # ===== SUMMARY =====
    print("\n=== SUMMARY ===")
    failures = []

    # Session 1 check
    s1_top = s1_recall["results"][0] if s1_recall["results"] else None
    if s1_top and s1_top.get("calibrated_confidence", 0) < 0.3:
        failures.append(f"Session 1 recall top result confidence too low: {s1_top.get('calibrated_confidence')}")

    # Session 2 check
    s2_top = s2_recall["results"][0] if s2_recall["results"] else None
    if s2_top and s2_top.get("calibrated_confidence", 0) < 0.3:
        failures.append(f"Session 2 recall top result confidence too low: {s2_top.get('calibrated_confidence')}")

    # Session 3 checks
    for rec in results["session3_recalls"]:
        for r in rec["response"]["results"]:
            sim = r.get("similarity", 0)
            conf = r.get("calibrated_confidence", 0)
            if sim > 0.4 and conf < 0.3:
                failures.append(f"Session 3 query '{rec['query']}': result similarity={sim:.3f} but confidence={conf:.3f}")

    # Links check
    if links["count"] == 0:
        failures.append("No auto-links found after stores")

    # ID field check
    for rec in results["session3_recalls"]:
        for r in rec["response"]["results"]:
            if r.get("id") is None:
                failures.append(f"Missing 'id' field in recall result for query '{rec['query']}'")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL CHECKS PASSED")

    # Write results to file
    with open("/Users/hustle/.arn_data/collab/live_integration_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to ~/.arn_data/collab/live_integration_results.json")


if __name__ == "__main__":
    main()
