"""
Integration test: store → recall → inject loop using the red team agent_id.

The agent_id used here is "openclaw-redteam" which matches the directory
already present at ~/.arn_data/openclaw-redteam/

Run:
    python3 arn_v9/scripts/test_redteam_loop.py [--agent-id openclaw-redteam]
"""

import os
import sys
import json
import argparse
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from arn_v9.plugin import ARNPlugin


def run_test(agent_id: str, data_root: str):
    print(f"\nARN store→recall integration test")
    print(f"  agent_id  : {agent_id}")
    print(f"  data_root : {data_root}")
    print()

    failures = []

    with ARNPlugin(agent_id=agent_id, data_root=data_root,
                   use_embeddings=True) as plugin:

        # 1. Store 4 episodic memories across different topics
        memories = [
            {
                "content": "Website_Red_Team identity: security researcher focused on SSRF and injection vulnerabilities",
                "importance": 0.9,
                "tags": ["identity", "security"],
                "source": "agent",
                "memory_type": "identity",
            },
            {
                "content": "Discovered open redirect at /api/redirect?url=<payload> — allows bypass of same-origin policy",
                "importance": 0.8,
                "tags": ["finding", "open-redirect"],
                "source": "agent",
                "memory_type": "error",
            },
            {
                "content": "Tool call: nuclei -t exposures -target https://example.com — returned 3 potential info disclosures",
                "importance": 0.7,
                "tags": ["tool-call", "nuclei"],
                "source": "tool:nuclei",
                "memory_type": "procedure",
            },
            {
                "content": "Fix applied: added SSRF blocklist to /api/fetch endpoint to prevent internal network requests",
                "importance": 0.75,
                "tags": ["fix", "ssrf"],
                "source": "agent",
                "memory_type": "fact",
            },
        ]

        stored_ids = []
        for m in memories:
            r = plugin.store(
                content=m["content"],
                importance=m["importance"],
                tags=m["tags"],
                source=m["source"],
                memory_type=m["memory_type"],
            )
            if not r.get("stored"):
                failures.append(f"store failed: {m['content'][:60]}")
            else:
                stored_ids.append(r["episode_id"])
                print(f"  STORED id={r['episode_id']} domain={r['domain']} error={r['prediction_error']:.3f}")

        if len(stored_ids) != 4:
            failures.append(f"Expected 4 stored, got {len(stored_ids)}")

        # 2. Recall with 3 different queries and check results
        queries = [
            ("What is the agent's identity and role?", stored_ids[0] if stored_ids else None),
            ("open redirect vulnerability found in the API", stored_ids[1] if len(stored_ids) > 1 else None),
            ("SSRF fix applied to fetch endpoint", stored_ids[3] if len(stored_ids) > 3 else None),
        ]

        print()
        for query, expected_id in queries:
            results = plugin.recall(query, top_k=5)
            if not results:
                failures.append(f"No results for query: {query[:60]}")
                continue

            top = results[0]
            print(f"  QUERY: {query[:60]!r}")
            print(f"    top id={top.get('id')} sim={top['similarity']:.3f} "
                  f"conf={top['calibrated_confidence']:.3f} tier={top['confidence_tier']}")
            print(f"    content: {top['content'][:80]!r}")

            # Check calibrated_confidence is reasonable (> 0.3 for any relevant result)
            if top["similarity"] >= 0.40 and top["calibrated_confidence"] < 0.25:
                failures.append(
                    f"calibrated_confidence too low: "
                    f"sim={top['similarity']:.3f} but conf={top['calibrated_confidence']:.3f}"
                )
            else:
                print(f"    confidence OK: sim={top['similarity']:.3f} → conf={top['calibrated_confidence']:.3f}")

        # 3. Verify context window injection
        print()
        ctx = plugin.get_context_window(query="security research findings", max_tokens=500)
        if not ctx or len(ctx) < 50:
            failures.append("get_context_window returned empty or trivial output")
        else:
            print(f"  CONTEXT WINDOW ({len(ctx)} chars):\n{ctx[:300]}")

        # 4. Check stats
        stats = plugin.get_stats()
        print(f"\n  STATS: episodes={stats['episodic_count']} semantics={stats['semantic_count']}")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


def main():
    parser = argparse.ArgumentParser(description="ARN red-team agent store→recall integration test")
    parser.add_argument("--agent-id", default="openclaw-redteam",
                        help="Agent namespace to use for the test")
    parser.add_argument("--data-root",
                        default=os.environ.get("ARN_DATA_ROOT",
                                               os.path.expanduser("~/.arn_data")),
                        help="Path to ARN data root")
    parser.add_argument("--fresh", action="store_true",
                        help="Use a fresh temp dir (doesn't pollute live data)")
    args = parser.parse_args()

    if args.fresh:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(run_test(args.agent_id, tmp))
    else:
        sys.exit(run_test(args.agent_id, args.data_root))


if __name__ == "__main__":
    main()
