import tempfile
import unittest
from pathlib import Path

from arn_v9 import collab


class CollabTests(unittest.TestCase):
    def test_init_claim_handoff_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = collab.init_collab(tmp, task_id="ARN-001")
            self.assertEqual(state["status"], "IDLE")
            self.assertEqual(collab.next_agent(state), "codex")

            state = collab.claim_task(tmp, "codex")
            self.assertEqual(state["status"], "CLAIMED_CODEX")
            self.assertEqual(state["locked_by"], "codex")

            handoff, validation = collab.create_handoff(
                tmp,
                agent="codex",
                status="complete",
                task_summary="Build collab state machine.",
                changes="Added file-based state transitions.",
                verification="Ran focused unit tests.",
                repo_dir=Path(tmp),
            )
            self.assertTrue(handoff.exists())
            self.assertTrue(validation["valid"], validation["errors"])

            state = collab.read_state(tmp)
            self.assertEqual(state["status"], "HANDOFF_CODEX")
            self.assertIsNone(state["locked_by"])
            self.assertEqual(collab.next_agent(state), "claude")

    def test_prevents_wrong_agent_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.init_collab(tmp, task_id="ARN-002")
            with self.assertRaisesRegex(RuntimeError, "next agent is codex"):
                collab.claim_task(tmp, "kimi")

    def test_prevents_double_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.init_collab(tmp, task_id="ARN-003")
            collab.claim_task(tmp, "codex")
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                collab.claim_task(tmp, "codex")

    def test_validate_rejects_bad_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "bad.md"
            handoff.write_text("---\nagent: \"codex\"\n---\n\nbody\n", encoding="utf-8")
            result = collab.validate_handoff(handoff)
            self.assertFalse(result["valid"])
            self.assertIn("missing required field: handoff_version", result["errors"])

    def test_custom_review_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = collab.init_collab(
                tmp,
                task_id="ARN-004",
                review_chain=["kimi", "codex", "claude"],
            )
            self.assertEqual(collab.next_agent(state), "kimi")
            collab.claim_task(tmp, "kimi")
            collab.create_handoff(
                tmp,
                agent="kimi",
                status="no_issues",
                task_summary="Review only.",
                changes="No changes.",
                verification="Inspected handoff.",
                repo_dir=Path(tmp),
            )
            self.assertEqual(collab.next_agent(collab.read_state(tmp)), "codex")


class CollabFeedTests(unittest.TestCase):
    def test_write_and_read_feed(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.ensure_collab_dirs(tmp)
            entry = collab.write_feed(tmp, message="focus on edge cases", target="claude")
            self.assertEqual(entry["to"], "claude")
            self.assertEqual(entry["from"], "human")
            self.assertIn("focus on edge cases", entry["message"])

            feeds = collab.read_feeds(tmp, agent="claude", limit=5)
            self.assertEqual(len(feeds), 1)
            self.assertEqual(feeds[0]["message"], "focus on edge cases")

    def test_feed_all_visible_to_any_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.ensure_collab_dirs(tmp)
            collab.write_feed(tmp, message="general broadcast", target="all")
            for agent in ("codex", "claude", "kimi"):
                feeds = collab.read_feeds(tmp, agent=agent, limit=5)
                self.assertEqual(len(feeds), 1)

    def test_feed_targeted_not_visible_to_other_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.ensure_collab_dirs(tmp)
            collab.write_feed(tmp, message="only for codex", target="codex")
            self.assertEqual(len(collab.read_feeds(tmp, agent="claude", limit=5)), 0)
            self.assertEqual(len(collab.read_feeds(tmp, agent="codex", limit=5)), 1)

    def test_write_feed_rejects_invalid_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.ensure_collab_dirs(tmp)
            with self.assertRaises(ValueError):
                collab.write_feed(tmp, message="bad target", target="gpt4")


class CollabListHandoffsTests(unittest.TestCase):
    def test_list_handoffs_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(collab.list_handoffs(tmp), [])

    def test_list_handoffs_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            collab.init_collab(tmp, task_id="ARN-LH-001")
            collab.claim_task(tmp, "codex")
            collab.create_handoff(
                tmp,
                agent="codex",
                status="complete",
                task_summary="Test list_handoffs",
                changes="None",
                verification="Manual",
                repo_dir=Path(tmp),
            )
            handoffs = collab.list_handoffs(tmp, limit=5)
            self.assertEqual(len(handoffs), 1)
            self.assertEqual(handoffs[0]["agent"], "codex")
            self.assertEqual(handoffs[0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
