import tempfile
import unittest
from pathlib import Path

from arn_v9 import collab, collab_runner


class CollabRunnerTests(unittest.TestCase):
    def test_dry_run_claims_first_agent_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = collab_runner.run_cycle(
                repo_dir=Path(tmp),
                data_dir=Path(tmp) / "data",
                task_id="ARN-RUNNER-001",
                review_chain=None,
                execute=False,
                force=True,
                timeout=1,
            )

            state = result["state"]
            self.assertEqual(state["status"], "CLAIMED_CODEX")
            self.assertEqual(state["locked_by"], "codex")
            self.assertTrue(Path(result["final_report"]).exists())
            self.assertEqual(result["events"][0]["event"], "claimed")
            self.assertEqual(result["events"][1]["event"], "ran_agent")
            self.assertTrue(result["events"][1]["result"]["dry_run"])

    def test_build_prompt_points_to_previous_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            repo_dir = Path(tmp)
            collab.init_collab(data_dir, task_id="ARN-RUNNER-002")
            collab.claim_task(data_dir, "codex")
            handoff, validation = collab.create_handoff(
                data_dir,
                agent="codex",
                status="complete",
                task_summary="Initial work",
                changes="Changed nothing",
                verification="None",
                repo_dir=repo_dir,
            )
            self.assertTrue(validation["valid"])

            collab.claim_task(data_dir, "claude")
            prompt = collab_runner.build_prompt("claude", repo_dir, data_dir)
            self.assertIn(str(handoff), prompt)
            self.assertIn("You are claude", prompt)


class CollabRunnerFeedTests(unittest.TestCase):
    def test_feed_injected_into_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            repo_dir = Path(tmp)
            collab.init_collab(data_dir, task_id="ARN-FEED-001")
            collab.ensure_collab_dirs(data_dir)
            collab.write_feed(data_dir, message="pay attention to error handling", target="codex")

            prompt = collab_runner.build_prompt("codex", repo_dir, data_dir)
            self.assertIn("pay attention to error handling", prompt)
            self.assertIn("Human Context", prompt)

    def test_feed_for_other_agent_not_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            repo_dir = Path(tmp)
            collab.init_collab(data_dir, task_id="ARN-FEED-002")
            collab.ensure_collab_dirs(data_dir)
            collab.write_feed(data_dir, message="only for kimi", target="kimi")

            prompt = collab_runner.build_prompt("codex", repo_dir, data_dir)
            self.assertNotIn("only for kimi", prompt)

    def test_no_feeds_no_human_context_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            repo_dir = Path(tmp)
            collab.init_collab(data_dir, task_id="ARN-FEED-003")
            collab.ensure_collab_dirs(data_dir)

            prompt = collab_runner.build_prompt("claude", repo_dir, data_dir)
            self.assertNotIn("Human Context", prompt)


if __name__ == "__main__":
    unittest.main()
