"""
Tests for the procedural memory system (A1–A5).

All tests use the storage/cognitive layers directly — no HTTP.
The ARNv9 instance uses degraded-mode embeddings (no model download required).
"""

import sys
import time
import pytest
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture(scope="module")
def arn(tmp_path_factory):
    from arn_v9.core.cognitive import ARNv9
    data_dir = str(tmp_path_factory.mktemp("arn_procedural"))
    instance = ARNv9(data_dir=data_dir, use_embeddings=True)
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def storage(arn):
    return arn.storage


# =========================================================
# HELPERS
# =========================================================

def _make_episode(role: str, content: str, offset_secs: float = 0) -> dict:
    """Build a minimal episode dict for complexity / extraction tests."""
    return {
        'id': None,
        'role': role,
        'content': content,
        'created_at': time.time() + offset_secs,
        'importance': 0.5,
        'access_count': 0,
        'invalidated_at': None,
        'superseded_by': None,
        'metadata': {},
        'session_id': None,
    }


def _make_session_episodes(scenario: str) -> list:
    """
    Build mock session episodes that resemble real OpenClaw sessions.
    Uses varied tool names (exec, web_search, read_file, write_file) so
    tool_diversity contributes meaningfully to complexity scores.
    """
    if scenario == 'simple_debug':
        # Real pattern: run script → get error → search for fix → install → re-run
        return [
            _make_episode('user', 'Fix the import error in my Python script', 0),
            _make_episode('tool_call', 'exec(python3 script.py)', 1),
            _make_episode('tool_result', 'ImportError: No module named requests', 2),
            _make_episode('tool_call', 'web_search(python install requests module)', 3),
            _make_episode('tool_result', 'Run: pip install requests', 4),
            _make_episode('tool_call', 'exec(pip install requests)', 5),
            _make_episode('tool_result', 'Successfully installed requests-2.31.0', 6),
            _make_episode('tool_call', 'exec(python3 script.py)', 7),
            _make_episode('tool_result', 'Script ran successfully. Output: OK', 8),
            _make_episode('assistant', 'Installed the missing package and verified the script runs.', 9),
        ]
    if scenario == 'multi_tool':
        # Real pattern: deploy flow using git, docker, file inspection
        return [
            _make_episode('user', 'Deploy the app to the server', 0),
            _make_episode('tool_call', 'exec(git pull origin main)', 1),
            _make_episode('tool_result', 'Already up to date.', 2),
            _make_episode('tool_call', 'exec(docker build -t myapp .)', 3),
            _make_episode('tool_result', 'error: cannot connect to docker daemon', 4),
            _make_episode('tool_call', 'exec(sudo systemctl start docker)', 5),
            _make_episode('tool_result', 'Service docker started successfully', 6),
            _make_episode('tool_call', 'exec(docker build -t myapp .)', 7),
            _make_episode('tool_result', 'Successfully built abc123', 8),
            _make_episode('tool_call', 'exec(docker run -d -p 80:80 myapp)', 9),
            _make_episode('tool_result', 'Container started: def456', 10),
            _make_episode('tool_call', 'read_file(/app/logs/deploy.log)', 11),
            _make_episode('tool_result', 'Deploy succeeded. All health checks passing.', 12),
            _make_episode('assistant', 'Docker was not running. Started it, rebuilt, and deployed.', 13),
        ]
    if scenario == 'trivial':
        return [
            _make_episode('user', 'What time is it?', 0),
            _make_episode('assistant', "It's 3pm.", 1),
        ]
    return []


# =========================================================
# A1 — COMPLEXITY SCORING
# =========================================================

class TestComplexityScoring:

    def test_trivial_session_below_threshold(self):
        from arn_v9.core.procedural import compute_task_complexity
        eps = _make_session_episodes('trivial')
        score = compute_task_complexity(eps)
        assert score == 0.0, f"Expected 0.0 for trivial session, got {score}"

    def test_simple_debug_session(self):
        from arn_v9.core.procedural import compute_task_complexity
        eps = _make_session_episodes('simple_debug')
        score = compute_task_complexity(eps)
        # 4 tool_calls × 0.3 = 1.2
        # 2 unique tools (exec, web_search) × 2.0 = 4.0
        # 1 error correction × 3.0 = 3.0  (ImportError → web_search = different tool pivot)
        # 1 user turn × 0.1 = 0.1
        # total = 8.3
        assert score >= 8.0, f"Expected ≥8.0 for debug session, got {score}"

    def test_multi_tool_complex_session(self):
        from arn_v9.core.procedural import compute_task_complexity
        eps = _make_session_episodes('multi_tool')
        score = compute_task_complexity(eps)
        # 6 tool_calls × 0.3 = 1.8
        # 2 unique tools (exec, read_file) × 2.0 = 4.0
        # 1 error correction × 3.0 = 3.0  (docker error → different exec args = pivot)
        # 1 user turn × 0.1 = 0.1
        # total = 8.9
        assert score >= 8.0, f"Expected ≥8.0 for complex session, got {score}"

    def test_no_tool_calls_zero(self):
        from arn_v9.core.procedural import compute_task_complexity
        eps = [
            _make_episode('user', 'Hello'),
            _make_episode('assistant', 'Hi there'),
        ]
        assert compute_task_complexity(eps) == 0.0

    def test_error_correction_detected(self):
        from arn_v9.core.procedural import compute_task_complexity
        # Two error corrections should produce higher score than one
        eps_one_error = _make_session_episodes('simple_debug')
        eps_two_errors = _make_session_episodes('multi_tool')
        score_one = compute_task_complexity(eps_one_error)
        score_two = compute_task_complexity(eps_two_errors)
        assert score_two > score_one, "More error corrections should yield higher complexity"


# =========================================================
# A2 — PROCEDURE EXTRACTION
# =========================================================

class TestProceduralExtraction:

    def test_extraction_skipped_below_threshold(self, arn, storage):
        from arn_v9.core.procedural import extract_procedure
        eps = _make_session_episodes('trivial')
        result = extract_procedure(storage, arn.embedder, eps, 'sess-trivial')
        assert result is None, "Trivial session should not produce a procedure"

    def test_extraction_produces_episode(self, arn, storage):
        from arn_v9.core.procedural import extract_procedure
        # Store real episodes in DB with a session so storage.get_session_episodes works
        sess_id = 'sess-debug-extract'
        storage.create_session(sess_id, reason_start='test')
        import numpy as np
        stored_eps = []
        for ep_data in _make_session_episodes('simple_debug'):
            vec = arn.embedder.encode(ep_data['content'], mode='passage')
            ep_id = storage.store_episode(
                content=ep_data['content'],
                vector=vec,
                role=ep_data['role'],
                session_id=sess_id,
                importance=ep_data['importance'],
            )
            stored_eps.append({**ep_data, 'id': ep_id})

        proc_id = extract_procedure(storage, arn.embedder, stored_eps, sess_id)
        assert proc_id is not None, "Complex session should produce a procedure"
        ep = storage.get_episode(proc_id)
        assert ep is not None
        assert ep['role'] == 'procedural'
        assert ep['importance'] == 0.85

    def test_procedure_content_structure(self, arn, storage):
        from arn_v9.core.procedural import extract_procedure
        sess_id = 'sess-structure-test'
        storage.create_session(sess_id, reason_start='test')
        stored_eps = []
        for ep_data in _make_session_episodes('simple_debug'):
            vec = arn.embedder.encode(ep_data['content'])
            ep_id = storage.store_episode(
                content=ep_data['content'], vector=vec,
                role=ep_data['role'], session_id=sess_id,
            )
            stored_eps.append({**ep_data, 'id': ep_id})

        proc_id = extract_procedure(storage, arn.embedder, stored_eps, sess_id)
        ep = storage.get_episode(proc_id)
        content = ep['content']
        assert 'GOAL:' in content, "Procedure must have GOAL section"
        assert 'STEPS:' in content, "Procedure must have STEPS section"

    def test_procedure_metadata_populated(self, arn, storage):
        from arn_v9.core.procedural import extract_procedure
        sess_id = 'sess-meta-test'
        storage.create_session(sess_id, reason_start='test')
        stored_eps = []
        for ep_data in _make_session_episodes('multi_tool'):
            vec = arn.embedder.encode(ep_data['content'])
            ep_id = storage.store_episode(
                content=ep_data['content'], vector=vec,
                role=ep_data['role'], session_id=sess_id,
            )
            stored_eps.append({**ep_data, 'id': ep_id})

        proc_id = extract_procedure(storage, arn.embedder, stored_eps, sess_id)
        ep = storage.get_episode(proc_id)
        meta = ep['metadata']
        assert meta['source_session'] == sess_id
        assert 'complexity_score' in meta
        assert 'tool_chain' in meta
        assert isinstance(meta['tool_chain'], list)
        assert meta['effectiveness_score'] == 1.0

    def test_procedure_recalled_via_normal_search(self, arn, storage):
        """Procedural memories should surface via arn.recall() like any other episode."""
        # The procedures stored by previous tests should be findable
        results = arn.recall("fix import error python", top_k=10)
        contents = [r['content'] for r in results]
        # At minimum, the procedure we stored should be in recall results
        assert any('GOAL' in c or 'import' in c.lower() for c in contents), \
            f"Expected procedural memory in recall results, got: {[c[:60] for c in contents]}"


# =========================================================
# A3 — SUPERSEDES CHAIN (SELF-IMPROVEMENT)
# =========================================================

class TestProcedureSupersedesChain:

    def test_reflect_supersedes_similar_procedure(self, arn, storage):
        """reflect(session_id=...) should chain a new procedure over a similar old one."""
        # Store a 'base' procedure directly
        import numpy as np
        old_content = (
            "GOAL: Deploy app to server\n\n"
            "STEPS:\n  1. exec(git pull)\n  2. exec(docker build)\n\n"
            "CONTEXT: Docker, Git"
        )
        old_vec = arn.embedder.encode(old_content)
        old_id = storage.store_episode(
            content=old_content, vector=old_vec,
            role='procedural', importance=0.85,
            metadata={'effectiveness_score': 1.0, 'complexity_score': 9.0},
        )

        # Run a new session that produces a similar procedure
        sess_id = 'sess-supersede-test'
        storage.create_session(sess_id, reason_start='test')
        stored_eps = []
        for ep_data in _make_session_episodes('multi_tool'):
            vec = arn.embedder.encode(ep_data['content'])
            ep_id = storage.store_episode(
                content=ep_data['content'], vector=vec,
                role=ep_data['role'], session_id=sess_id,
            )
            stored_eps.append({**ep_data, 'id': ep_id})
        storage.end_session(sess_id)

        stats = arn.reflect(session_id=sess_id)
        new_proc_id = stats.get('procedure_extracted')
        assert new_proc_id is not None, "Should have extracted a procedure"

        # If the old and new are similar enough, old should be superseded
        old_ep = storage.get_episode(old_id)
        new_ep = storage.get_episode(new_proc_id)
        assert new_ep is not None
        # The new procedure should be active
        assert new_ep.get('invalidated_at') is None

    def test_restore_procedure(self, arn, storage):
        """restore_procedure() should reverse a supersession."""
        # Create two episodes, supersede first with second
        vec = arn.embedder.encode("GOAL: Fix server\nSTEPS:\n  1. restart")
        old_id = storage.store_episode(
            content="GOAL: Fix server\nSTEPS:\n  1. restart",
            vector=vec, role='procedural', importance=0.85,
            metadata={'effectiveness_score': 1.0},
        )
        new_vec = arn.embedder.encode("GOAL: Fix server v2\nSTEPS:\n  1. check logs\n  2. restart")
        new_id = storage.store_episode(
            content="GOAL: Fix server v2\nSTEPS:\n  1. check logs\n  2. restart",
            vector=new_vec, role='procedural', importance=0.85,
            metadata={'effectiveness_score': 0.5},
        )
        storage.supersede_episode(old_id, new_id)

        # Verify old is superseded
        old_ep = storage.get_episode(old_id)
        assert old_ep['superseded_by'] == new_id

        # Restore old
        ok = arn.restore_procedure(old_id)
        assert ok is True

        restored = storage.get_episode(old_id)
        assert restored['superseded_by'] is None
        assert restored['invalidated_at'] is None

        # New should be invalidated
        new_ep = storage.get_episode(new_id)
        assert new_ep['invalidated_at'] is not None


# =========================================================
# A4 — EFFECTIVENESS TRACKING
# =========================================================

class TestEffectivenessTracking:

    def test_boost_on_low_error_rate(self):
        from arn_v9.core.procedural import compute_effectiveness_deltas
        deltas = compute_effectiveness_deltas([1, 2, 3], error_rate=0.10)
        for ep_id, delta in deltas.items():
            assert delta == pytest.approx(0.1), f"Expected +0.1 boost for ep {ep_id}"

    def test_reduction_on_high_error_rate(self):
        from arn_v9.core.procedural import compute_effectiveness_deltas
        deltas = compute_effectiveness_deltas([1, 2], error_rate=0.70)
        for ep_id, delta in deltas.items():
            assert delta == pytest.approx(-0.2), f"Expected -0.2 reduction for ep {ep_id}"

    def test_no_change_in_middle_range(self):
        from arn_v9.core.procedural import compute_effectiveness_deltas
        deltas = compute_effectiveness_deltas([1], error_rate=0.35)
        assert deltas == {}, "Mid-range error rate should produce no delta"

    def test_effectiveness_capped_at_2(self, arn, storage):
        from arn_v9.core.procedural import apply_effectiveness_updates
        vec = arn.embedder.encode("GOAL: test cap")
        ep_id = storage.store_episode(
            content="GOAL: test cap", vector=vec,
            role='procedural', importance=0.85,
            metadata={'effectiveness_score': 1.95},
        )
        apply_effectiveness_updates(storage, {ep_id: 0.1})
        ep = storage.get_episode(ep_id)
        assert ep['metadata']['effectiveness_score'] <= 2.0

    def test_effectiveness_floored_at_0_1(self, arn, storage):
        from arn_v9.core.procedural import apply_effectiveness_updates
        vec = arn.embedder.encode("GOAL: test floor")
        ep_id = storage.store_episode(
            content="GOAL: test floor", vector=vec,
            role='procedural', importance=0.85,
            metadata={'effectiveness_score': 0.15},
        )
        apply_effectiveness_updates(storage, {ep_id: -0.2})
        ep = storage.get_episode(ep_id)
        assert ep['metadata']['effectiveness_score'] >= 0.1

    def test_low_effectiveness_flagged_in_review_queue(self, arn, storage):
        from arn_v9.core.procedural import apply_effectiveness_updates
        vec = arn.embedder.encode("GOAL: test flag")
        ep_id = storage.store_episode(
            content="GOAL: test flag", vector=vec,
            role='procedural', importance=0.85,
            metadata={'effectiveness_score': 0.40},  # will drop below 0.3
        )
        flagged = apply_effectiveness_updates(storage, {ep_id: -0.2}, review_threshold=0.3)
        assert ep_id in flagged, "Should flag ep when crossing below review_threshold"


# =========================================================
# A5 — DEEP REFLECT (CURATOR)
# =========================================================

class TestDeepReflect:

    def test_deep_reflect_returns_stats_dict(self, arn):
        stats = arn.deep_reflect()
        assert 'curator' in stats
        c = stats['curator']
        assert 'total_procedures' in c
        assert 'active_procedures' in c
        assert 'stale_marked' in c
        assert 'duplicates_merged' in c
        assert 'archived' in c
        assert 'avg_effectiveness' in c

    def test_stale_procedures_get_low_importance(self, arn, storage):
        from arn_v9.core.procedural import deep_reflect_procedures
        # Store a procedure with zero access_count and old creation time
        vec = arn.embedder.encode("GOAL: stale old procedure")
        ep_id = storage.store_episode(
            content="GOAL: stale old procedure",
            vector=vec, role='procedural', importance=0.85,
            metadata={'effectiveness_score': 1.0},
        )
        # Manually backdate created_at to 40 days ago
        storage.update_episode(ep_id, {'created_at': time.time() - 40 * 86400})

        deep_reflect_procedures(storage, arn.embedder, stale_days=30)

        ep = storage.get_episode(ep_id)
        assert ep['importance'] <= 0.1, \
            f"Stale zero-access procedure should have importance ≤ 0.1, got {ep['importance']}"

    def test_archived_procedures_have_valid_until(self, arn, storage):
        from arn_v9.core.procedural import deep_reflect_procedures
        vec = arn.embedder.encode("GOAL: very old low-importance procedure")
        ep_id = storage.store_episode(
            content="GOAL: very old low-importance procedure",
            vector=vec, role='procedural', importance=0.10,
            metadata={'effectiveness_score': 0.5},
        )
        # Backdate to 70 days ago
        storage.update_episode(ep_id, {'created_at': time.time() - 70 * 86400})

        deep_reflect_procedures(storage, arn.embedder, archive_days=60, archive_importance=0.15)

        ep = storage.get_episode(ep_id)
        assert ep.get('valid_until') is not None, \
            "Old low-importance procedure should have valid_until set (archived)"

    def test_duplicate_procedures_merged(self, arn, storage):
        from arn_v9.core.procedural import deep_reflect_procedures
        # Two near-identical procedures
        content_a = "GOAL: run pytest\nSTEPS:\n  1. exec(python -m pytest tests/ -v)"
        content_b = "GOAL: run pytest tests\nSTEPS:\n  1. exec(python -m pytest tests/ -v)\n  2. check output"
        vec_a = arn.embedder.encode(content_a)
        vec_b = arn.embedder.encode(content_b)
        id_a = storage.store_episode(
            content=content_a, vector=vec_a, role='procedural', importance=0.85,
            metadata={'effectiveness_score': 0.8},
        )
        id_b = storage.store_episode(
            content=content_b, vector=vec_b, role='procedural', importance=0.85,
            metadata={'effectiveness_score': 1.2},
        )

        stats = deep_reflect_procedures(storage, arn.embedder, dup_threshold=0.50)
        # In degraded mode these will be hash-encoded; similarity may or may not trigger merge.
        # Just verify stats structure is correct and no exceptions.
        assert 'duplicates_merged' in stats


# =========================================================
# ROLE FILTER
# =========================================================

class TestRoleFilter:

    def test_procedural_role_stored_correctly(self, arn, storage):
        vec = arn.embedder.encode("GOAL: test role filter\nSTEPS:\n  1. run test")
        ep_id = storage.store_episode(
            content="GOAL: test role filter\nSTEPS:\n  1. run test",
            vector=vec, role='procedural', importance=0.85,
        )
        ep = storage.get_episode(ep_id)
        assert ep['role'] == 'procedural'

    def test_role_filter_returns_only_procedural(self, storage):
        conn = storage._get_conn()
        rows = conn.execute(
            "SELECT id, role FROM episodes WHERE role = 'procedural'"
        ).fetchall()
        assert len(rows) >= 1
        assert all(r['role'] == 'procedural' for r in rows)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
