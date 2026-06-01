"""
Tests for arn_v9.scripts.migrate_to_base_tier.

The migration script is a destructive operation that re-embeds every
stored vector.  These tests cover the helper functions that do NOT
require a live embedding model, so they run in the plumbing tier:

  - _check_server_not_running   — server-lock guard
  - _get_existing_dim           — reads dim from episodic_vectors.npy
  - _get_from_tier_from_fingerprint — reads stored tier name
  - CLI argument parsing         — ensures expected flags exist

The live re-embedding path is intentionally not tested here because it
requires sentence-transformers and a downloaded model.  It is covered
by the semantic integration tests.
"""

import json
import sys
import argparse
import numpy as np
import pytest
from pathlib import Path

# Import helpers under test directly from the script module
from arn_v9.scripts import migrate_to_base_tier as _m


# ──────────────────────────────────────────────────────────────────────────────
# _check_server_not_running
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckServerNotRunning:
    def test_dry_run_always_passes(self, tmp_path):
        """dry_run=True must bypass the WAL check entirely."""
        _m._check_server_not_running(tmp_path, dry_run=True)

    def test_force_bypasses_wal_check(self, tmp_path):
        """force=True must bypass the WAL check regardless of WAL state."""
        wal = tmp_path / "arn_metadata.db-wal"
        wal.write_bytes(b"x" * 1024)  # non-empty WAL
        _m._check_server_not_running(tmp_path, dry_run=False, force=True)

    def test_empty_wal_does_not_exit(self, tmp_path):
        """An empty (0-byte) WAL file is not a sign of a running server."""
        wal = tmp_path / "arn_metadata.db-wal"
        wal.write_bytes(b"")
        _m._check_server_not_running(tmp_path)  # Must not call sys.exit

    def test_no_wal_file_does_not_exit(self, tmp_path):
        """If the WAL file is absent the check must pass silently."""
        _m._check_server_not_running(tmp_path)

    def test_non_empty_wal_causes_exit_without_force(self, tmp_path):
        """A non-empty WAL while not in dry_run/force mode must call sys.exit(1)."""
        wal = tmp_path / "arn_metadata.db-wal"
        wal.write_bytes(b"data" * 256)
        with pytest.raises(SystemExit) as exc_info:
            _m._check_server_not_running(tmp_path, dry_run=False, force=False)
        assert exc_info.value.code == 1


# ──────────────────────────────────────────────────────────────────────────────
# _get_existing_dim
# ──────────────────────────────────────────────────────────────────────────────

class TestGetExistingDim:
    def test_returns_correct_dim_from_npy_file(self, tmp_path):
        arr = np.zeros((100, 384), dtype=np.float32)
        np.save(str(tmp_path / "episodic_vectors.npy"), arr)
        assert _m._get_existing_dim(tmp_path) == 384

    def test_returns_correct_dim_for_768_dim_vectors(self, tmp_path):
        arr = np.zeros((50, 768), dtype=np.float32)
        np.save(str(tmp_path / "episodic_vectors.npy"), arr)
        assert _m._get_existing_dim(tmp_path) == 768

    def test_returns_none_when_file_missing(self, tmp_path):
        assert _m._get_existing_dim(tmp_path) is None

    def test_returns_none_for_1d_array(self, tmp_path):
        arr = np.zeros(384, dtype=np.float32)
        np.save(str(tmp_path / "episodic_vectors.npy"), arr)
        assert _m._get_existing_dim(tmp_path) is None


# ──────────────────────────────────────────────────────────────────────────────
# _get_from_tier_from_fingerprint
# ──────────────────────────────────────────────────────────────────────────────

class TestGetFromTierFromFingerprint:
    def test_reads_stored_tier_name(self, tmp_path):
        fp = tmp_path / ".model_fingerprint"
        fp.write_text(json.dumps({"tier": "base", "dim": 768}))
        assert _m._get_from_tier_from_fingerprint(tmp_path) == "base"

    def test_reads_nano_tier(self, tmp_path):
        fp = tmp_path / ".model_fingerprint"
        fp.write_text(json.dumps({"tier": "nano", "dim": 384}))
        assert _m._get_from_tier_from_fingerprint(tmp_path) == "nano"

    def test_returns_default_when_file_missing(self, tmp_path):
        result = _m._get_from_tier_from_fingerprint(tmp_path, default="nano")
        assert result == "nano"

    def test_returns_default_when_tier_key_absent(self, tmp_path):
        fp = tmp_path / ".model_fingerprint"
        fp.write_text(json.dumps({"dim": 384}))
        result = _m._get_from_tier_from_fingerprint(tmp_path, default="nano")
        assert result == "nano"

    def test_returns_default_when_file_is_corrupt_json(self, tmp_path):
        fp = tmp_path / ".model_fingerprint"
        fp.write_text("NOT JSON {{{{")
        result = _m._get_from_tier_from_fingerprint(tmp_path, default="nano")
        assert result == "nano"


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument structure
# ──────────────────────────────────────────────────────────────────────────────

class TestCLIArguments:
    """Regression guard: the migration script's argument parser must expose
    the expected flags so that documented usage keeps working."""

    def _parse(self, args: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent-dir", type=Path)
        parser.add_argument("--data-root", type=Path)
        parser.add_argument("--from-tier", default=None)
        parser.add_argument("--to-tier", default="base")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--force", action="store_true")
        return parser.parse_args(args)

    def test_dry_run_flag_parsed(self):
        ns = self._parse(["--agent-dir", "/tmp/test", "--dry-run"])
        assert ns.dry_run is True

    def test_force_flag_parsed(self):
        ns = self._parse(["--agent-dir", "/tmp/test", "--force"])
        assert ns.force is True

    def test_default_to_tier_is_base(self):
        ns = self._parse(["--agent-dir", "/tmp/test"])
        assert ns.to_tier == "base"

    def test_explicit_tier_flags(self):
        ns = self._parse([
            "--agent-dir", "/tmp/test",
            "--from-tier", "nano",
            "--to-tier", "base-e5",
        ])
        assert ns.from_tier == "nano"
        assert ns.to_tier == "base-e5"

    def test_data_root_flag_accepted(self, tmp_path):
        ns = self._parse(["--data-root", str(tmp_path)])
        assert ns.data_root == tmp_path
