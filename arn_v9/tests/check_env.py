#!/usr/bin/env python3
"""
ARN v9 Environment Check
==========================
Run this BEFORE the test suite to verify the environment is ready.
Checks Python version, dependencies, model availability, and disk space.

Usage:
    python3 arn_v9/tests/check_env.py

Exit codes:
    0 = ready for full 44/44 test suite
    1 = can only run plumbing tests (model missing)
    2 = broken (missing critical dependencies)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check():
    errors = []
    warnings = []

    print("ARN v9 Environment Check")
    print("=" * 50)

    # Python version
    v = sys.version_info
    if v >= (3, 10):
        print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")
    else:
        errors.append(f"Python {v.major}.{v.minor} < 3.10")
        print(f"  ✗ Python {v.major}.{v.minor} (need 3.10+)")

    # numpy
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError:
        errors.append("numpy not installed")
        print("  ✗ numpy NOT INSTALLED (pip install numpy)")

    # sqlite3
    try:
        import sqlite3
        print(f"  ✓ sqlite3 {sqlite3.sqlite_version}")
    except ImportError:
        errors.append("sqlite3 not available")
        print("  ✗ sqlite3 NOT AVAILABLE")

    # sentence-transformers
    try:
        import sentence_transformers
        print(f"  ✓ sentence-transformers {sentence_transformers.__version__}")
    except ImportError:
        warnings.append("sentence-transformers not installed")
        print("  ⚠ sentence-transformers NOT INSTALLED")
        print("    Install: pip install sentence-transformers")
        print("    Without this, only plumbing tests pass (semantic tests skip)")

    # torch (comes with sentence-transformers)
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError:
        if 'sentence-transformers' not in str(warnings):
            warnings.append("torch not installed")
        print("  ⚠ torch NOT INSTALLED (installed with sentence-transformers)")

    # Model availability
    model_loaded = False
    try:
        from arn_v9.core.embeddings import EmbeddingEngine
        engine = EmbeddingEngine(use_model=True)
        if not engine.is_degraded:
            import numpy as np
            v1 = engine.encode("test")
            v2 = engine.encode("test two")
            sim = float(np.dot(v1, v2))
            print(f"  ✓ Embedding model loaded (sanity sim={sim:.3f})")
            model_loaded = True
        else:
            warnings.append("Embedding model failed to load")
            print("  ⚠ Embedding model DEGRADED")
    except Exception as e:
        warnings.append(f"Embedding model error: {e}")
        print(f"  ⚠ Embedding model error: {e}")

    # arn_v9 package
    try:
        from arn_v9 import ARNv9
        print(f"  ✓ arn_v9 package importable")
    except ImportError as e:
        errors.append(f"arn_v9 import failed: {e}")
        print(f"  ✗ arn_v9 import failed: {e}")

    # Disk space
    try:
        import shutil
        usage = shutil.disk_usage(os.path.expanduser("~"))
        free_mb = usage.free / 1024 / 1024
        if free_mb > 100:
            print(f"  ✓ Disk space: {free_mb:.0f} MB free")
        else:
            warnings.append(f"Low disk space: {free_mb:.0f} MB")
            print(f"  ⚠ Low disk space: {free_mb:.0f} MB free")
    except Exception:
        pass

    # Summary
    print()
    print("=" * 50)
    if errors:
        print("RESULT: BROKEN — fix critical errors before testing")
        for e in errors:
            print(f"  ✗ {e}")
        return 2
    elif warnings:
        print("RESULT: PARTIAL — plumbing tests will pass, semantic tests will SKIP")
        for w in warnings:
            print(f"  ⚠ {w}")
        if not model_loaded:
            print()
            print("To get full 44/44:")
            print("  pip install sentence-transformers")
            print("  # Then re-run this check")
        return 1
    else:
        print("RESULT: READY — full 44/44 test suite should pass")
        return 0


if __name__ == "__main__":
    sys.exit(check())
