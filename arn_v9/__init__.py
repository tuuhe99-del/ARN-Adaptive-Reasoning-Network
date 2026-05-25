"""
ARN v9 — Adaptive Reasoning Network
=====================================
Brain-inspired persistent memory for AI agents.

Two ways to use:

1. Plugin API (Level 1 — tool-based):
   from arn_v9.plugin import ARNPlugin

2. Core engine (advanced):
   from arn_v9.core.cognitive import ARNv9
"""

from .core.cognitive import ARNv9
from .plugin import ARNPlugin

__version__ = "9.0.0"
__all__ = ["ARNv9", "ARNPlugin"]
