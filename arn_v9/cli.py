"""
ARN v9 CLI entry point.

Wrapper around arn_v9.scripts.arn_cli for pip-installed usage.
"""

import sys
from arn_v9.scripts.arn_cli import main

if __name__ == "__main__":
    sys.exit(main())
