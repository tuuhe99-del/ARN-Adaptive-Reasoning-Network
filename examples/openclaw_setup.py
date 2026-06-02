"""
How to connect ARN to OpenClaw.

This script shows what `arn connect` does under the hood.
Run `arn connect` from the CLI for the one-command version.
"""

import subprocess
import sys


def check_arn_running(port: int = 7900) -> bool:
    """Check if the ARN daemon is already running."""
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/health", timeout=2)
        return True
    except Exception:
        return False


def main():
    print("ARN → OpenClaw integration setup\n")

    # Step 1: Start the ARN daemon if not running
    if check_arn_running():
        print("✓ ARN daemon is already running on port 7900")
    else:
        print("Starting ARN daemon on port 7900...")
        subprocess.run(["arn", "server", "--daemon", "--port", "7900"], check=True)
        print("✓ ARN daemon started")

    # Step 2: Wire up the OpenClaw plugin
    print("\nConnecting ARN to OpenClaw...")
    result = subprocess.run(["arn", "connect"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ OpenClaw plugin installed and registered")
        print("  - Memories will be auto-injected before every LLM call")
        print("  - All messages, tool calls, and outputs are captured")
        print("  - Post-session reflection runs when the conversation ends")
        print("\nAgent tools available:")
        print("  arn_recall   — targeted memory search")
        print("  arn_pin      — pin a permanent fact")
        print("  arn_forget   — remove an outdated memory")
        print("  arn_sessions — list past sessions")
        print("  arn_review   — check flagged contradictions")
    else:
        print(f"Connection failed:\n{result.stderr}")
        sys.exit(1)

    print("\nVerify with: arn status")
    print("Disconnect with: arn disconnect")


if __name__ == "__main__":
    main()
