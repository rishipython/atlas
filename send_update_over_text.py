#!/usr/bin/env python3
"""
Send a text message update with a required [CODEX UPDATE] prefix.

Usage:
  python send_update_over_text.py "your update text"
"""

from __future__ import annotations

import argparse
import subprocess
import sys


PHONE_NUMBER = "+17039278738"
PREFIX = "[CODEX UPDATE]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prefixed SMS update to 703-927-8738 from 703-927-8738."
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="Text content of the update message.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    body = f"{PREFIX} {' '.join(args.text)}"

    applescript = """
on run argv
    set targetNumber to item 1 of argv
    set msgBody to item 2 of argv
    tell application "Messages"
        set targetService to missing value
        try
            set targetService to 1st service whose service type = iMessage
        end try
        if targetService is missing value then
            set targetService to 1st service
        end if
        set targetBuddy to buddy targetNumber of targetService
        send msgBody to targetBuddy
    end tell
end run
"""

    try:
        subprocess.run(
            ["osascript", "-e", applescript, PHONE_NUMBER, body],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("osascript not found. This script requires macOS.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        if err:
            print(f"Failed to send message: {err}", file=sys.stderr)
        else:
            print("Failed to send message via Messages.", file=sys.stderr)
        return 1

    notification_script = """
on run argv
    set notificationBody to item 1 of argv
    display notification notificationBody with title "CODEX UPDATE"
end run
"""
    try:
        subprocess.run(
            ["osascript", "-e", notification_script, body],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        # Notification failure should not make send fail.
        pass

    print("Message sent via Messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
