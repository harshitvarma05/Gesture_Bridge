"""Configure cross-platform emergency phone notifications without editing code."""

import argparse
from pathlib import Path
import secrets


def read_settings(path):
    settings = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return settings
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings


def main():
    parser = argparse.ArgumentParser(description="Set up Gesture-Bridge phone alerts")
    parser.add_argument("--contact", default="Doctor / caregiver")
    parser.add_argument("--location", default="Location not configured")
    parser.add_argument("--topic-url", help="Existing private ntfy topic URL")
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--enable-live", action="store_true", help="allow real network notifications")
    parser.add_argument("--output", default=".env")
    args = parser.parse_args()

    topic_url = args.topic_url or f"https://ntfy.sh/gesture-bridge-{secrets.token_urlsafe(18)}"
    if not topic_url.startswith("https://"):
        parser.error("--topic-url must use HTTPS")

    output = Path(args.output)
    settings = read_settings(output)
    settings.update({
        "GESTURE_BRIDGE_LIVE_ALERTS": "1" if args.enable_live else "0",
        "GESTURE_BRIDGE_NTFY_URL": topic_url,
        "GESTURE_BRIDGE_CONTACT_NAME": args.contact,
        "GESTURE_BRIDGE_LOCATION": args.location,
        "GESTURE_BRIDGE_ALERT_COUNTDOWN_SECONDS": str(max(2, min(args.countdown, 30))),
    })
    content = "# Gesture-Bridge local configuration (keep private)\n" + "\n".join(
        f"{key}={value}" for key, value in sorted(settings.items())
    ) + "\n"
    output.write_text(content, encoding="utf-8")

    mode = "LIVE" if args.enable_live else "DEMO"
    print(f"Saved {output} in {mode} mode.")
    print(f"Doctor/caregiver subscription URL: {topic_url}")
    print("Install the ntfy phone app, subscribe to that exact private URL, then run:")
    print("  python hardware_self_test.py --send-test-alert")
    if not args.enable_live:
        print("Real sending is disabled. Re-run with --enable-live after the recipient subscribes.")


if __name__ == "__main__":
    main()
