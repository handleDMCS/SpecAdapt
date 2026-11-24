"""Follow a detached Kaggle run from a terminal: live heartbeats, stall warning, end notice.

    uv run python -m demo.kaggle_watch <timestamp>

Read-only: it never changes or cancels the run. Stop it with Ctrl+C at any time.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time

from demo.kaggle_run import PROJECT_ROOT, get_api

STALL_SECONDS = 600
FINAL = {"complete", "error", "cancelled", "canceled"}


def parse_event(line: str) -> dict | None:
    """One server-sent log event: {"stream_name", "time", "data"}, or None for other lines."""
    if not line.startswith("data:"):
        return None
    try:
        event = json.loads(line[len("data:"):].strip())
    except ValueError:
        return None
    return event if isinstance(event, dict) and "data" in event else None


def is_progress(text: str) -> bool:
    return text.startswith(("[heartbeat", "Traceback", "Submission status", "Result JSON")) or "Error" in text


def status_of(api, kernel: str) -> str:
    return getattr(api.kernels_status(kernel).status, "name", "unknown").lower()


def stream_events(api, owner: str, slug: str):
    """Yield log lines from Kaggle's live session log; returns when the stream closes or idles."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name, request.kernel_slug, request.version_label = owner, slug, ""
    request.wait_for_logs_url_seconds = 5
    with api.build_kaggle_client() as client:
        service = client.kernels.kernels_api_client
        # The SDK sets no read timeout; without one an idle stream blocks forever.
        http = service._client
        http._init_session()
        original_send = http._session.send

        def send(*args, **kwargs):
            kwargs["timeout"] = (15, 60)
            return original_send(*args, **kwargs)

        http._session.send = send
        response = service.get_kernel_session_logs_stream(request)
        try:
            for line in response.iter_lines(chunk_size=1):
                if line:
                    yield line.decode("utf-8", errors="backslashreplace")
        finally:
            response.close()


def watch(timestamp: str) -> int:
    stage = PROJECT_ROOT / "build/submissions" / timestamp
    metadata_path = stage / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"No staged submission {timestamp}; use the timestamp from the run card.")
    kernel = json.loads(metadata_path.read_text(encoding="utf-8"))["id"].lower()
    owner, slug = kernel.split("/")
    api = get_api()
    print(f"Watching {kernel} (https://www.kaggle.com/code/{kernel}). Ctrl+C stops watching, not the run.",
          flush=True)
    last_time, last_progress, warned = -1.0, time.monotonic(), False
    while True:
        status = status_of(api, kernel)
        if status in FINAL:
            print(f"\a\nKaggle: {status}. Collect: uv run python -m demo.kaggle_run --resume {timestamp}",
                  flush=True)
            return 0 if status == "complete" else 1
        if status == "running":
            try:
                for line in stream_events(api, owner, slug):
                    event = parse_event(line)
                    # The stream replays the session from the start; skip what was already shown.
                    if event is None or event.get("time", 0) <= last_time:
                        continue
                    last_time = event.get("time", last_time)
                    text = event["data"].rstrip("\n")
                    if is_progress(text):
                        print(text, flush=True)
                        last_progress, warned = time.monotonic(), False
                    warned = warn_if_stalled(last_progress, warned)
            except Exception as error:  # Timeouts and closed streams: reconnect after a status check.
                if not re.search(r"timeout|timed out|connection|chunked", f"{type(error).__name__} {error}", re.I):
                    print(f"(log stream: {type(error).__name__}: {error})", flush=True)
            warned = warn_if_stalled(last_progress, warned)
        else:
            print(f"Kaggle: {status}", flush=True)
            last_progress = time.monotonic()
        time.sleep(30)


def warn_if_stalled(last_progress: float, warned: bool) -> bool:
    if not warned and time.monotonic() - last_progress > STALL_SECONDS:
        print(f"\aSTALLED: no heartbeat for {STALL_SECONDS // 60} minutes. Check the kernel page.", flush=True)
        return True
    return warned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timestamp", help="The timestamp printed on the run card")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}T\d{12}Z", args.timestamp):
        parser.error("Expected the UTC timestamp from the run card.")
    try:
        return watch(args.timestamp)
    except KeyboardInterrupt:
        print("\nStopped watching. The run continues on Kaggle.")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
