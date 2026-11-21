"""Shared, download-free definitions of the experiment's persistent assets."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import tomllib

MANIFEST_NAME = "specadapt-cache.json"


def wait_cache_ready(api, ref: str, timeout: float = 1800, interval: float = 10) -> None:
    """Wait for an owned, successfully created cache to finish ingestion."""
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        try:
            status = api.dataset_status(ref).lower()
        except Exception as error:
            http_status = getattr(getattr(error, "response", None), "status_code", None)
            if http_status not in (403, 404):
                raise
            # Large private datasets can remain invisible to this endpoint
            # throughout ingestion, even though the creation call succeeded.
            status = "waiting_for_visibility"
        if status != previous:
            print(f"Cache {ref}: {status}", flush=True)
            previous = status
        if status == "ready":
            return
        if status in {"error", "failed"}:
            raise RuntimeError(f"Kaggle failed to process {ref}.")
        time.sleep(interval)
    raise TimeoutError(f"Cache {ref} is still processing; retry after it is ready.")


def load_plan(path: Path) -> dict:
    with path.open("rb") as stream:
        plan = tomllib.load(stream)
    for category in ("models", "datasets"):
        entries = plan.get(category, [])
        if not entries:
            raise ValueError(f"No {category} configured in {path}.")
        keys = set()
        for entry in entries:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", entry["key"]) or entry["key"] in keys:
                raise ValueError(f"Invalid or duplicate asset key: {entry['key']}")
            keys.add(entry["key"])
            if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
                raise ValueError(f"{entry['key']} must use a full Hugging Face commit hash.")
            if not entry.get("files"):
                raise ValueError(f"{entry['key']} has no configured files.")
            for filename in entry["files"]:
                file_path = PurePosixPath(filename)
                if file_path.is_absolute() or ".." in file_path.parts or "\\" in filename:
                    raise ValueError(f"Invalid asset filename: {filename}")
    return plan


def cache_assets(plan: dict) -> list[dict]:
    """A separate dataset per model and one dataset holding all task sources."""
    assets = [{"key": model["key"], "kind": "model", "entries": [model]} for model in plan["models"]]
    assets.append({"key": "datasets", "kind": "datasets", "entries": plan["datasets"]})
    for asset in assets:
        encoded = json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()
        asset["fingerprint"] = hashlib.sha256(encoded).hexdigest()
        key = asset["key"].replace("_", "-")
        asset["slug"] = f"specadapt-{key}-{asset['fingerprint'][:12]}"
    return assets


def asset_files(asset: dict) -> list[str]:
    if asset["kind"] == "model":
        return asset["entries"][0]["files"]
    return [f"datasets/{entry['key']}/{name}" for entry in asset["entries"] for name in entry["files"]]


def validate_cache(directory: Path, asset: dict) -> dict:
    """Reject partial or mismatched assets without consulting Hugging Face."""
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != asset["fingerprint"]:
        raise ValueError(f"Cache revision mismatch in {directory}.")
    file_sizes = manifest.get("files", {})
    if set(file_sizes) != set(asset_files(asset)):
        raise ValueError(f"Cache file list is incomplete in {directory}.")
    for filename, expected_size in file_sizes.items():
        path = directory / filename
        if not path.is_file() or path.stat().st_size != expected_size or expected_size <= 0:
            raise ValueError(f"Cache file missing or truncated: {path}")
    return manifest
