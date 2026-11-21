"""Cache the experiment assets on Kaggle, then report paths and row counts.

Submit this file with kaggle_run.py; do not execute it on the laptop.
"""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from demo.assets import MANIFEST_NAME, asset_files, cache_assets, load_plan, validate_cache, wait_cache_ready


def read_secret(name: str) -> str | None:
    names = (name, "HUGGING_FACE_TOKEN") if name == "HF_TOKEN" else (name,)
    for candidate in names:
        value = os.environ.get(candidate)
        if value:
            return value
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    except Exception:
        client = None
    if client is not None:
        for candidate in names:
            try:
                value = client.get_secret(candidate)
                if value:
                    return value
            except Exception:
                continue
    # This optional input is created/attached only with explicit authorization.
    # It lives under read-only Kaggle inputs, never in source or result folders.
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for credentials_path in input_root.rglob("specadapt-credentials.json"):
            credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
            for candidate in names:
                value = credentials.get(candidate)
                if isinstance(value, str) and value:
                    return value
    return None


def ensure_module(name: str, requirement: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", requirement], check=True)
        return importlib.import_module(name)


def count_dataset_rows(directory: Path, asset: dict) -> dict:
    parquet = ensure_module("pyarrow.parquet", "pyarrow>=15")
    counts = {}
    for entry in asset["entries"]:
        splits = {}
        for filename in entry["files"]:
            split = Path(filename).parent.name
            path = directory / "datasets" / entry["key"] / filename
            splits[split] = splits.get(split, 0) + parquet.ParquetFile(path).metadata.num_rows
        counts[entry["key"]] = {"splits": splits, "total": sum(splits.values())}
    return counts


def download_asset(directory: Path, asset: dict, hf_token: str | None) -> dict:
    hub = ensure_module("huggingface_hub", "huggingface_hub>=0.28,<2")
    importlib.import_module("huggingface_hub.utils").disable_progress_bars()
    directory.mkdir(parents=True, exist_ok=True)
    for entry in asset["entries"]:
        target = directory if asset["kind"] == "model" else directory / "datasets" / entry["key"]
        print(f"Downloading {entry['repo']} at {entry['revision']}", flush=True)
        hub.snapshot_download(repo_id=entry["repo"], revision=entry["revision"],
                              repo_type="model" if asset["kind"] == "model" else "dataset",
                              allow_patterns=entry["files"], local_dir=str(target), token=hf_token)
    sizes = {}
    for filename in asset_files(asset):
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Download is incomplete: {path}")
        sizes[filename] = path.stat().st_size
    manifest = {"fingerprint": asset["fingerprint"], "key": asset["key"],
                "kind": asset["kind"], "sources": asset["entries"], "files": sizes}
    if asset["kind"] == "datasets":
        manifest["rows"] = count_dataset_rows(directory, asset)
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class Publisher:
    """Upload original assets as private Kaggle datasets, outside notebook output."""

    def __init__(self, owner: str):
        token = read_secret("KAGGLE_API_TOKEN")
        if not token:
            raise RuntimeError("Publishing needs KAGGLE_API_TOKEN in the remote session. "
                               "Use Add-ons secrets for a browser-started run, or explicitly attach the approved "
                               "private credential dataset for a CLI run. The laptop's .env is not bundled.")
        os.environ["KAGGLE_API_TOKEN"] = token
        try:
            version = importlib.metadata.version("kaggle")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "2.2.4":
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "kaggle==2.2.4"], check=True)
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()
        self.api.authenticate()
        if self.api.get_config_value(self.api.CONFIG_NAME_USER) != owner:
            raise RuntimeError("The Kaggle secret's account does not match the submission's cache owner.")
        self.owner = owner
        self.existing = set()
        page = 1
        while True:
            entries = self.api.dataset_list(mine=True, search="specadapt", page=page) or []
            if not entries:
                break
            self.existing.update(entry.ref for entry in entries if entry is not None)
            page += 1

    def __call__(self, directory: Path, asset: dict) -> None:
        ref = f"{self.owner}/{asset['slug']}"
        metadata = {"id": ref, "title": asset["slug"], "licenses": [{"name": "other"}],
                    "description": "Private SpecAdapt cache of pinned original Hugging Face assets. "
                                   "Original source licenses apply; see included model cards and license files."}
        (directory / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if ref not in self.existing:
            print(f"Publishing private cache: {ref}", flush=True)
            response = self.api.dataset_create_new(str(directory), public=False, quiet=True,
                                                  convert_to_csv=False, dir_mode="tar",
                                                  ignore_patterns=[".cache", ".cache/**", "**/.cache/**"])
            if response.error:
                raise RuntimeError(f"Could not publish {ref}: {response.error}")
            self.existing.add(ref)
        wait_cache_ready(self.api, ref)


def prepare(plan_path: Path, input_root: Path, scratch_root: Path, owner: str,
            *, secret_reader=read_secret, downloader=download_asset, publisher_factory=Publisher,
            row_counter=count_dataset_rows) -> dict:
    plan = load_plan(plan_path)
    mounted = {}
    if input_root.exists():
        for manifest_path in input_root.rglob(MANIFEST_NAME):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mounted[manifest.get("fingerprint")] = manifest_path.parent
    model_paths, dataset_paths, dataset_rows, pending = {}, {}, {}, []
    publisher = None
    hf_token = None
    token_checked = False
    for asset in cache_assets(plan):
        directory = mounted.get(asset["fingerprint"])
        local = scratch_root / asset["slug"]
        if directory is not None:
            manifest = validate_cache(directory, asset)
            state = "cache_hit"
        elif (local / MANIFEST_NAME).is_file():
            directory = local
            manifest = validate_cache(local, asset)
            state = "session_cache_hit"
            if not (local / ".published").is_file():
                if publisher is None:
                    publisher = publisher_factory(owner)
                publisher(local, asset)
                (local / ".published").touch()
        else:
            if not token_checked:
                hf_token = secret_reader("HF_TOKEN")
                token_checked = True
            if asset["kind"] == "model" and asset["entries"][0].get("gated") and not hf_token:
                pending.append(asset["key"])
                print(f"[pending] {asset['key']}: needs HF_TOKEN with approved Llama 3.1 access.", flush=True)
                continue
            # Authenticate the publisher before downloading large assets.
            if publisher is None:
                publisher = publisher_factory(owner)
            directory = local
            try:
                manifest = downloader(local, asset, hf_token)
            except Exception as error:
                http_status = getattr(getattr(error, "response", None), "status_code", None)
                if (asset["kind"] == "model" and asset["entries"][0].get("gated")
                        and http_status in (401, 403)):
                    pending.append(asset["key"])
                    print(f"[pending] {asset['key']}: Hugging Face denied model access (HTTP {http_status}). "
                          "Other assets will still be cached.", flush=True)
                    continue
                raise
            validate_cache(local, asset)
            state = "downloaded"
            publisher(local, asset)
            (local / ".published").touch()
        if asset["kind"] == "model":
            model_paths[asset["key"]] = str(directory)
            print(f"[model] {asset['key']} {state} path={directory} revision={asset['entries'][0]['revision']}", flush=True)
        else:
            dataset_rows = row_counter(directory, asset)
            if dataset_rows != manifest["rows"]:
                raise RuntimeError(f"Dataset row counts changed in cache {directory}.")
            for entry in asset["entries"]:
                key = entry["key"]
                dataset_paths[key] = str(directory / "datasets" / key)
                rows = dataset_rows[key]
                print(f"[dataset] {key} {state} path={dataset_paths[key]} total_rows={rows['total']} "
                      f"splits={json.dumps(rows['splits'], sort_keys=True)}", flush=True)
    report = {"MODEL_PATHS": model_paths, "DATASET_PATHS": dataset_paths,
              "DATASET_ROWS": dataset_rows, "PENDING_MODELS": pending, "CACHE_COMPLETE": not pending}
    print(f"Cache complete: {report['CACHE_COMPLETE']}; models={len(model_paths)}; dataset_sources={len(dataset_paths)}", flush=True)
    return report


def main() -> dict:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Submit this script with: uv run python -m demo.kaggle_run --cpu src/demo/prepare_data.py")
    project_root = Path(os.environ.get("SPECADAPT_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    owner = os.environ.get("SPECADAPT_CACHE_OWNER")
    if not owner:
        raise RuntimeError("Missing cache owner; run this script through kaggle_run.py.")
    os.environ.setdefault("HF_HOME", "/tmp/specadapt-hf")
    return prepare(project_root / "asset_plan.toml", Path("/kaggle/input"), Path("/tmp/specadapt-cache"), owner)


if __name__ == "__main__":
    globals().update(main())
