"""Submit root-relative Python scripts to one Kaggle session, in argument order."""

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
import tomllib
from urllib.parse import urlparse
import zipfile

from demo.kaggle_runtime import ARTIFACTS_DIR, write_result

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRS = {".git", ".venv", "build", "results", "logs", "__pycache__", ARTIFACTS_DIR}
KERNEL_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?")


def root_file(filename: str, root: Path = PROJECT_ROOT) -> Path:
    if Path(filename).is_absolute() or PureWindowsPath(filename).drive:
        raise ValueError(f"Use a path relative to the project root: {filename}")
    path = (root / filename.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or path outside the project: {filename}")
    return path


def get_api():
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def git_state(root: Path) -> dict:
    """The commit a submission was bundled from, and whether the tree differed from it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": bool(status.strip())}


def job_file(root: Path, filename: str) -> str:
    """Validate a job TOML locally, so a typo fails before any Kaggle time is spent."""
    path = root_file(filename, root)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"Expected a TOML job file: {filename}")
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    minutes = config.get("max_minutes")
    if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or minutes <= 0):
        raise ValueError(f"max_minutes must be a positive number in {filename}")
    return path.relative_to(root).as_posix()


def name_kernel(metadata: dict, name: str) -> dict:
    """Run in <owner>/specadapt-<name>, so parallel jobs never replace each other's version."""
    if not KERNEL_NAME.fullmatch(name):
        raise ValueError("--kernel takes lowercase letters, digits and hyphens, at most 40 characters.")
    owner = metadata["id"].split("/")[0]
    # Kaggle derives the slug from the title, so both must agree.
    return {**metadata, "id": f"{owner}/specadapt-{name}", "title": f"specadapt-{name}"}


def read_metadata(filename: str, cpu: bool = False) -> dict:
    metadata = json.loads(root_file(filename).read_text(encoding="utf-8-sig"))
    if not re.fullmatch(r"[\w-]+/[\w-]+", metadata.get("id", "")):
        raise ValueError("kernel-metadata.json needs an id in owner/kernel-slug format.")
    metadata.update(code_file="submission.py", language="python", kernel_type="script")
    metadata.pop("id_no", None)
    if cpu:
        metadata.update(enable_gpu=False, enable_tpu=False)
        metadata.pop("machine_shape", None)
    return metadata


def ready_cache_sources(api, root: Path, owner: str) -> list[str]:
    from demo.assets import cache_assets, load_plan, wait_cache_ready

    plan_path = root / "asset_plan.toml"
    if not plan_path.exists():
        return []
    expected = {f"{owner}/{asset['slug']}" for asset in cache_assets(load_plan(plan_path))}
    found = set()
    page = 1
    while True:
        datasets = api.dataset_list(mine=True, search="specadapt", page=page) or []
        if not datasets:
            break
        found.update(dataset.ref for dataset in datasets if dataset is not None)
        page += 1
    sources = []
    for ref in sorted(expected & found):
        wait_cache_ready(api, ref)
        sources.append(ref)
    return sources


def build_submission(root: Path, scripts: list[str], timestamp: str, owner: str,
                     provenance: dict | None = None) -> tuple[str, dict]:
    normalized, hashes = [], {}
    for filename in scripts:
        path = root_file(filename, root)
        if path.suffix.lower() != ".py":
            raise ValueError(f"Expected a Python script: {filename}")
        with tokenize.open(path) as stream:
            compile(stream.read(), str(path), "exec")
        relative = path.relative_to(root).as_posix()
        normalized.append(relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    submission = {"timestamp": timestamp, "scripts": normalized,
                  "script_sha256": hashes, "cache_owner": owner, **(provenance or {})}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder, dirs, filenames in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS and not name.startswith(".")
                             and not (Path(folder) / name).is_symlink())
            for name in sorted(filenames):
                path = Path(folder) / name
                if not name.startswith(".") and not path.is_symlink() and path.suffix.lower() in {".py", ".toml"}:
                    archive.write(path, path.relative_to(root).as_posix())
        runtime = PROJECT_ROOT / "src/demo/kaggle_runtime.py"
        if "src/demo/kaggle_runtime.py" not in archive.namelist():
            archive.write(runtime, "src/demo/kaggle_runtime.py")
            archive.writestr("src/demo/__init__.py", "")
        missing = set(normalized) - set(archive.namelist())
        if missing:
            raise ValueError(f"Scripts are in excluded directories: {sorted(missing)}")
    encoded = base64.b64encode(payload.getvalue()).decode("ascii")
    source = (
        "import base64, io, json, sys, tempfile, zipfile\n"
        "from pathlib import Path\n"
        f"_submission = json.loads({json.dumps(submission)!r})\n"
        "with tempfile.TemporaryDirectory(prefix='specadapt-project-') as _directory:\n"
        "    _root = Path(_directory)\n"
        f"    with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as _bundle:\n"
        "        _bundle.extractall(_root)\n"
        "    sys.path.insert(0, str(_root / 'src'))\n"
        "    from demo.kaggle_runtime import run_submission\n"
        "    run_submission(_root, _submission)\n"
    )
    return source, submission


def wait_for_run(api, kernel: str, poll_interval: float, wait_timeout: float) -> str:
    deadline = time.monotonic() + wait_timeout
    previous = None
    while time.monotonic() < deadline:
        response = api.kernels_status(kernel)
        status = getattr(response.status, "name", str(response.status)).lower()
        if status != previous:
            print(f"Kaggle: {status}", flush=True)
            previous = status
        if status in {"complete", "error", "cancelled", "canceled"}:
            return status
        if status not in {"queued", "running"}:
            raise RuntimeError(f"Unexpected Kaggle status: {status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Stopped waiting for {kernel}; the remote job may still be running.")


def kernel_ref(ref: str, fallback: str) -> str:
    path = urlparse(ref).path.strip("/")
    if path.startswith("code/"):
        path = path.removeprefix("code/")
    return path if re.fullmatch(r"[\w-]+/[\w-]+", path) else fallback


def fetch_result(api, kernel: str, submission: dict, status: str, destination: Path) -> tuple[Path, dict]:
    filename = f"submission-{submission['timestamp']}.json"
    destination.mkdir(parents=True, exist_ok=True)
    result = destination / filename
    with tempfile.TemporaryDirectory(prefix="specadapt-result-") as temporary:
        api.kernels_output(kernel, temporary, file_pattern=rf"^(?:{re.escape(filename)}|.*\.log)$", quiet=True)
        downloaded = Path(temporary) / filename
        if downloaded.is_file():
            report = json.loads(downloaded.read_text(encoding="utf-8"))
            if report.get("timestamp") != submission["timestamp"]:
                raise RuntimeError("Kaggle returned a different submission's result.")
        else:
            if status == "complete":
                raise RuntimeError("Kaggle completed without this submission's result JSON.")
            logs = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                             for path in sorted(Path(temporary).rglob("*.log")))
            message = f"Kaggle job ended with status: {status}"
            report = {**submission, "status": "platform_error", "kaggle_status": status,
                      "error": message, "elapsed_seconds": None,
                      "run_log": logs or f"{message}\n", "outputs": {}}
    return result, write_result(result, report)


def fetch_artifacts(api, kernel: str, report: dict, destination: Path) -> Path:
    """Download the listed artifacts and keep them only if every size and SHA-256 matches."""
    expected = report["artifacts"]
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="specadapt-artifacts-", dir=destination) as temporary:
        api.kernels_output(kernel, temporary, file_pattern=rf"^{ARTIFACTS_DIR}/", force=True, quiet=True)
        for name, info in expected.items():
            path = Path(temporary) / name
            if not path.is_file():
                raise RuntimeError(f"Kaggle did not return the artifact {name}.")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != info["bytes"] or digest != info["sha256"]:
                raise RuntimeError(f"Size or SHA-256 mismatch for the artifact {name}.")
        for name in expected:
            target = destination / name
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite the existing artifact {target}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(Path(temporary) / name, target)
    return destination / ARTIFACTS_DIR


def print_report(report: dict, run_dir: Path) -> None:
    """Print a job's summary.json if it wrote one; otherwise the whole run log, as before."""
    summary = report.get("outputs", {}).get("summary.json")
    if summary is None:
        print(report["run_log"])
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if report["status"] != "complete":
        print(f"\nStatus: {report['status']}: {report.get('error')}\nLast 20 log lines:")
        print("\n".join(report["run_log"].splitlines()[-20:]))
    print(f"\nRun folder: {run_dir}")


def run_card(kernel: str, submission: dict, metadata: dict, expected_minutes: float | None) -> str:
    gpu = metadata.get("machine_shape") if metadata.get("enable_gpu") in (True, "true") else "CPU"
    dirty = " (dirty tree)" if submission.get("git_dirty") else ""
    lines = [f"Timestamp:  {submission['timestamp']}",
             f"Kernel:     https://www.kaggle.com/code/{kernel}",
             f"Machine:    {gpu}",
             f"Git:        {submission.get('git_sha')}{dirty}"]
    if submission.get("job"):
        lines.append(f"Job:        {submission['job']}")
    if expected_minutes:
        lines.append(f"Expected:   about {expected_minutes:g} minutes")
    lines.append(f"Collect:    uv run python -m demo.kaggle_run --resume {submission['timestamp']}")
    return "\n".join(lines)


def positive_number(value: str) -> float:
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("Value must be a positive finite number.")
    return number


def staged_submission(stage: Path) -> dict:
    report_path = stage / "submission.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    # Recover older staged runs without executing the generated source.
    tree = ast.parse((stage / "submission.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_submission"
                                                for target in node.targets):
            return json.loads(ast.literal_eval(node.value.args[0]))
    raise ValueError(f"No submission record in {stage}.")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scripts", nargs="*", help="Python files relative to the root; executed in this order")
    parser.add_argument("--resume", help="Resume waiting/downloading a previously submitted UTC timestamp")
    parser.add_argument("--metadata", default="kernel-metadata.json", help="Root-relative Kaggle metadata file")
    parser.add_argument("--cpu", action="store_true", help="Run without a GPU (recommended for preparation/tests)")
    parser.add_argument("--dry-run", action="store_true", help="Stage and validate; do not contact Kaggle")
    parser.add_argument("--no-cache", action="store_true", help="Skip automatic attachment of experiment caches")
    parser.add_argument("--dataset", action="append", default=[], help="Additional owner/dataset-slug input (repeatable)")
    parser.add_argument("--kernel", help="Run in kernel <owner>/specadapt-<name> instead of the metadata id")
    parser.add_argument("--detach", action="store_true", help="Push, print the run card and exit without waiting")
    parser.add_argument("--expected-minutes", type=positive_number, help="Shown on the run card")
    parser.add_argument("--job", help="Root-relative job TOML; scripts read it with demo.kaggle_runtime.job_config()")
    parser.add_argument("--poll-interval", type=positive_number, default=10)
    parser.add_argument("--wait-timeout", type=positive_number, default=50400, help="Maximum wait in seconds")
    args = parser.parse_args(argv)
    if bool(args.scripts) == bool(args.resume):
        parser.error("Supply scripts, or --resume TIMESTAMP, but not both.")
    if args.resume and (args.detach or args.kernel or args.job):
        parser.error("--detach, --kernel and --job apply to new submissions only.")
    try:
        if args.resume:
            if not re.fullmatch(r"\d{8}T\d{12}Z", args.resume):
                raise ValueError("Invalid submission timestamp.")
            stage = PROJECT_ROOT / "build/submissions" / args.resume
            submission = staged_submission(stage)
            metadata = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
            timestamp = submission["timestamp"]
            api = None if args.dry_run else get_api()
        else:
            metadata = read_metadata(args.metadata, args.cpu)
            if args.kernel:
                metadata = name_kernel(metadata, args.kernel)
            owner = metadata["id"].split("/")[0]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            provenance = git_state(PROJECT_ROOT)
            if args.job:
                provenance["job"] = job_file(PROJECT_ROOT, args.job)
            if provenance["git_dirty"]:
                print("Warning: the working tree has uncommitted changes; results will cite a dirty tree.",
                      file=sys.stderr)
            source, submission = build_submission(PROJECT_ROOT, args.scripts, timestamp, owner, provenance)
            api = None if args.dry_run else get_api()
            caches = [] if args.no_cache or api is None else ready_cache_sources(api, PROJECT_ROOT, owner)
            metadata["dataset_sources"] = list(dict.fromkeys([*metadata.get("dataset_sources", []), *caches, *args.dataset]))
            stage = PROJECT_ROOT / "build/submissions" / timestamp
            stage.mkdir(parents=True)
            (stage / "submission.py").write_text(source, encoding="utf-8")
            (stage / "submission.json").write_text(json.dumps(submission, indent=2), encoding="utf-8")
            (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Submission (UTC): {timestamp}")
        print(f"Scripts: {' -> '.join(submission['scripts'])}")
        print(f"Attached inputs: {len(metadata['dataset_sources'])}")
        if args.dry_run:
            print(f"Dry run staged at: {stage}")
            return 0
        kernel = metadata["id"].lower()
        if not args.resume:
            response = api.kernels_push(str(stage))
            invalid = [*response.invalid_dataset_sources, *response.invalid_kernel_sources,
                       *response.invalid_competition_sources, *response.invalid_model_sources]
            if response.error or invalid:
                raise RuntimeError(f"Kaggle rejected submission: {response.error or invalid}")
            kernel = kernel_ref(response.ref, kernel)
            print(f"Submitted {kernel}, version {response.version_number}", flush=True)
            if args.detach:
                print(run_card(kernel, submission, metadata, args.expected_minutes))
                return 0
        status = wait_for_run(api, kernel, args.poll_interval, args.wait_timeout)
        result, report = fetch_result(api, kernel, submission, status, PROJECT_ROOT / "results")
        run_dir = PROJECT_ROOT / "results/runs" / timestamp
        if report.get("artifacts"):
            folder = fetch_artifacts(api, kernel, report, run_dir)
            print(f"Artifacts: {len(report['artifacts'])} files, SHA-256 verified, in {folder}")
        print_report(report, run_dir)
        print(f"Result: {result}")
        return 0 if status == "complete" and report["status"] == "complete" else 1
    except (Exception, KeyboardInterrupt) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
