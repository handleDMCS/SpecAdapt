"""Run bundled scripts together inside Kaggle and publish one JSON result."""

import base64
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import math
import tokenize
import tomllib
import traceback
import types

RESULT_SCHEMA_VERSION = 2
# Files a script writes here are kept as Kaggle output files, not embedded in the result JSON.
ARTIFACTS_DIR = "artifacts"
JOB_ENVIRONMENT = "SPECADAPT_JOB"
# The kernel's real stdout while capture_output() redirects fd 1 into the run log.
_live_fd = None
_phase_started = {}
# Set from the job TOML's max_minutes; every heartbeat after it aborts the job.
_deadline = None


class JobAborted(RuntimeError):
    """A job stopped itself on a failed check; the reason is the message."""


def job_config() -> dict:
    """The TOML passed with kaggle_run --job, parsed."""
    path = os.environ.get(JOB_ENVIRONMENT)
    if not path:
        raise RuntimeError("No job TOML: submit with kaggle_run --job training/jobs/<job>.toml.")
    with open(path, "rb") as stream:
        return tomllib.load(stream)


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_duration(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def heartbeat(phase, step=None, total=None, **metrics):
    """Print one progress line to the run log and, live, to the kernel's own stdout.

    Call heartbeat(phase, 0, total) when a phase starts; the ETA is measured from the
    first call for each phase. The job aborts on a NaN or infinite float metric, and on
    any heartbeat after the job TOML's max_minutes.
    """
    now = time.monotonic()
    started = _phase_started.setdefault(phase, now)
    parts = [f"[heartbeat {datetime.now(timezone.utc):%H:%M:%S}Z]", f"phase={phase}"]
    if step is not None:
        parts.append(f"step={step}" + (f"/{total}" if total else ""))
        if total and step > 0:
            parts.append(f"eta={_format_duration((now - started) * (total - step) / step)}")
    parts.extend(f"{key}={_format_value(value)}" for key, value in metrics.items())
    line = _emit(" ".join(parts))
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            abort(f"{key} is {value} in phase {phase}")
    if _deadline is not None and now > _deadline:
        abort("max_minutes from the job TOML exceeded")
    return line


def _emit(line):
    print(line, flush=True)
    if _live_fd is not None:
        os.write(_live_fd, (line + "\n").encode("utf-8", "backslashreplace"))
    return line


def abort(reason):
    """Report a failed self-check live, then stop the job with a failed status."""
    _emit(f"[heartbeat {datetime.now(timezone.utc):%H:%M:%S}Z] phase=abort reason={reason}")
    raise JobAborted(reason)


@contextmanager
def capture_output(path: Path):
    """Capture Python output and inherited stdout/stderr from subprocesses."""
    global _live_fd
    sys.stdout.flush()
    sys.stderr.flush()
    with path.open("w", encoding="utf-8", buffering=1) as log:
        saved_fds = [os.dup(fd) for fd in (1, 2)]
        try:
            for fd in (1, 2):
                os.dup2(log.fileno(), fd)
            _live_fd = saved_fds[0]
            with redirect_stdout(log), redirect_stderr(log):
                yield
        finally:
            _live_fd = None
            log.flush()
            for fd, saved in zip((1, 2), saved_fds):
                os.dup2(saved, fd)
                os.close(saved)


def execute_scripts(project_root: Path, scripts: list[str], namespace: dict | None = None) -> dict:
    """Keep one notebook-like namespace and stop on the first failing script."""
    namespace = {} if namespace is None else namespace
    caller_namespace = namespace
    original_main = sys.modules.get("__main__")
    main_module = types.ModuleType("__main__")
    main_module.__dict__.update(namespace)
    namespace = main_module.__dict__
    sys.modules["__main__"] = main_module
    original_argv, original_path = sys.argv[:], sys.path[:]
    try:
        for filename in scripts:
            script = project_root / filename
            print(f"\n===== {filename} =====", flush=True)
            namespace.update(__name__="__main__", __file__=str(script), __package__=None)
            sys.argv = [str(script)]
            sys.path[:] = [str(script.parent), str(project_root), str(project_root / "src"), *original_path]
            with tokenize.open(script) as stream:
                source = stream.read()
            try:
                exec(compile(source, str(script), "exec"), namespace)
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
    finally:
        sys.argv[:], sys.path[:] = original_argv, original_path
        caller_namespace.update(namespace)
        if original_main is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = original_main
    return namespace


def file_hash(path: Path) -> bytes:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def structure_artifact(path: Path, content: str) -> object:
    """Embed JSON documents and JSON Lines records as native JSON values."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        return content

    text = content.removeprefix("\ufeff")
    try:
        if suffix == ".json":
            value = json.loads(text)
        else:
            value = [json.loads(line) for line in text.split("\n") if line.strip()]
        # Keep artifacts with non-finite numbers or invalid Unicode as text.
        # They must not prevent the containing report from being written.
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
        return value
    except (ValueError, RecursionError):
        return content


def artifact_content(path: Path) -> object:
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"encoding": "base64", "data": base64.b64encode(content).decode("ascii")}
    return structure_artifact(path, text)


def write_result(path: Path, report: dict) -> dict:
    """Save a report in the standard readable format, upgrading legacy artifacts once."""
    version = report.get("schema_version", 1)
    report = {"schema_version": version, **report}
    if version == 1:
        report["schema_version"] = RESULT_SCHEMA_VERSION
        report["outputs"] = {
            name: structure_artifact(Path(name), value) if isinstance(value, str) else value
            for name, value in report["outputs"].items()
        }
    report["run_log"] = report.pop("run_log")
    # Replace the previous file only after successfully writing the whole report.
    with tempfile.TemporaryDirectory(prefix=".specadapt-result-", dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pending.replace(path)
    return report


def run_submission(project_root: Path, submission: dict, output_root: Path | None = None) -> Path:
    output_root = Path("/kaggle/working") if output_root is None else output_root
    output_root.mkdir(parents=True, exist_ok=True)
    result = output_root / f"submission-{submission['timestamp']}.json"
    # Relative file writes go to a temporary work directory, not the model cache
    # or /kaggle/working. The only published file is the final JSON result.
    with tempfile.TemporaryDirectory(prefix="specadapt-work-") as temporary:
        work = Path(temporary) / "project"
        shutil.copytree(project_root, work)
        original_files = {path.relative_to(work): file_hash(path)
                          for path in work.rglob("*") if path.is_file()}
        log = Path(temporary) / "run.log"
        existing_outputs = set(output_root.rglob("*"))
        artifacts = output_root / ARTIFACTS_DIR
        artifacts_existed = artifacts.exists()
        artifacts.mkdir(exist_ok=True)
        existing_artifacts = set(artifacts.rglob("*"))
        # Scripts write to the relative artifacts/ folder. A symlink keeps finished files on
        # disk even if Kaggle kills the session; without symlinks, files are copied at the end.
        local_artifacts = work / ARTIFACTS_DIR
        linked = False
        if not local_artifacts.exists():
            try:
                local_artifacts.symlink_to(artifacts, target_is_directory=True)
                linked = True
            except OSError:
                pass
        original_cwd = Path.cwd()
        original_environment = {key: os.environ.get(key)
                                for key in ("SPECADAPT_PROJECT_ROOT", "SPECADAPT_CACHE_OWNER", JOB_ENVIRONMENT)}
        status, error_message = "complete", None
        started = time.monotonic()
        os.environ["SPECADAPT_PROJECT_ROOT"] = str(work)
        os.environ["SPECADAPT_CACHE_OWNER"] = submission["cache_owner"]
        global _deadline
        try:
            os.chdir(work)
            with capture_output(log):
                print(f"Submission (UTC): {submission['timestamp']}")
                print(f"Scripts in order: {', '.join(submission['scripts'])}")
                try:
                    if submission.get("job"):
                        os.environ[JOB_ENVIRONMENT] = str(work / submission["job"])
                        minutes = job_config().get("max_minutes")
                        if minutes is not None:
                            _deadline = started + 60 * float(minutes)
                    execute_scripts(work, submission["scripts"])
                except BaseException as error:
                    status = "failed"
                    error_message = f"{type(error).__name__}: {error}"
                    traceback.print_exc()
                print(f"\nSubmission status: {status}", flush=True)
        finally:
            _deadline = None
            _phase_started.clear()
            os.chdir(original_cwd)
            for key, value in original_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        summary = work / "summary.json"
        if status != "complete" and (submission.get("job") or summary.is_file()):
            # A failed job always reports verdict "fail", whatever it managed to write.
            try:
                fields = json.loads(summary.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                fields = {}
            fields = fields if isinstance(fields, dict) else {}
            fields.update(verdict="fail", error=error_message)
            summary.write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = {"schema_version": RESULT_SCHEMA_VERSION, **submission, "status": status, "error": error_message,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        if not linked and local_artifacts.is_dir():
            for path in sorted(local_artifacts.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    destination = artifacts / path.relative_to(local_artifacts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
        listing = {path.relative_to(output_root).as_posix(): {"bytes": path.stat().st_size,
                                                               "sha256": file_hash(path).hex()}
                   for path in sorted(artifacts.rglob("*"))
                   if path.is_file() and not path.is_symlink() and path not in existing_artifacts}
        if listing:
            report["artifacts"] = listing
        elif not artifacts_existed:
            shutil.rmtree(artifacts)
        report["outputs"] = {}
        for artifact in sorted(work.rglob("*")):
            relative = artifact.relative_to(work)
            if (relative.parts[0] != ARTIFACTS_DIR and artifact.is_file() and not artifact.is_symlink()
                    and "__pycache__" not in relative.parts
                    and (relative not in original_files or original_files[relative] != file_hash(artifact))):
                report["outputs"][relative.as_posix()] = artifact_content(artifact)
        new_outputs = {path for path in set(output_root.rglob("*")) - existing_outputs - {result}
                       if path != artifacts and not path.is_relative_to(artifacts)}
        for artifact in sorted(new_outputs):
            if artifact.is_file() and not artifact.is_symlink():
                report["outputs"][f"kaggle_working/{artifact.relative_to(output_root).as_posix()}"] = artifact_content(artifact)
        report["run_log"] = log.read_text(encoding="utf-8", errors="replace")
        # Store metadata outside the scripts' work directory to avoid collisions.
        write_result(result, report)
        # Fold explicit /kaggle/working writes into the same JSON as relative
        # writes, removing only files created by this submission.
        for artifact in sorted(new_outputs, key=lambda path: len(path.parts), reverse=True):
            if artifact.is_symlink() or artifact.is_file():
                artifact.unlink()
            elif artifact.is_dir():
                artifact.rmdir()
    # Return normally even after a user-script failure so Kaggle publishes the
    # diagnostics. The laptop reads the result JSON and returns a nonzero exit.
    print(f"Result JSON: {result}")
    return result
