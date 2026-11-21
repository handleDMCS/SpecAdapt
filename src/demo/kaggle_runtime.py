"""Run bundled scripts together inside Kaggle and publish one result archive."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tokenize
import traceback
import types
import zipfile


@contextmanager
def capture_output(path: Path):
    """Capture Python output and inherited stdout/stderr from subprocesses."""
    sys.stdout.flush()
    sys.stderr.flush()
    with path.open("w", encoding="utf-8", buffering=1) as log:
        saved_fds = [os.dup(fd) for fd in (1, 2)]
        try:
            for fd in (1, 2):
                os.dup2(log.fileno(), fd)
            with redirect_stdout(log), redirect_stderr(log):
                yield
        finally:
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


def run_submission(project_root: Path, submission: dict, output_root: Path | None = None) -> Path:
    output_root = Path("/kaggle/working") if output_root is None else output_root
    output_root.mkdir(parents=True, exist_ok=True)
    result = output_root / f"submission-{submission['timestamp']}.zip"
    # Relative file writes go to a temporary work directory, not the model cache
    # or /kaggle/working. The only published file is the final archive.
    with tempfile.TemporaryDirectory(prefix="specadapt-work-") as temporary:
        work = Path(temporary) / "project"
        shutil.copytree(project_root, work)
        original_files = {path.relative_to(work): file_hash(path)
                          for path in work.rglob("*") if path.is_file()}
        log = Path(temporary) / "run.log"
        existing_outputs = set(output_root.rglob("*"))
        original_cwd = Path.cwd()
        original_environment = {key: os.environ.get(key)
                                for key in ("SPECADAPT_PROJECT_ROOT", "SPECADAPT_CACHE_OWNER")}
        status, error_message = "complete", None
        started = time.monotonic()
        os.environ["SPECADAPT_PROJECT_ROOT"] = str(work)
        os.environ["SPECADAPT_CACHE_OWNER"] = submission["cache_owner"]
        try:
            os.chdir(work)
            with capture_output(log):
                print(f"Submission (UTC): {submission['timestamp']}")
                print(f"Scripts in order: {', '.join(submission['scripts'])}")
                try:
                    execute_scripts(work, submission["scripts"])
                except BaseException as error:
                    status = "failed"
                    error_message = f"{type(error).__name__}: {error}"
                    traceback.print_exc()
                print(f"\nSubmission status: {status}", flush=True)
        finally:
            os.chdir(original_cwd)
            for key, value in original_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        report = {**submission, "status": status, "error": error_message,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        # Store metadata outside the scripts' work directory to avoid collisions.
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", json.dumps(report, indent=2))
            archive.write(log, "run.log")
            for artifact in sorted(work.rglob("*")):
                relative = artifact.relative_to(work)
                if (artifact.is_file() and not artifact.is_symlink() and "__pycache__" not in relative.parts
                        and (relative not in original_files or original_files[relative] != file_hash(artifact))):
                    archive.write(artifact, f"outputs/{relative.as_posix()}")
            new_outputs = set(output_root.rglob("*")) - existing_outputs - {result}
            for artifact in sorted(new_outputs):
                if artifact.is_file() and not artifact.is_symlink():
                    archive.write(artifact, f"outputs/kaggle_working/{artifact.relative_to(output_root).as_posix()}")
        # Fold explicit /kaggle/working writes into the same archive as relative
        # writes, removing only files created by this submission.
        for artifact in sorted(new_outputs, key=lambda path: len(path.parts), reverse=True):
            if artifact.is_symlink() or artifact.is_file():
                artifact.unlink()
            elif artifact.is_dir():
                artifact.rmdir()
    # Return normally even after a user-script failure so Kaggle publishes the
    # diagnostics. The laptop reads submission.json and returns a nonzero exit.
    print(f"Result archive: {result}")
    return result
