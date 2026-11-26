"""Submit a Kaggle run once one of the watched kernels has left the GPU.

Runs as a CPU job. It builds the run's bundle from its own copy of the project with
demo.kaggle_run.build_submission, under the timestamp staged locally and its own commit, and checks
the script and job TOML against the SHA-256 recorded locally. Until it fires, it only reads the status of the watched kernels. The Kaggle key
comes from the attached private credentials dataset (demo.prepare_data.read_secret); it is never
printed or written. With check_only = true it logs in, reads each status once and stops without
submitting anything.

The staged run keeps its local timestamp, so `kaggle_run --resume <staged_timestamp>` collects it.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from demo import kaggle_runtime as runtime
from demo.kaggle_run import build_submission
from demo.prepare_data import read_secret
from training.kaggle.train_pair import submission_provenance, write_json

DONE = {"complete", "error", "cancelled", "canceled"}
MAX_FAILED_POLLS = 10


def kaggle_api(version):
    token = read_secret("KAGGLE_API_TOKEN")
    if not token:
        runtime.abort("No KAGGLE_API_TOKEN: attach the private credentials dataset.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", f"kaggle=={version}"], check=True)
    # Importing the kaggle package authenticates at once, so the key must be in place first.
    os.environ["KAGGLE_API_TOKEN"] = token
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def status_of(api, kernel):
    return getattr(api.kernels_status(kernel).status, "name", "unknown").lower()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(job, provenance):
    for key in ("script", "job"):
        if sha256(job[key]) != job[f"{key}_sha256"]:
            runtime.abort(f"{job[key]} differs from the file staged locally.")
    source, submission = build_submission(Path.cwd(), [job["script"]], job["staged_timestamp"],
                                          job["metadata"]["id"].split("/")[0],
                                          dict(git_sha=provenance["git_sha"], git_dirty=provenance["git_dirty"],
                                               job=job["job"]))
    folder = Path(tempfile.mkdtemp(prefix="staged-"))
    (folder / "submission.py").write_text(source, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps(job["metadata"], indent=2), encoding="utf-8")
    return folder, submission


def run(job, summary, provenance):
    folder, submission = stage(job, provenance)
    summary["staged"] = dict(submission, bundle_bytes=(folder / "submission.py").stat().st_size)
    api = kaggle_api(job["kaggle_version"])
    summary["logged_in"] = True
    failures = 0
    while True:
        try:
            states = {kernel: status_of(api, kernel) for kernel in job["watch"]}
            failures = 0
        except Exception as error:
            failures += 1
            runtime.heartbeat("watch", failed_polls=failures, error=type(error).__name__)
            if failures >= MAX_FAILED_POLLS:
                runtime.abort(f"{failures} status checks in a row failed.")
            time.sleep(job["poll_seconds"])
            continue
        summary.update(polls=summary["polls"] + 1, last_states=states)
        runtime.heartbeat("watch", **{kernel.split("/")[1]: state for kernel, state in states.items()})
        write_json(Path("summary.json"), summary)
        if job["check_only"]:
            return
        freed = [kernel for kernel, state in states.items() if state in DONE]
        if freed:
            response = api.kernels_push(str(folder))
            invalid = [*response.invalid_dataset_sources, *response.invalid_kernel_sources,
                       *response.invalid_competition_sources, *response.invalid_model_sources]
            if response.error or invalid:
                runtime.abort(f"Kaggle rejected the staged submission: {response.error or invalid}")
            summary.update(fired=True, freed_by=freed, kernel=response.ref, version=response.version_number)
            runtime.heartbeat("result", fired=job["metadata"]["id"], freed_by=",".join(freed),
                              staged_timestamp=job["staged_timestamp"])
            return
        time.sleep(job["poll_seconds"])


def main():
    job = runtime.job_config()
    summary = dict(verdict="running", staged_timestamp=job["staged_timestamp"], target=job["metadata"]["id"],
                   watch=job["watch"], check_only=job["check_only"], logged_in=False, polls=0, fired=False)
    write_json(Path("summary.json"), summary)
    try:
        run(job, summary, submission_provenance())
    except Exception as error:
        summary.update(verdict="fail", error=f"{type(error).__name__}: {error}")
        write_json(Path("summary.json"), summary)
        raise
    summary["verdict"] = "pass"
    write_json(Path("summary.json"), summary)


if __name__ == "__main__":
    main()
