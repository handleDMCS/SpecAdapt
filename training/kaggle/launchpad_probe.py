"""Launch pad probe: live heartbeats, a 1 MB binary artifact, and an optional NaN self-abort."""

import hashlib
import json
import os
from pathlib import Path
import time

from demo.kaggle_runtime import heartbeat, job_config


def write_summary(summary):
    # Rewritten every step, so a job that dies late still reports how far it got.
    Path("summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


config = job_config()
steps, interval = config["steps"], config["interval_seconds"]
payload = os.urandom(config["artifact_bytes"])
artifact = Path("artifacts/probe.bin")
artifact.parent.mkdir(exist_ok=True)
artifact.write_bytes(payload)
summary = dict(verdict="running", artifact=artifact.as_posix(), artifact_bytes=len(payload),
               artifact_sha256=hashlib.sha256(payload).hexdigest(), steps_done=0, steps=steps)
write_summary(summary)
heartbeat("probe", 0, steps, interval_seconds=interval)
for step in range(1, steps + 1):
    time.sleep(interval)
    loss = float("nan") if step == config.get("nan_at_step") else 1 / step
    summary["steps_done"] = step
    write_summary(summary)
    heartbeat("probe", step, steps, loss=loss)
summary["verdict"] = "pass"
write_summary(summary)
