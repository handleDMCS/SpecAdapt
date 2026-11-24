"""Detached submission, live heartbeats and the artifacts channel."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demo import kaggle_run, kaggle_runtime
from demo.kaggle_runtime import heartbeat, run_submission


class HeartbeatTests(unittest.TestCase):
    def test_line_has_phase_progress_eta_and_metrics(self):
        with redirect_stdout(io.StringIO()):
            heartbeat("hb-format", 0, 10)
            line = heartbeat("hb-format", 5, 10, loss=0.123456, note="ok")
        self.assertRegex(line, r"^\[heartbeat \d\d:\d\d:\d\dZ\] phase=hb-format step=5/10 eta=\d+m\d\ds "
                               r"loss=0\.1235 note=ok$")

    def test_line_reaches_the_live_descriptor(self):
        read, write = os.pipe()
        try:
            with patch.object(kaggle_runtime, "_live_fd", write), redirect_stdout(io.StringIO()) as log:
                line = heartbeat("hb-live", acceptance=0.5)
            os.close(write)
            write = None
            with os.fdopen(read, encoding="utf-8") as stream:
                read = None
                self.assertEqual(stream.read(), line + "\n")
            self.assertEqual(log.getvalue(), line + "\n")
        finally:
            for fd in (read, write):
                if fd is not None:
                    os.close(fd)


class ArtifactRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.outputs = self.root / "outputs"

    def run_script(self, source, timestamp="test"):
        (self.project / "job.py").write_text(source, encoding="utf-8")
        result = run_submission(self.project, {"timestamp": timestamp, "scripts": ["job.py"],
                                               "cache_owner": "owner"}, self.outputs)
        return json.loads(result.read_text(encoding="utf-8"))

    def test_artifacts_stay_as_files_and_are_listed_not_embedded(self):
        report = self.run_script(
            "from pathlib import Path\nimport json\n"
            "from demo.kaggle_runtime import heartbeat\n"
            "Path('artifacts/sub').mkdir(parents=True, exist_ok=True)\n"
            "Path('artifacts/weights.bin').write_bytes(bytes(range(256)) * 4)\n"
            "Path('artifacts/sub/table.txt').write_text('row')\n"
            "heartbeat('train', 1, 2, loss=1.5)\n"
            "Path('summary.json').write_text(json.dumps({'verdict': 'pass'}))\n")
        self.assertEqual(report["status"], "complete")
        weights = bytes(range(256)) * 4
        self.assertEqual(report["artifacts"], {
            "artifacts/sub/table.txt": {"bytes": 3, "sha256": hashlib.sha256(b"row").hexdigest()},
            "artifacts/weights.bin": {"bytes": 1024, "sha256": hashlib.sha256(weights).hexdigest()},
        })
        self.assertEqual(report["outputs"], {"summary.json": {"verdict": "pass"}})
        self.assertEqual((self.outputs / "artifacts/weights.bin").read_bytes(), weights)
        self.assertIn("phase=train step=1/2", report["run_log"])

    def test_abort_fails_the_job_and_keeps_finished_artifacts(self):
        report = self.run_script(
            "from pathlib import Path\nfrom demo.kaggle_runtime import abort\n"
            "Path('artifacts').mkdir(exist_ok=True)\n"
            "Path('artifacts/partial.bin').write_bytes(b'done')\n"
            "abort('loss is NaN at step 7')\n"
            "Path('artifacts/never.bin').write_bytes(b'x')\n")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "JobAborted: loss is NaN at step 7")
        self.assertIn("phase=abort reason=loss is NaN at step 7", report["run_log"])
        self.assertEqual(list(report["artifacts"]), ["artifacts/partial.bin"])

    def run_job(self, source, config):
        (self.project / "jobs").mkdir(exist_ok=True)
        (self.project / "jobs/probe.toml").write_text(config, encoding="utf-8")
        (self.project / "job.py").write_text(source, encoding="utf-8")
        result = run_submission(self.project, {"timestamp": "job", "scripts": ["job.py"], "cache_owner": "owner",
                                               "job": "jobs/probe.toml"}, self.outputs)
        return json.loads(result.read_text(encoding="utf-8"))

    def test_job_config_is_readable_and_success_keeps_the_summary(self):
        report = self.run_job("from pathlib import Path\nimport json\nfrom demo.kaggle_runtime import job_config\n"
                              "Path('summary.json').write_text(json.dumps({'verdict': 'pass', **job_config()}))\n",
                              "steps = 3\n")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["outputs"]["summary.json"], {"verdict": "pass", "steps": 3})
        self.assertNotIn("SPECADAPT_JOB", os.environ)

    def test_nan_metric_aborts_and_forces_a_fail_verdict(self):
        report = self.run_job("from pathlib import Path\nimport json\nfrom demo.kaggle_runtime import heartbeat\n"
                              "Path('summary.json').write_text(json.dumps({'verdict': 'pass', 'step': 2}))\n"
                              "heartbeat('train', 2, 4, loss=float('nan'))\n", "steps = 4\n")
        self.assertEqual(report["error"], "JobAborted: loss is nan in phase train")
        self.assertEqual(report["outputs"]["summary.json"],
                         {"verdict": "fail", "step": 2, "error": "JobAborted: loss is nan in phase train"})

    def test_heartbeat_after_max_minutes_aborts(self):
        report = self.run_job("import time\nfrom demo.kaggle_runtime import heartbeat\n"
                              "time.sleep(0.05)\nheartbeat('train', 1, 2)\n", "max_minutes = 0.0001\n")
        self.assertEqual(report["error"], "JobAborted: max_minutes from the job TOML exceeded")
        self.assertEqual(report["outputs"]["summary.json"]["verdict"], "fail")
        # The deadline belongs to that submission only.
        with redirect_stdout(io.StringIO()):
            heartbeat("after", 1, 2)

    def test_no_artifacts_leaves_no_folder_and_no_report_key(self):
        report = self.run_script("print('nothing to keep')\n")
        self.assertNotIn("artifacts", report)
        self.assertEqual([path.name for path in self.outputs.iterdir()], ["submission-test.json"])


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fake_api(self, files):
        def download(kernel, path, **kwargs):
            self.assertEqual(kwargs["file_pattern"], r"^artifacts/")
            for name, content in files.items():
                target = Path(path) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        return SimpleNamespace(kernels_output=download)

    def test_verified_artifacts_are_kept(self):
        report = {"artifacts": {"artifacts/a.bin": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}}
        folder = kaggle_run.fetch_artifacts(self.fake_api({"artifacts/a.bin": b"abc"}), "o/k", report,
                                            self.root / "run")
        self.assertEqual((folder / "a.bin").read_bytes(), b"abc")
        self.assertEqual(sorted(path.name for path in (self.root / "run").iterdir()), ["artifacts"])

    def test_mismatched_or_missing_artifacts_are_rejected(self):
        report = {"artifacts": {"artifacts/a.bin": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}}
        for files in ({"artifacts/a.bin": b"abd"}, {}):
            with self.subTest(files=files), self.assertRaises(RuntimeError):
                kaggle_run.fetch_artifacts(self.fake_api(files), "o/k", report, self.root / "run")
            self.assertEqual(list((self.root / "run").iterdir()), [])

    def test_summary_is_printed_and_saved_with_the_log_tail_on_failure(self):
        report = {"status": "failed", "error": "JobAborted: x", "outputs": {"summary.json": {"verdict": "fail"}},
                  "run_log": "\n".join(f"line {i}" for i in range(50))}
        stream = io.StringIO()
        with redirect_stdout(stream):
            kaggle_run.print_report(report, self.root / "run")
        text = stream.getvalue()
        self.assertIn('"verdict": "fail"', text)
        self.assertIn("line 49", text)
        self.assertIn("line 30", text)
        self.assertNotIn("line 29\n", text)
        self.assertEqual(json.loads((self.root / "run/summary.json").read_text()), {"verdict": "fail"})


class SubmitTests(unittest.TestCase):
    def test_kernel_name_sets_matching_id_and_title(self):
        metadata = kaggle_run.name_kernel({"id": "owner/specAdapt", "title": "specAdapt"}, "p2-smoke")
        self.assertEqual((metadata["id"], metadata["title"]), ("owner/specadapt-p2-smoke", "specadapt-p2-smoke"))
        for name in ("P2", "-p2", "p2_smoke", "a" * 41, ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                kaggle_run.name_kernel({"id": "owner/x"}, name)

    def test_job_file_must_be_a_valid_toml_with_positive_max_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name, text in {"ok.toml": "max_minutes = 20\n", "zero.toml": "max_minutes = 0\n",
                               "text.toml": "max_minutes = '20'\n", "broken.toml": "max_minutes =\n",
                               "job.py": "pass\n"}.items():
                (root / name).write_text(text, encoding="utf-8")
            self.assertEqual(kaggle_run.job_file(root, "ok.toml"), "ok.toml")
            for name in ("zero.toml", "text.toml", "broken.toml", "job.py", "missing.toml"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    kaggle_run.job_file(root, name)

    def test_detach_pushes_prints_the_card_and_does_not_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "job.py").write_text("pass\n", encoding="utf-8")
            (root / "src/demo").mkdir(parents=True)
            (root / "src/demo/kaggle_runtime.py").write_bytes(Path(kaggle_runtime.__file__).read_bytes())
            pushed = []
            response = SimpleNamespace(error=None, invalid_dataset_sources=[], invalid_kernel_sources=[],
                                       invalid_competition_sources=[], invalid_model_sources=[],
                                       ref="https://www.kaggle.com/code/owner/specadapt-probe", version_number=3)
            api = SimpleNamespace(kernels_push=lambda stage: pushed.append(stage) or response)
            stream = io.StringIO()
            with (patch.object(kaggle_run, "PROJECT_ROOT", root),
                  patch.object(kaggle_run, "read_metadata", return_value={"id": "owner/specAdapt"}),
                  patch.object(kaggle_run, "get_api", return_value=api),
                  patch.object(kaggle_run, "wait_for_run") as wait,
                  redirect_stdout(stream)):
                code = kaggle_run.main(["job.py", "--no-cache", "--detach", "--kernel", "probe",
                                        "--expected-minutes", "3"])
            self.assertEqual(code, 0)
            wait.assert_not_called()
            stage = Path(pushed[0])
            metadata = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["id"], "owner/specadapt-probe")
            submission = json.loads((stage / "submission.json").read_text(encoding="utf-8"))
            self.assertIn("git_sha", submission)
            text = stream.getvalue()
            self.assertIn("https://www.kaggle.com/code/owner/specadapt-probe", text)
            self.assertIn(f"--resume {submission['timestamp']}", text)
            self.assertIn("about 3 minutes", text)


if __name__ == "__main__":
    unittest.main()
