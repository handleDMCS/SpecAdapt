import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from demo import kaggle_run
from demo import prepare_data
from demo.assets import MANIFEST_NAME, asset_files, cache_assets, load_plan, validate_cache, wait_cache_ready
from demo.kaggle_runtime import execute_scripts, run_submission


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()

    def script(self, name, source):
        path = self.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return name

    def test_order_namespace_main_and_future_imports(self):
        self.script("later.py", "from __future__ import annotations\nshared = [__name__]\n")
        self.script("earlier.py", "assert shared == ['__main__']\nshared.append('second')\n")
        namespace = execute_scripts(self.project, ["later.py", "earlier.py"])
        self.assertEqual(namespace["shared"], ["__main__", "second"])

    def test_dataclasses_and_main_imports_share_the_notebook_namespace(self):
        self.script("one.py", "from __future__ import annotations\nfrom dataclasses import dataclass\n"
                    "@dataclass\nclass Item:\n    value: int\nitem = Item(7)\n")
        self.script("two.py", "from __main__ import Item\nassert isinstance(item, Item)\nassert item.value == 7\n")
        execute_scripts(self.project, ["one.py", "two.py"])

    def test_system_exit_zero_continues_but_failure_stops(self):
        self.script("one.py", "raise SystemExit(0)\n")
        self.script("two.py", "reached = True\nraise RuntimeError('expected')\n")
        self.script("three.py", "should_not_run = True\n")
        namespace = {}
        with self.assertRaisesRegex(RuntimeError, "expected"):
            execute_scripts(self.project, ["one.py", "two.py", "three.py"], namespace)
        self.assertTrue(namespace["reached"])
        self.assertNotIn("should_not_run", namespace)

    def test_one_archive_contains_stdout_stderr_subprocess_and_artifacts(self):
        outputs = self.root / "outputs"
        self.script("one.py", "from pathlib import Path\nimport sys, subprocess\n"
                    "print('python stdout')\nprint('python stderr', file=sys.stderr)\n"
                    "subprocess.run([sys.executable, '-c', \"print('child output')\"], check=True)\n"
                    "Path('report.txt').write_text('answer')\n"
                    f"Path({str(outputs / 'explicit.txt')!r}).write_text('absolute')\n")
        result = run_submission(self.project, {"timestamp": "test", "scripts": ["one.py"], "cache_owner": "owner"}, outputs)
        self.assertEqual(list(outputs.iterdir()), [result])
        with zipfile.ZipFile(result) as archive:
            log = archive.read("run.log").decode()
            for text in ("python stdout", "python stderr", "child output"):
                self.assertIn(text, log)
            self.assertEqual(archive.read("outputs/report.txt"), b"answer")
            self.assertEqual(archive.read("outputs/kaggle_working/explicit.txt"), b"absolute")
            self.assertNotIn("outputs/one.py", archive.namelist())

    def test_failure_keeps_diagnostics_and_skips_remaining_scripts(self):
        self.script("bad.py", "raise ValueError('intentional test failure')\n")
        self.script("after.py", "from pathlib import Path\nPath('wrong.txt').touch()\n")
        result = run_submission(self.project, {"timestamp": "failure", "scripts": ["bad.py", "after.py"],
                                               "cache_owner": "owner"}, self.root / "outputs")
        with zipfile.ZipFile(result) as archive:
            self.assertEqual(json.loads(archive.read("submission.json"))["status"], "failed")
            self.assertIn(b"intentional test failure", archive.read("run.log"))
            self.assertNotIn("outputs/wrong.txt", archive.namelist())

    def test_submission_restores_environment_and_working_directory(self):
        self.script("one.py", "pass\n")
        before = {key: os.environ.get(key) for key in ("SPECADAPT_PROJECT_ROOT", "SPECADAPT_CACHE_OWNER")}
        original_cwd = Path.cwd()
        run_submission(self.project, {"timestamp": "cleanup", "scripts": ["one.py"], "cache_owner": "owner"},
                       self.root / "outputs")
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(before, {key: os.environ.get(key) for key in before})

    def test_bundle_validates_paths_and_excludes_credentials_and_environment(self):
        self.script("folder/with spaces.py", "shared = 7\n")
        self.script(".venv/private.py", "secret = 'do not upload'\n")
        (self.project / ".env").write_text("TOKEN=do not upload\n")
        source, submission = kaggle_run.build_submission(self.project, ["folder\\with spaces.py"], "test", "owner")
        self.assertEqual(submission["scripts"], ["folder/with spaces.py"])
        self.assertNotIn("do not upload", source)
        compile(source, "generated.py", "exec")
        for path in ("../outside.py", "/tmp/absolute.py", "C:\\absolute.py", ".venv/private.py"):
            with self.assertRaises(ValueError):
                kaggle_run.build_submission(self.project, [path], "test", "owner")
        self.script("bad.py", "this is not valid python !!!\n")
        with self.assertRaises(SyntaxError):
            kaggle_run.build_submission(self.project, ["bad.py"], "test", "owner")

    def test_result_download_leaves_one_file_and_preserves_failure_status(self):
        submission = {"timestamp": "test", "scripts": []}
        def download(kernel, path, **kwargs):
            with zipfile.ZipFile(Path(path) / "submission-test.zip", "w") as archive:
                archive.writestr("submission.json", json.dumps({**submission, "status": "failed"}))
                archive.writestr("run.log", "traceback")
            (Path(path) / "extra-sdk.log").write_text("platform log")
        output = self.root / "downloaded"
        result, report = kaggle_run.fetch_result(SimpleNamespace(kernels_output=download), "owner/kernel", submission, "complete", output)
        self.assertEqual(list(output.iterdir()), [result])
        self.assertEqual(report["status"], "failed")

    def test_platform_failure_is_packaged_and_stale_success_is_rejected(self):
        def download(kernel, path, **kwargs):
            (Path(path) / "kernel.log").write_text("platform crashed")
        api = SimpleNamespace(kernels_output=download)
        submission = {"timestamp": "test", "scripts": []}
        result, report = kaggle_run.fetch_result(api, "owner/kernel", submission, "error", self.root / "out")
        self.assertEqual(report["status"], "platform_error")
        with zipfile.ZipFile(result) as archive:
            self.assertEqual(archive.read("run.log"), b"platform crashed")
        with self.assertRaisesRegex(RuntimeError, "without this submission"):
            kaggle_run.fetch_result(api, "owner/kernel", submission, "complete", self.root / "other")

    def test_cancelled_status_does_not_poll_forever(self):
        api = SimpleNamespace(kernels_status=lambda _: SimpleNamespace(status=SimpleNamespace(name="CANCELLED")))
        self.assertEqual(kaggle_run.wait_for_run(api, "owner/kernel", 1, 10), "cancelled")

    def test_kaggle_push_references_are_normalized(self):
        for ref in ("owner/kernel", "/code/owner/kernel", "https://www.kaggle.com/code/owner/kernel"):
            self.assertEqual(kaggle_run.kernel_ref(ref, "fallback/kernel"), "owner/kernel")

    def test_recovery_reads_staged_metadata_without_executing_source(self):
        self.script("one.py", "raise RuntimeError('never run during staging')\n")
        source, submission = kaggle_run.build_submission(self.project, ["one.py"], "test", "owner")
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "submission.py").write_text(source)
        self.assertEqual(kaggle_run.staged_submission(stage), submission)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = self.root / "plan.toml"
        self.plan.write_text('[[models]]\nkey="tiny"\nrepo="owner/tiny"\nrevision="' + 'a' * 40 +
                             '"\nfiles=["config.json", "model.safetensors"]\n'
                             '[[models]]\nkey="gated"\nrepo="owner/gated"\nrevision="' + 'b' * 40 +
                             '"\ngated=true\nfiles=["config.json"]\n'
                             '[[datasets]]\nkey="toy"\nrepo="owner/toy"\nrevision="' + 'c' * 40 +
                             '"\nfiles=["default/train/0000.parquet", "default/test/0000.parquet"]\n')
        self.downloads, self.uploads = [], []

    def fake_download(self, directory, asset, token):
        directory.mkdir(parents=True)
        sizes = {}
        for filename in asset_files(asset):
            path = directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"small fake data")
            sizes[filename] = path.stat().st_size
        manifest = {"fingerprint": asset["fingerprint"], "files": sizes}
        if asset["kind"] == "datasets":
            manifest["rows"] = self.fake_rows(directory, asset)
        (directory / MANIFEST_NAME).write_text(json.dumps(manifest))
        self.downloads.append(asset["key"])
        return manifest

    def fake_rows(self, directory, asset):
        return {"toy": {"splits": {"train": 3, "test": 2}, "total": 5}}

    def fake_publisher(self, owner):
        return lambda directory, asset: self.uploads.append(asset["key"])

    def prepare(self, **overrides):
        args = dict(secret_reader=lambda _: "test-token", downloader=self.fake_download,
                    publisher_factory=self.fake_publisher, row_counter=self.fake_rows)
        args.update(overrides)
        return prepare_data.prepare(self.plan, self.root / "input", self.root / "scratch", "owner", **args)

    def test_first_execution_downloads_and_second_session_uses_only_mounted_cache(self):
        first = self.prepare()
        self.assertTrue(first["CACHE_COMPLETE"])
        self.assertEqual(self.downloads, ["tiny", "gated", "datasets"])
        self.assertEqual(self.uploads, self.downloads)
        shutil.copytree(self.root / "scratch", self.root / "input")
        shutil.rmtree(self.root / "scratch")
        def forbidden(*args):
            self.fail("A mounted cache hit must not authenticate, download or upload.")
        second = self.prepare(secret_reader=forbidden, downloader=forbidden, publisher_factory=forbidden)
        self.assertEqual(second["DATASET_ROWS"]["toy"]["total"], 5)
        self.assertTrue(all("input" in path for path in second["MODEL_PATHS"].values()))

    def test_repeated_script_in_one_session_does_not_redownload_or_republish(self):
        self.prepare()
        def forbidden(*args):
            self.fail("Session cache should need no secret, download or upload.")
        self.prepare(secret_reader=forbidden, downloader=forbidden, publisher_factory=forbidden)
        self.assertEqual(len(self.downloads), 3)

    def test_missing_hf_token_defers_only_gated_model(self):
        report = self.prepare(secret_reader=lambda _: None)
        self.assertFalse(report["CACHE_COMPLETE"])
        self.assertEqual(report["PENDING_MODELS"], ["gated"])
        self.assertEqual(self.downloads, ["tiny", "datasets"])
        self.assertEqual(report["DATASET_ROWS"]["toy"]["total"], 5)

    def test_denied_gated_access_does_not_prevent_dataset_preparation(self):
        def download(directory, asset, token):
            if asset["key"] == "gated":
                error = RuntimeError("Access denied")
                error.response = SimpleNamespace(status_code=403)
                raise error
            return self.fake_download(directory, asset, token)
        report = self.prepare(downloader=download)
        self.assertEqual(report["PENDING_MODELS"], ["gated"])
        self.assertEqual(self.downloads, ["tiny", "datasets"])
        self.assertEqual(report["DATASET_ROWS"]["toy"]["total"], 5)

    def test_truncated_cache_fails_without_redownloading(self):
        self.prepare()
        asset = cache_assets(load_plan(self.plan))[0]
        cache = self.root / "scratch" / asset["slug"]
        (cache / "model.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.prepare()
        self.assertEqual(len(self.downloads), 3)

    def test_changed_revision_gets_a_different_cache_name(self):
        plan = load_plan(self.plan)
        before = cache_assets(plan)[0]["slug"]
        plan["models"][0]["revision"] = "d" * 40
        self.assertNotEqual(before, cache_assets(plan)[0]["slug"])

    def test_cache_discovery_uses_all_pages_and_excludes_unrelated_datasets(self):
        shutil.copyfile(self.plan, self.root / "asset_plan.toml")
        assets = cache_assets(load_plan(self.plan))
        ref = f"owner/{assets[0]['slug']}"
        calls = []
        def listing(**kwargs):
            calls.append(kwargs["page"])
            return {1: [SimpleNamespace(ref="owner/unrelated")], 2: [SimpleNamespace(ref=ref)]}.get(kwargs["page"], [])
        api = SimpleNamespace(dataset_list=listing, dataset_status=lambda _: "ready")
        self.assertEqual(kaggle_run.ready_cache_sources(api, self.root, "owner"), [ref])
        self.assertEqual(calls, [1, 2, 3])

    def test_private_cache_visibility_403_is_retried_after_successful_creation(self):
        calls = []
        def status(ref):
            calls.append(ref)
            if len(calls) == 1:
                error = RuntimeError("Not visible yet")
                error.response = SimpleNamespace(status_code=403)
                raise error
            return "ready"
        with patch("demo.assets.time.sleep") as sleep:
            wait_cache_ready(SimpleNamespace(dataset_status=status), "owner/cache")
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once()

    def test_hugging_face_token_environment_alias(self):
        with patch.dict(os.environ, {"HF_TOKEN": "", "HUGGING_FACE_TOKEN": "synthetic-test-value"}):
            self.assertEqual(prepare_data.read_secret("HF_TOKEN"), "synthetic-test-value")

    def test_count_rows_reads_parquet_metadata_without_loading_huggingface(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        asset = cache_assets(load_plan(self.plan))[-1]
        for filename, count in zip(asset_files(asset), (3, 2)):
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(arrow.table({"text": ["row"] * count}), path)
        self.assertEqual(prepare_data.count_dataset_rows(self.root, asset)["toy"]["total"], 5)


if __name__ == "__main__":
    unittest.main()
