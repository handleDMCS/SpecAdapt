"""Oracle correctness and provenance checks; execute on Kaggle."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demo.extract_oracle import extract
from demo.oracle import digest, dumps, loads, load_for_prompts, make_contract, make_reference
from specadapt.oracle import GreedyOracle


class Tokenizer:
    eos_token_id = 127
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = "synthetic"
    backend_tokenizer = SimpleNamespace(to_str=lambda: "synthetic-backend")

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return "U:" + messages[0]["content"] + " A:"


class OracleTests(unittest.TestCase):
    def test_prefix_and_horizon_guards(self):
        oracle = GreedyOracle([41, 72, 19, 8], [127])
        self.assertEqual(oracle.predictions([], [41, 99, 20]), (41, 72, 19, 8))
        self.assertEqual(oracle.predictions([41, 72], [19]), (19, 8))
        with self.assertRaisesRegex(ValueError, "prefix"):
            oracle.predictions([99], [])
        with self.assertRaisesRegex(ValueError, "truncated"):
            oracle.validate_limit(5)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            oracle.predictions([41, 72, 19, 8], [])
        GreedyOracle([41, 127], [127]).validate_limit(512)
        for tokens in ([], [41, 127, 19], [True], [-1]):
            with self.assertRaises(ValueError):
                GreedyOracle(tokens, [127])

    def test_shared_commit_loop_handles_eos_bonus_rejection_and_limits(self):
        from specadapt.decoding import generate
        tokenizer = Tokenizer()
        cases = [([41, 127, 99], [41, 127], 10, 2),
                 ([41, 99, 20], [41, 127], 10, 1),
                 ([], [127], 10, 0),
                 ([41, 72, 19], [41, 72], 2, 2)]
        for proposed, reference, limit, accepted in cases:
            with self.subTest(proposed=proposed):
                state = SimpleNamespace(propose=lambda *args: proposed, steps=1, seconds=0.01)
                with (patch("torch.cuda.synchronize"),
                      patch("specadapt.decoding.CachedModel", return_value=state) as cached,
                      patch("specadapt.decoding.text_proposal", side_effect=lambda d, t, ctx, raw, tc, k: raw)):
                    row = generate(None, object(), tokenizer, tokenizer, "prompt", mode="baseline",
                                   max_new_tokens=limit, oracle=GreedyOracle(reference, [127]))
                cached.assert_called_once()  # Only draft state; no target state or calls.
                self.assertEqual(row["token_ids"], reference)
                self.assertEqual(row["accepted"], accepted)
                self.assertEqual(row["rounds"], 1)
                from demo.benchmark import summarize
                summary = summarize([row])
                for field in ("tokens_per_second", "speedup_vs_target", "exact_matches", "seconds"):
                    self.assertNotIn(field, summary)


class ReferenceFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer = Tokenizer()
        plan = {"models": [{"key": "qwen_7b", "revision": "target-rev"}],
                "datasets": [{"key": "gsm8k", "revision": "dataset-rev"}]}
        self.contract = make_contract(plan, self.tokenizer, [127], {"torch": "test"}, "T4")
        record = make_reference(7, "question", self.tokenizer, [41, 72, 19], 3, [127])
        self.document = dict(schema_version=1, contract=self.contract, source={"source_hash": "test"}, references=[record])
        self.path = self.root / "references.toml"
        self.path.write_text(dumps(self.document), encoding="utf-8")

    def test_round_trip_and_shorter_replay(self):
        self.assertEqual(loads(self.path.read_text()), self.document)
        oracles, fingerprint = load_for_prompts(self.path, self.contract, {7: "question"}, self.tokenizer, 2)
        self.assertEqual(oracles[7].token_ids, (41, 72, 19))
        self.assertEqual(fingerprint, digest(self.document))
        self.assertNotIn("k", self.contract)
        self.assertNotIn("draft", self.contract)

    def test_provenance_prompt_and_horizon_mismatches_fail(self):
        for key in ("target", "dataset", "versions", "tokenizer_sha256", "gpu", "eos_token_ids"):
            changed = deepcopy(self.contract)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "provenance"):
                load_for_prompts(self.path, changed, {7: "question"}, self.tokenizer, 3)
        with self.assertRaisesRegex(ValueError, "prompt/tokenization"):
            load_for_prompts(self.path, self.contract, {7: "different"}, self.tokenizer, 3)
        with self.assertRaisesRegex(ValueError, "no reference"):
            load_for_prompts(self.path, self.contract, {8: "question"}, self.tokenizer, 3)
        with self.assertRaisesRegex(ValueError, "truncated"):
            load_for_prompts(self.path, self.contract, {7: "question"}, self.tokenizer, 4)
        with patch.object(self.tokenizer, "encode", return_value=[1, 2]):
            with self.assertRaisesRegex(ValueError, "prompt/tokenization"):
                load_for_prompts(self.path, self.contract, {7: "question"}, self.tokenizer, 3)

    def test_damage_duplicates_and_bad_termination_fail(self):
        text = self.path.read_text().replace("token_ids = [41, 72, 19]", "token_ids = [41, 72, 20]")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            loads(text)
        changed = deepcopy(self.document)
        changed["references"].append(changed["references"][0])
        with self.assertRaisesRegex(ValueError, "unique"):
            dumps(changed)
        changed = deepcopy(self.document)
        changed["references"][0]["stop_reason"] = "eos"
        with self.assertRaisesRegex(ValueError, "termination"):
            dumps(changed)

    def test_extract_json_artifact_and_reject_failed_legacy_or_duplicate_results(self):
        source = self.root / "submission-test.json"
        output = self.root / "extracted.toml"
        report = {"status": "complete", "outputs": {"demo_results/target_references.toml": dumps(self.document)}}
        source.write_text(json.dumps(report), encoding="utf-8")
        extract(source, output)
        self.assertEqual(loads(output.read_text()), self.document)
        with self.assertRaises(FileExistsError):
            extract(source, output)
        for invalid in ({"status": "failed", "outputs": report["outputs"]},
                        {"status": "complete", "outputs": {"demo_results/m1.jsonl": "old reference"}},
                        {"status": "complete", "outputs": {**report["outputs"], "target_references.toml": dumps(self.document)}}):
            source.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                extract(source, self.root / "invalid.toml")
