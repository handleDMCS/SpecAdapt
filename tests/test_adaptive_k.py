"""CPU checks for adaptive K: the policies, the text label rule, lossless replay and the k_policy job."""

import gzip
import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specadapt.adaptive_k import (
    FEATURES, EntropyKPolicy, LogisticKPolicy, auc, load_policy, percentile, save_policy, text_labels,
)

JOB = Path(__file__).resolve().parents[1] / "training/jobs/p2-k.toml"
EOS = 127
SURE, UNSURE = "sure", "unsure"
# Next character after the last one: (prediction, confidence). T2 is right more often than B1.
T2_TABLE = {"a": ("b", SURE), "b": ("c", SURE), "c": ("a", UNSURE)}
B1_TABLE = {"a": ("b", SURE), "b": ("c", UNSURE), "c": ("x", UNSURE)}

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None


class Cache:
    def __init__(self):
        self.ids = []

    def crop(self, size):
        self.ids = self.ids[:size]


class CharDrafter:
    """Predicts the next character from the last one; confidence sets the entropy."""

    device = "cpu"

    def __init__(self, table):
        self.table = table

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        cache = past_key_values or Cache()
        ids = input_ids[0].tolist()
        logits = torch.zeros(1, len(ids), 128)
        for i, token in enumerate(ids):
            prediction, confidence = self.table.get(chr(token), ("a", UNSURE))
            if confidence == SURE:
                logits[0, i, ord(prediction)] = 10.0
            else:
                logits[0, i, ord(prediction)] = 2.0
                logits[0, i, ord("y")] = 1.8
            cache.ids.append(token)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class CharTokenizer:
    eos_token_id = EOS
    all_special_ids = [EOS]

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids if token != EOS)

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def apply_chat_template(self, messages, **kwargs):
        return "U:" + messages[0]["content"] + "\nA:"


class PairTokenizer(CharTokenizer):
    """A different vocabulary: two characters per token, greedy from the left."""

    def encode(self, text, **kwargs):
        ids = []
        for start in range(0, len(text), 2):
            piece = text[start:start + 2]
            ids.append(ord(piece) if len(piece) == 1 else 128 + 128 * ord(piece[0]) + ord(piece[1]))
        return ids

    def decode(self, ids, **kwargs):
        text = []
        for token in ids:
            if token == EOS:
                continue
            text.append(chr(token) if token < 128 else chr((token - 128) // 128) + chr((token - 128) % 128))
        return "".join(text)


def oracle_for(tokenizer, text):
    from specadapt.oracle import GreedyOracle

    return GreedyOracle(tokenizer.encode(text) + [EOS], [EOS])


TEXTS = ("abcabcabdabcabcab", "abcabdabdabcabcabcabcab", "cabcabcabcabdab", "bcabcabcxabcabcabd")


def replay(drafter, target_tokenizer, text, **options):
    from specadapt.decoding import generate

    oracle = oracle_for(target_tokenizer, text)
    with patch("torch.cuda.synchronize"):
        data = generate(None, drafter, target_tokenizer, CharTokenizer(), "q " + text[:3], oracle=oracle,
                        max_new_tokens=256, **options)
    return data, oracle


class PolicyTests(unittest.TestCase):
    def test_entropy_policy_stops_on_high_entropy_and_at_k_max(self):
        policy = EntropyKPolicy(1.0, k_max=3)
        self.assertTrue(policy.should_continue(dict(entropy=0.5), 1))
        self.assertTrue(policy.should_continue(dict(entropy=1.0), 2))
        self.assertFalse(policy.should_continue(dict(entropy=1.01), 1))
        self.assertFalse(policy.should_continue(dict(entropy=0.0), 3))
        with self.assertRaises(ValueError):
            EntropyKPolicy(float("nan"))

    def test_logistic_policy_scores_and_round_trips_as_json(self):
        weights = dict(features=list(FEATURES), mean=[0, 0, 0], scale=[1, 1, 1], intercept=-5.0, coef=[0, 10, 0])
        policy = LogisticKPolicy(0.5, 8, weights)
        sure = dict(entropy=0.1, top1_prob=0.99, top1_top2_margin=0.98)
        unsure = dict(entropy=4.0, top1_prob=0.1, top1_top2_margin=0.01)
        self.assertGreater(policy.score(sure), 0.99)
        self.assertLess(policy.score(unsure), 0.02)
        self.assertTrue(policy.should_continue(sure, 7))
        self.assertFalse(policy.should_continue(sure, 8))
        self.assertFalse(policy.should_continue(unsure, 1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            save_policy(policy, path)
            loaded = load_policy(path)
            self.assertEqual((loaded.threshold, loaded.k_max, loaded.weights), (0.5, 8, weights))
            save_policy(EntropyKPolicy(2.5, 6), path)
            self.assertEqual(load_policy(path).describe(), dict(kind="entropy", threshold=2.5, k_max=6))
        with self.assertRaises(ValueError):
            LogisticKPolicy(0.5, 8, dict(weights, features=["entropy"]))

    @unittest.skipIf(numpy is None, "numpy is on the Kaggle image")
    def test_fit_learns_that_confident_tokens_are_accepted(self):
        rows, labels = [], []
        for i in range(400):
            p = (i % 100) / 100
            rows.append(dict(entropy=3 * (1 - p), top1_prob=p, top1_top2_margin=p / 2))
            labels.append(int(p > 0.3 if i % 7 else p < 0.3))  # mostly separable, some noise
        weights = LogisticKPolicy.fit(rows, labels, l2=1.0)
        self.assertEqual(weights["features"], list(FEATURES))
        self.assertEqual(weights["samples"], 400)
        policy = LogisticKPolicy(0.5, 8, weights)
        scores = [policy.score(row) for row in rows]
        self.assertGreater(auc(scores, labels), 0.8)
        self.assertGreater(policy.score(dict(entropy=0.1, top1_prob=0.95, top1_top2_margin=0.5)),
                           policy.score(dict(entropy=2.9, top1_prob=0.05, top1_top2_margin=0.02)))
        json.dumps(weights, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "both"):
            LogisticKPolicy.fit(rows[:3], [1, 1, 1])

    def test_auc_and_percentile(self):
        self.assertEqual(auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertEqual(auc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)
        self.assertIsNone(auc([1, 2], [1, 1]))
        self.assertEqual(percentile([4, 1, 3, 2, 5], 50), 3)
        self.assertAlmostEqual(percentile([0, 10], 30), 3.0)

    def test_text_label_rule(self):
        # Same tokenizer: one character per token, two tokens accepted.
        self.assertEqual(text_labels([1, 2, 3, 4], 2), [1, 1, 0])
        # Cross tokenizer: drafted "a", "ab", "abc", "abcd"; one two-character target token accepted.
        self.assertEqual(text_labels([1, 2, 3, 4], 2), [1, 1, 0])
        # A drafted token whose text ends inside the accepted text is accepted.
        self.assertEqual(text_labels([3, 5, 9], 6), [1, 1, 0])
        self.assertEqual(text_labels([3, 5], 9), [1, 1])
        self.assertEqual(text_labels([2, 4], 0), [0])
        # The whole (cut) proposal was accepted: text beyond it was never verified, so no label.
        self.assertEqual(text_labels([2, 4, 6], 4, verified_length=4), [1, 1])
        self.assertEqual(text_labels([2, 4, 6], 2, verified_length=4), [1, 0])


@unittest.skipIf(torch is None, "torch is on the Kaggle image")
class AdaptiveDecodingTests(unittest.TestCase):
    def test_defaults_are_unchanged_and_carry_no_trace(self):
        from specadapt.decoding import generate

        target = PairTokenizer()
        for mode in ("baseline", "m1"):
            implicit, _ = replay(CharDrafter(T2_TABLE), target, TEXTS[0], mode=mode, k=3)
            explicit, _ = replay(CharDrafter(T2_TABLE), target, TEXTS[0], mode=mode, k=3, policy=None,
                                 cache_k=None, trace=False)
            self.assertNotIn("trace", implicit)
            for key in ("token_ids", "rounds", "accepted", "proposed", "draft_steps", "cache_rounds"):
                self.assertEqual(implicit[key], explicit[key], (mode, key))
            self.assertEqual(sum(hit["rounds"] for hit in implicit["cache_by_match"].values()),
                             implicit["cache_rounds"])
        with self.assertRaisesRegex(ValueError, "k_max"):
            replay(CharDrafter(T2_TABLE), target, TEXTS[0], mode="baseline", k=3, policy=EntropyKPolicy(1.0, 8))
        with self.assertRaisesRegex(ValueError, "cache_k"):
            replay(CharDrafter(T2_TABLE), target, TEXTS[0], mode="m1", k=3, cache_k=0)
        self.assertTrue(callable(generate))

    def check_policy_rounds(self, data, keeps, k_max):
        drafted = [record["drafted"] for record in data["trace"] if record["source"] == "draft"]
        self.assertGreater(len(set(drafted)), 1, "the policy should vary K")
        self.assertLessEqual(max(drafted), k_max)
        self.assertGreaterEqual(min(drafted), 1)
        for record in data["trace"]:
            if record["source"] != "draft":
                continue
            tokens = record["tokens"]
            # Labels stop at the first rejection, so check the stop rule on the labeled prefix.
            for token in tokens[:-1]:
                self.assertTrue(keeps(token))
            if len(tokens) == record["drafted"] and record["drafted"] < k_max and tokens:
                self.assertFalse(keeps(tokens[-1]), record)

    def test_each_policy_varies_k_and_stops_as_specified(self):
        target = PairTokenizer()
        entropy = EntropyKPolicy(1.0, 8)
        weights = dict(features=list(FEATURES), mean=[0, 0, 0], scale=[1, 1, 1], intercept=-5.0, coef=[0, 10, 0])
        logistic = LogisticKPolicy(0.5, 8, weights)
        runs = {}
        for name, policy in (("entropy", entropy), ("logistic", logistic)):
            data, _ = replay(CharDrafter(T2_TABLE), target, TEXTS[1], mode="baseline", k=8, policy=policy,
                             trace=True)
            self.check_policy_rounds(data, lambda token, policy=policy: policy.keeps(policy.score(token)), 8)
            runs[name] = data
        # Both policies separate the same confident and unsure tokens here.
        self.assertEqual(runs["entropy"]["draft_tokens"], runs["logistic"]["draft_tokens"])
        never, _ = replay(CharDrafter(T2_TABLE), target, TEXTS[1], mode="baseline", k=8,
                          policy=EntropyKPolicy(1e9, 8))
        fixed, _ = replay(CharDrafter(T2_TABLE), target, TEXTS[1], mode="baseline", k=8, trace=True)
        for key in ("token_ids", "rounds", "accepted", "proposed", "draft_steps", "draft_tokens"):
            self.assertEqual(never[key], fixed[key], key)
        self.assertLess(runs["entropy"]["draft_steps"], fixed["draft_steps"])

    def test_replay_stays_lossless_under_both_policies(self):
        weights = dict(features=list(FEATURES), mean=[0, 0, 0], scale=[1, 1, 1], intercept=-5.0, coef=[0, 10, 0])
        policies = (EntropyKPolicy(1.0, 8), LogisticKPolicy(0.5, 8, weights), EntropyKPolicy(0.0, 8))
        for tokenizer, same in ((PairTokenizer(), False), (CharTokenizer(), True)):
            for table in (T2_TABLE, B1_TABLE):
                for text in TEXTS:
                    for mode in ("baseline", "m1"):
                        for policy in policies:
                            for cache_k in (None, 3):
                                data, oracle = replay(CharDrafter(table), tokenizer, text, mode=mode, k=8,
                                                      policy=policy, cache_k=cache_k, same_tokenizer=same)
                                self.assertEqual(tuple(data["token_ids"]), oracle.token_ids,
                                                 (same, text, mode, policy.describe()))

    def test_labels_on_a_same_tokenizer_case(self):
        data, _ = replay(CharDrafter(T2_TABLE), CharTokenizer(), TEXTS[1], mode="baseline", k=8, trace=True,
                         same_tokenizer=True)
        seen_rejection = False
        for record in data["trace"]:
            if record["source"] != "draft":
                continue
            labels = [token["label"] for token in record["tokens"]]
            expected = [1] * min(record["accepted"], record["drafted"])
            if record["accepted"] < record["drafted"]:
                expected.append(0)
                seen_rejection = True
            self.assertEqual(labels, expected, record)
            for token in record["tokens"]:
                self.assertEqual(set(token) - {"label"}, set(FEATURES))
                self.assertTrue(0 <= token["top1_top2_margin"] <= token["top1_prob"] <= 1)
        self.assertTrue(seen_rejection)

    def test_labels_on_a_cross_tokenizer_case(self):
        target = PairTokenizer()
        data, _ = replay(CharDrafter(T2_TABLE), target, TEXTS[1], mode="baseline", k=8, trace=True)
        by_text = False
        for record in data["trace"]:
            if record["source"] != "draft":
                continue
            labels = [token["label"] for token in record["tokens"]]
            # One drafted character per label; a target token covers up to two characters.
            accepted_chars = sum(labels)
            self.assertLessEqual(accepted_chars, 2 * record["accepted"])
            self.assertGreaterEqual(accepted_chars, min(record["drafted"], 2 * record["accepted"] - 1))
            if 0 in labels:
                self.assertEqual(labels[-1], 0)
                self.assertEqual(labels.count(0), 1)
            if accepted_chars > record["accepted"]:
                by_text = True
        self.assertTrue(by_text, "labels must count drafted text, not target tokens")

    def test_policy_and_trace_paths_never_cut_a_k_max_proposal(self):
        # A drafter whose every token maps to two target tokens: "ab" pieces under a one-char target.
        from specadapt.decoding import generate

        target = CharTokenizer()
        PairDrafterTokenizer = PairTokenizer

        class PairDrafter:
            device = "cpu"

            def __call__(self, input_ids, past_key_values=None, **kwargs):
                cache = past_key_values or Cache()
                ids = input_ids[0].tolist()
                logits = torch.zeros(1, len(ids), 16512)
                for i, token in enumerate(ids):
                    logits[0, i, 128 + 128 * ord("a") + ord("b")] = 10.0
                    cache.ids.append(token)
                return SimpleNamespace(logits=logits, past_key_values=cache)

        oracle = oracle_for(target, "ab" * 40)
        with patch("torch.cuda.synchronize"):
            cut = generate(None, PairDrafter(), target, PairDrafterTokenizer(), "q", mode="baseline", k=8,
                           oracle=oracle, max_new_tokens=256)
            traced = generate(None, PairDrafter(), target, PairDrafterTokenizer(), "q", mode="baseline", k=8,
                              oracle=oracle, max_new_tokens=256, trace=True)
            policed = generate(None, PairDrafter(), target, PairDrafterTokenizer(), "q", mode="baseline", k=8,
                               oracle=oracle, max_new_tokens=256, policy=EntropyKPolicy(1e9, 8))
        self.assertEqual(max(record["proposed"] for record in traced["trace"]), 16)
        self.assertLess(cut["proposed"] / cut["rounds"], traced["proposed"] / traced["rounds"])
        self.assertEqual(policed["proposed"], traced["proposed"])
        self.assertEqual(tuple(traced["token_ids"]), oracle.token_ids)


def synthetic_rows(target_tokenizer, prefix, count_per_task):
    from specadapt.decoding import target_prompt
    from specadapt.training_inputs import TASKS

    rows = []
    for task_index, task in enumerate(TASKS):
        for n in range(count_per_task):
            text = TEXTS[(task_index + n) % len(TEXTS)] * (1 + n % 2)
            prompt = f"{prefix} {task} {n}: " + text[:3]
            rows.append(dict(prompt_id=f"{prefix}-{task}-{n}", task=task, prompt=prompt,
                             input_ids=target_tokenizer.encode(target_prompt(target_tokenizer, prompt)),
                             token_ids=target_tokenizer.encode(text) + [EOS], text=text, stop_reason="eos"))
    return rows


@unittest.skipIf(torch is None or numpy is None, "torch and numpy are on the Kaggle image")
class KPolicyJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, self.cwd)
        patcher = patch("torch.cuda.synchronize")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.target, self.draft = PairTokenizer(), CharTokenizer()
        self.models = {"B1": CharDrafter(B1_TABLE), "T2": CharDrafter(T2_TABLE)}
        self.fit = synthetic_rows(self.target, "fit", 3)
        self.validation = synthetic_rows(self.target, "val", 2)
        self.record = dict(eos_token_ids=[EOS])

    def job(self):
        job = tomllib.loads(JOB.read_text())
        job.update(fit_prompts_per_task=2, b1_fit_prompts_per_task=1, sweep_prompts_per_task=1,
                   percentiles=[10, 50, 90])
        from training.kaggle.m1_combo import replay_arm, with_oracles
        from training.kaggle.train_pair import Progress

        prompts = with_oracles(self.validation, self.record, self.target)
        with Progress() as progress:
            counts = {}
            for drafter in ("B1", "T2"):
                rows = replay_arm(self.models[drafter], self.draft, self.target, prompts, "gate", progress,
                                  mode="baseline", k=3, same_tokenizer=False)
                counts[drafter] = [sum(row[key] for row in rows) for key in ("accepted", "proposed")]
        job["gate"] = dict(b1_accepted=counts["B1"][0], b1_proposed=counts["B1"][1],
                           t2_acceptance=counts["T2"][0] / counts["T2"][1], t2_tolerance_pp=0.1)
        return job

    def assets(self):
        return self.models, self.draft, self.target, self.record, self.fit, self.validation

    def test_job_toml_is_pair_generic(self):
        job = tomllib.loads(JOB.read_text())
        self.assertEqual((job["pair"], job["cost_ratio"], job["same_tokenizer"]), ("P2", 0.0623, False))
        self.assertEqual((job["k"], job["k_max"], job["cache_k"], job["seed"]), (3, 8, 8, 42))
        self.assertEqual(job["percentiles"], [10, 30, 50, 70, 90])
        self.assertEqual((job["fit_prompts_per_task"], job["sweep_prompts_per_task"]), (180, 20))
        self.assertEqual((job["gate"]["b1_accepted"], job["gate"]["b1_proposed"]), (18316, 98317))
        self.assertRegex(job["t2_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(0 < job["preflight_minutes"] <= 5 < job["max_minutes"])
        source = (Path(__file__).resolve().parents[1] / "training/kaggle/k_policy.py").read_text()
        for specific in ("P2", "0.0623", "fe8bf05e", "18316", "qwen"):
            self.assertNotIn(specific, source)

    def test_end_to_end_on_tiny_models(self):
        from training.kaggle import k_policy
        from training.kaggle.train_pair import Progress

        job = self.job()
        summary = dict(verdict="running", arms={}, sweeps={})
        with Progress() as progress:
            k_policy.run(job, summary, progress, self.assets())
        self.assertEqual(list(summary["arms"]), list(k_policy.ARMS))
        self.assertEqual(summary["gates"]["b1_exact"], True)
        for name, arm in summary["arms"].items():
            for key in ("acceptance", "tokens_per_round", "draft_steps_per_token", "mean_k", "cache_round_share",
                        "modeled_cost_per_token", "cost_vs_b1_k3"):
                self.assertIn(key, arm, name)
            self.assertEqual(arm["prompts"], 8)
            self.assertEqual(set(arm["per_task"]), {"math", "code", "instruction", "summarization"})
            self.assertIn("cost_vs_b1_k3", arm["per_task"]["code"])
            self.assertEqual("cache" in arm, k_policy.ARMS[name][1] == "m1", name)
            self.assertEqual("policy" in arm, k_policy.ARMS[name][2] is not None, name)
        self.assertEqual(summary["arms"]["B1-k3"]["cost_vs_b1_k3"], 1.0)
        self.assertEqual(summary["arms"]["T2-k3"]["mean_k"], 3.0)
        self.assertGreater(summary["arms"]["T2-logistic-m1"]["cache_round_share"], 0)
        table = summary["arms"]["T2-logistic-m1"]["cache"]["by_match_length"]
        self.assertTrue(table and all(set(row) == {"rounds", "proposed", "accepted", "acceptance"}
                                      for row in table.values()))
        fit = summary["policy_fit"]
        for split in ("fit", "val"):
            self.assertIn("logistic_auc", fit[split])
            self.assertIn("label_acceptance", fit[split])
            self.assertEqual(set(fit[split]["per_task"]), {"math", "code", "instruction", "summarization"})
        self.assertIn("label_acceptance_pp", fit["gap"])
        ceiling = summary["ceiling"]
        self.assertLessEqual(ceiling["ceiling_cost_per_token"], ceiling["fixed_k_max_cost_per_token"])
        for key, sweep in summary["sweeps"].items():
            self.assertEqual([point["percentile"] for point in sweep["points"]], [10, 50, 90])
            self.assertEqual(sweep["chosen"]["modeled_cost_per_token"],
                             min(point["modeled_cost_per_token"] for point in sweep["points"]))
            self.assertTrue(Path(f"artifacts/policy-{key}.json").is_file())
        self.assertEqual(summary["arms"]["T2-logistic"]["policy"]["threshold"],
                         summary["sweeps"]["logistic-t2"]["chosen"]["threshold"])
        self.assertIn("logistic_m1_cost_cut_vs_k3_m1", summary["decision"])
        with gzip.open("artifacts/labels-fit.json.gz", "rt", encoding="utf-8") as stream:
            labels = json.load(stream)
        self.assertEqual(len(labels["prompts"]["prompt_id"]), 8)
        self.assertEqual(len(labels["tokens"]["label"]), len(labels["tokens"]["entropy"]))
        self.assertTrue(Path("artifacts/labels-val.json.gz").is_file())
        self.assertTrue(Path("artifacts/logistic-weights.json").is_file())
        self.assertEqual(len(json.loads(Path("artifacts/arm-T2-k3.json").read_text())), 8)
        json.loads(Path("summary.json").read_text())
        # Refit from the saved labels alone, without any model.
        rows = list(zip(*(labels["tokens"][name] for name in FEATURES)))
        refit = LogisticKPolicy.fit(rows, labels["tokens"]["label"], l2=job["l2"])
        for a, b in zip(refit["coef"], fit["weights"]["coef"]):
            self.assertAlmostEqual(a, b, places=3)

    def test_gate_mismatch_aborts_before_labels(self):
        from demo.kaggle_runtime import JobAborted
        from training.kaggle import k_policy
        from training.kaggle.train_pair import Progress

        job = self.job()
        job["gate"]["b1_accepted"] += 1
        summary = dict(verdict="running", arms={}, sweeps={})
        with Progress() as progress, self.assertRaisesRegex(JobAborted, "B1 gate"):
            k_policy.run(job, summary, progress, self.assets())
        self.assertNotIn("labels", summary)
        job = self.job()
        job["gate"]["t2_acceptance"] += 0.01
        with Progress() as progress, self.assertRaisesRegex(JobAborted, "T2 gate"):
            k_policy.run(job, dict(arms={}, sweeps={}), progress, self.assets())


if __name__ == "__main__":
    unittest.main()
