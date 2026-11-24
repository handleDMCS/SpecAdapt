"""CPU checks for draft-only training and the oracle join."""

import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demo.oracle import digest
from training.build_training_inputs import join_rows
from demo import kaggle_runtime as runtime
from training.kaggle.train_pair import Progress, attach, checked_merge, grid, run_tiers, select_rows, tier_order

JOBS = Path(__file__).resolve().parents[1] / "training/jobs"


def load_job(name):
    return tomllib.loads((JOBS / f"{name}.toml").read_text())

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from torch import nn
    from specadapt.alignment import IGNORE_LABEL, M2Trainer, adapter_parameters, encode_example, pack_batches

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.projections = nn.ModuleDict({f"{name}_proj": nn.Linear(8, 8, bias=False)
                                              for name in ("q", "k", "v", "o", "gate", "up", "down")})

        def forward(self, input_ids, **kwargs):
            hidden = self.embedding(input_ids).cumsum(dim=1)
            for projection in self.projections.values():
                hidden = hidden + projection(hidden).tanh()
            return SimpleNamespace(last_hidden_state=hidden)

    class TinyDraft(nn.Module):
        device = "cpu"
        config = SimpleNamespace(num_hidden_layers=1)

        def __init__(self):
            super().__init__()
            self.model = TinyBackbone()
            self.lm_head = nn.Linear(8, 16, bias=False)
            self.lm_head.weight = self.model.embedding.weight

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, module):
            self.lm_head = module

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))


class BoundaryTokenizer:
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        if text != "P\n\nabc" or not kwargs["return_offsets_mapping"]:
            raise AssertionError("Tokenize the full prompt and answer together with offsets.")
        # Token 5 straddles the prompt/answer boundary. Token 2 is auto-added by the tokenizer.
        return dict(input_ids=[1, 4, 5, 6, 2], offset_mapping=[(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)])


class TrainingJoinTests(unittest.TestCase):
    def setUp(self):
        self.prompt = dict(prompt_id="x", task="math", prompt="P", prompt_sha256=digest("P"))
        self.oracle = dict(prompt_id="x", task="math", text="abc", input_ids=[7], token_ids=[8, 2],
                           stop_reason="eos", max_new_tokens=4, prompt_sha256=digest("P"))

    def test_join_keeps_prompt_order_and_required_fields(self):
        prompts = {"x": self.prompt, "y": dict(self.prompt, prompt_id="y")}
        oracle = {"y": dict(self.oracle, prompt_id="y"), "x": self.oracle}
        rows = join_rows(prompts, oracle, 2, [2], 4)
        self.assertEqual([row["prompt_id"] for row in rows], ["x", "y"])
        self.assertEqual(rows[0], dict(self.prompt, text="abc", input_ids=[7], token_ids=[8, 2], stop_reason="eos"))

    def test_wrong_prompt_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            join_rows({"x": self.prompt}, {"x": dict(self.oracle, prompt_sha256=digest("wrong"))}, 1, [2], 4)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            join_rows({"x": dict(self.prompt, prompt="wrong")}, {"x": self.oracle}, 1, [2], 4)

    def test_join_rejects_missing_row_even_if_counts_match(self):
        with self.assertRaisesRegex(ValueError, "1 missing oracle rows and 1 extra"):
            join_rows({"x": self.prompt}, {"y": dict(self.oracle, prompt_id="y")}, 1, [2], 4)

    def test_job_grid_and_smoke_rows(self):
        jobs = Path(__file__).resolve().parents[1] / "training/jobs"
        configs = [tomllib.loads((jobs / f"{name}.toml").read_text()) for name in ("p2-smoke", "p2", "p1", "p4")]
        smoke = configs[0]
        fit = list(range(1440))
        validation = [dict(task=task, n=n) for task in ("math", "code", "instruction", "summarization")
                      for n in range(80)]
        subset, val = select_rows(smoke, fit, validation)
        self.assertEqual((len(set(subset)), len(val)), (64, 40))
        self.assertEqual([row["n"] for row in val], list(range(10)) * 4)
        for full in configs[1:]:
            self.assertEqual({key: full[key] for key in ("t1", "t2")}, {key: smoke[key] for key in ("t1", "t2")})
            self.assertEqual([(len(trials), len(epochs)) for trials, epochs in grid(full).values()], [(3, 2), (6, 1)])
        self.assertEqual([(len(trials), len(epochs)) for trials, epochs in grid(smoke).values()], [(1, 1), (1, 1)])


class TierOrderTests(unittest.TestCase):
    def test_jobs_run_lora_first_and_default_keeps_the_old_order(self):
        for name in ("p2-smoke", "p2", "p1", "p4"):
            job = load_job(name)
            self.assertEqual(list(grid(job)), ["T2", "T1"], name)
            del job["tier_order"]
            self.assertEqual(list(grid(job)), ["T1", "T2"], name)
        job = load_job("p2")
        self.assertEqual(list(grid(dict(job, tier_order=["T1"]))), ["T1"])
        for bad in ([], ["T2", "T2"], ["T3"], ["B1", "T2"]):
            with self.assertRaisesRegex(ValueError, "tier_order"):
                tier_order(dict(job, tier_order=bad))

    def test_short_path_is_p2_with_lora_only_and_one_learning_rate(self):
        short, full = load_job("p2-lora"), load_job("p2")
        self.assertEqual((short.pop("tier_order"), short.pop("max_minutes")), (["T2"], 60))
        self.assertEqual(short.pop("t2"), dict(full["t2"], learning_rates=[0.0003]))
        for key in ("tier_order", "max_minutes", "t2"):
            full.pop(key)
        self.assertEqual(short, full)
        self.assertEqual(grid(load_job("p2-lora")), {"T2": ([dict(rank=16, lr=0.0003, epochs=4)], [2, 4])})


@unittest.skipIf(torch is None, "PyTorch is required for adapter and trainer checks")
class ResultLineTests(unittest.TestCase):
    def test_each_tier_reports_its_result_live_before_the_next_tier_trains(self):
        events, lines = [], []
        b1 = 0.2
        per_tier = {"T2": 0.25, "T1": 0.21}
        summary = dict(tiers={"B1": dict(acceptance=b1)})

        def train_and_replay(tier, configs, checkpoints):
            model = TinyDraft()
            attach(model, tier, dict(configs[0], rank=2))
            events.append(("train", tier))
            self.assertEqual(len(adapter_parameters(model)), 14 if tier == "T2" else 3)
            summary["tiers"][tier] = dict(acceptance=per_tier[tier], per_task={
                task: dict(acceptance=per_tier[tier] + offset / 100)
                for offset, task in enumerate(("math", "code", "instruction", "summarization"))})

        def emit(line):
            events.append(("line", line.split("] ", 1)[1].split()[1]))
            lines.append(line)
            return line

        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            os.chdir(folder)
            try:
                Path("artifacts").mkdir()
                progress = Progress()
                with patch.object(runtime, "_emit", side_effect=emit):
                    run_tiers(load_job("p2-smoke"), summary, progress, train_and_replay)
                written = json.loads(Path("summary.json").read_text())
            finally:
                os.chdir(previous)
        self.assertEqual(events, [("train", "T2"), ("line", "tier=T2"), ("train", "T1"), ("line", "tier=T1")])
        self.assertIsNone(progress.state, "The keepalive must not repeat the result line.")
        fields = dict(part.split("=", 1) for part in lines[0].split("] ", 1)[1].split())
        self.assertEqual(fields, dict(phase="result", tier="T2", acceptance="0.25", b1="0.2", gain_pp="5",
                                      math="0.25", code="0.26", instruction="0.27", summarization="0.28"))
        self.assertAlmostEqual(written["t2_gain_pp"], 5)
        self.assertAlmostEqual(written["t1_gain_pp"], 1)


@unittest.skipIf(torch is None, "PyTorch is required for adapter and trainer checks")
class AlignmentTests(unittest.TestCase):
    def example(self, reason="eos"):
        return encode_example(BoundaryTokenizer(), dict(prompt="P", text="abc", stop_reason=reason, prompt_id="x"), 8)

    def test_answer_boundary_and_eos_mask(self):
        eos = self.example()
        length = self.example("length")
        self.assertEqual(eos.input_ids, [1, 4, 5, 6, 2])
        self.assertEqual(eos.labels, [IGNORE_LABEL, IGNORE_LABEL, 5, 6, 2])
        self.assertEqual(length.input_ids, [1, 4, 5, 6])
        self.assertEqual(length.labels, [IGNORE_LABEL, IGNORE_LABEL, 5, 6])

    def test_batched_logits_match_full_forward_at_answer_positions(self):
        torch.manual_seed(42)
        model = TinyDraft()
        trainer = M2Trainer(model, 2, 16, 16000, 42, on_oom=self.fail)
        eos, length = self.example(), self.example("length")
        logits, labels = trainer.logits_and_labels([eos, length])
        expected = torch.cat([model(torch.tensor([example.input_ids])).logits[0, 1:-1]
                              for example in (eos, length)])
        torch.testing.assert_close(logits, expected)
        self.assertEqual(labels.tolist(), [5, 6, 2, 5, 6])
        self.assertEqual(pack_batches([eos, length], 16, 9), [[length], [eos]])

    def assert_trained_merge(self, tier):
        torch.manual_seed(42)
        model = TinyDraft()
        embeddings = model.model.embedding.weight.detach().clone()
        example = self.example()
        cfg = dict(rank=2, lr=0.01, epochs=1)
        adapter = attach(model, tier, cfg)
        parameters = adapter_parameters(model)
        self.assertEqual(len(parameters), 3 if tier == "T1" else 14)
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        trainer = M2Trainer(model, 2, 16, 16000, 42, on_oom=self.fail)
        trainer.fit([example], cfg, on_step=lambda *args, **kw: None, on_epoch=lambda epoch: None)
        self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in parameters.items()))
        agreement = trainer.evaluate([example], lambda *args, **kw: None)
        self.assertEqual(agreement["answer_tokens"], 3)
        self.assertLess(checked_merge(model, adapter, tier, example), 1e-3)
        self.assertEqual(adapter_parameters(model), {})
        torch.testing.assert_close(model.model.embedding.weight, embeddings, rtol=0, atol=0)

    def test_trained_logit_bias_and_delta_merge(self):
        self.assert_trained_merge("T1")

    def test_trained_lora_merges_all_seven_projections(self):
        self.assert_trained_merge("T2")

    def test_oom_halves_limits_once_and_retries_the_step(self):
        model = TinyDraft()
        cfg = dict(rank=2, lr=0.01, epochs=1)
        attach(model, "T1", cfg)
        retried = []
        trainer = M2Trainer(model, 2, 4, 32, 42, on_oom=lambda *limits: retried.append(limits))
        example = self.example()
        original = trainer.logits_and_labels
        with patch.object(trainer, "logits_and_labels",
                          side_effect=[torch.cuda.OutOfMemoryError(), original([example])]) as forward:
            trainer.fit([example], cfg, on_step=lambda *args, **kw: None, on_epoch=lambda epoch: None)
            self.assertEqual(forward.call_count, 2)
        self.assertEqual(retried, [(2, 16)])
        with patch.object(trainer, "logits_and_labels", side_effect=torch.cuda.OutOfMemoryError):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                trainer.fit([example], cfg, on_step=lambda *args, **kw: None, on_epoch=lambda epoch: None)
        self.assertEqual(retried, [(2, 16)])
