"""P1 (llama_68m -> vicuna_7b) through the vocabulary bridge, with the pinned tokenizers.

P1's drafter-only arms share one tokenizer, but its M1 arms go through the bridge, which P1 had
never run. Two recorded validation rows are replayed in mode m1 against their oracle, at fixed K = 3
and under a policy at k_max = 8 with cache_k = 8, and must reproduce the recorded sequence. The
drafter is a stand-in that mostly proposes the recorded text, so both accepted and rejected
proposals cross the bridge. Downloads only tokenizer files (Kaggle has internet).
"""

import importlib.util
from pathlib import Path
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    import transformers  # noqa: F401
    import huggingface_hub  # noqa: F401
except ImportError:
    torch = None


def fixture():
    spec = importlib.util.spec_from_file_location("p1_val_rows", Path(__file__).with_name("p1_val_rows.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_tokenizer(key):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    plan = {entry["key"]: entry for entry in tomllib.loads((ROOT / "asset_plan.toml").read_text())["models"]}
    spec = plan[key]
    files = [name for name in spec["files"] if name.startswith("tokenizer") or name in
             {"config.json", "special_tokens_map.json"}]
    folder = snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=files)
    return AutoTokenizer.from_pretrained(folder, local_files_only=True)


class Cache:
    def __init__(self):
        self.ids = []

    def crop(self, size):
        self.ids = self.ids[:size]


class TeacherDrafter:
    """Proposes the next drafter token of the recorded text; every seventh call proposes a wrong one."""

    device = "cpu"

    def __init__(self, tokenizer, prompt, text):
        from specadapt.decoding import draft_prompt

        # The drafter's own encoding of prompt + recorded text, as generate() builds its context.
        self.full = tokenizer.encode(draft_prompt(prompt) + text, add_special_tokens=True)
        self.vocab = len(tokenizer)
        self.wrong = tokenizer.encode("zebra", add_special_tokens=False)[-1]
        self.eos = tokenizer.eos_token_id
        self.calls = 0

    def next_token(self, ids):
        self.calls += 1
        if self.calls % 7 == 0:
            return self.wrong
        if ids == self.full[:len(ids)]:
            return self.full[len(ids)] if len(ids) < len(self.full) else self.eos
        # A re-encoded boundary that splits differently: propose something wrong, the oracle rejects it.
        return self.wrong

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        cache = past_key_values or Cache()
        cache.ids.extend(input_ids[0].tolist())
        logits = torch.zeros(1, input_ids.shape[1], self.vocab)
        logits[0, -1, self.next_token(cache.ids)] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


@unittest.skipIf(torch is None, "torch, transformers and huggingface_hub are on the Kaggle image")
class P1BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.draft = pinned_tokenizer("llama_68m")
        cls.target = pinned_tokenizer("vicuna_7b")

    def test_recorded_target_prompt_and_tokenizer(self):
        from demo.oracle import tokenizer_digest
        from specadapt.decoding import target_prompt

        import transformers

        # The fingerprint includes the serialized backend, which other transformers versions write differently.
        if transformers.__version__ == "4.57.1":
            self.assertEqual(tokenizer_digest(self.target), self.data.TOKENIZER_SHA256)
        for row in self.data.ROWS:
            ids = self.target.encode(target_prompt(self.target, row["prompt"]), add_special_tokens=False)
            self.assertEqual(ids, row["input_ids"], row["prompt_id"])

    def test_m1_replay_reproduces_the_recorded_sequence(self):
        from specadapt.adaptive_k import EntropyKPolicy
        from specadapt.decoding import generate
        from specadapt.oracle import GreedyOracle

        settings = (dict(k=3), dict(k=8, cache_k=8, policy=EntropyKPolicy(2.0, 8)), dict(k=8, trace=True))
        for row in self.data.ROWS:
            for options in settings:
                oracle = GreedyOracle(row["token_ids"], self.data.EOS_TOKEN_IDS)
                drafter = TeacherDrafter(self.draft, row["prompt"], row["text"])
                with patch("torch.cuda.synchronize"):
                    data = generate(None, drafter, self.target, self.draft, row["prompt"], mode="m1",
                                    max_new_tokens=256, oracle=oracle, **options)
                label = (row["prompt_id"], sorted(options))
                self.assertEqual(tuple(data["token_ids"]), oracle.token_ids, label)
                self.assertGreater(data["accepted"], 0, label)
                self.assertLess(data["accepted"], data["proposed"], label)
                self.assertGreater(data["rounds"] - data["cache_rounds"], 0, label)


class PairJobTests(unittest.TestCase):
    def test_pair_jobs_differ_from_p2_only_in_pair_values(self):
        jobs = {pair: tomllib.loads((ROOT / f"training/jobs/{pair}-k.toml").read_text())
                for pair in ("p1", "p2", "p4")}
        pair_keys = {"pair", "t2_sha256", "cost_ratio", "same_tokenizer", "max_minutes", "gate"}
        for pair in ("p1", "p4"):
            self.assertEqual(set(jobs[pair]), set(jobs["p2"]))
            for key in set(jobs["p2"]) - pair_keys:
                self.assertEqual(jobs[pair][key], jobs["p2"][key], (pair, key))
            self.assertEqual(set(jobs[pair]["gate"]), set(jobs["p2"]["gate"]))
            self.assertRegex(jobs[pair]["t2_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual((jobs["p1"]["pair"], jobs["p1"]["cost_ratio"], jobs["p1"]["same_tokenizer"]),
                         ("P1", 0.0575, True))
        self.assertEqual((jobs["p4"]["pair"], jobs["p4"]["cost_ratio"], jobs["p4"]["same_tokenizer"]),
                         ("P4", 0.2315, False))
        self.assertEqual((jobs["p1"]["gate"]["b1_accepted"], jobs["p1"]["gate"]["b1_proposed"]), (19817, 79588))
        self.assertEqual((jobs["p4"]["gate"]["b1_accepted"], jobs["p4"]["gate"]["b1_proposed"]), (21411, 89483))


if __name__ == "__main__":
    unittest.main()
