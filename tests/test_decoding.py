"""Exercise full verification/cache rollback on synthetic models, on Kaggle."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch


class DecoderTests(unittest.TestCase):
    def test_model_cache_rewinds_rejected_suffix(self):
        import torch
        from specadapt.decoding import CachedModel

        class Cache:
            def __init__(self):
                self.ids = []
            def crop(self, size):
                self.ids = self.ids[:size]

        class Model:
            device = "cpu"
            def __call__(self, input_ids, past_key_values=None, **kwargs):
                cache = past_key_values or Cache()
                values = []
                for token in input_ids[0].tolist():
                    cache.ids.append(token)
                    values.append(sum((i + 1) * x for i, x in enumerate(cache.ids)))
                return SimpleNamespace(logits=torch.tensor(values).reshape(1, -1, 1), past_key_values=cache)

        state = CachedModel(Model())
        with patch("torch.cuda.synchronize"):
            for ids in ([1, 2, 3], [1, 2, 4, 5], [1], [1, 6], [1, 6]):
                value = int(state.sync(ids)[0, -1, 0])
                self.assertEqual(value, sum((i + 1) * x for i, x in enumerate(ids)))
                self.assertEqual(state.cache.ids, list(ids))

    def test_lossless_with_rejections_cache_hits_and_length_limit(self):
        import torch
        from specadapt.decoding import generate

        class Cache:
            def __init__(self):
                self.ids = []
            def crop(self, size):
                self.ids = self.ids[:size]

        class Model:
            device = "cpu"
            generation_config = SimpleNamespace(eos_token_id=127)
            def __init__(self, is_draft=False):
                self.is_draft = is_draft
            def __call__(self, input_ids, past_key_values=None, **kwargs):
                cache = past_key_values or Cache()
                ids = input_ids[0].tolist()
                logits = torch.zeros(1, len(ids), 128)
                for i, token in enumerate(ids):
                    # Target alternates a,b. Drafter agrees on a, then rejects.
                    prediction = 98 if token == 97 else 97
                    if self.is_draft and token == 98:
                        prediction = 99
                    logits[0, i, prediction] = 10
                    cache.ids.append(token)
                return SimpleNamespace(logits=logits, past_key_values=cache)

        class Tokenizer:
            eos_token_id = 127
            all_special_ids = [127]
            def encode(self, text, **kwargs):
                return [ord(char) for char in text]
            def decode(self, ids, **kwargs):
                return "".join(chr(token) for token in ids if token != 127)
            def get_vocab(self):
                return {chr(i): i for i in range(128)}
            def apply_chat_template(self, messages, **kwargs):
                return "U:" + messages[0]["content"] + "\nA:"

        tokenizer = Tokenizer()
        with patch("torch.cuda.synchronize"):
            results = {mode: generate(Model(), Model(True), tokenizer, tokenizer, "ab ab ab",
                                      mode=mode, k=3, max_new_tokens=17)
                       for mode in ("target", "baseline", "m1")}
            from specadapt.bridge import FrozenContinuationCache
            wrong_seed = FrozenContinuationCache([[97, 98, 97, 120, 121, 122]])
            seeded = generate(Model(), Model(True), tokenizer, tokenizer, "question",
                              mode="m1", k=3, max_new_tokens=17, seed_cache=wrong_seed)
        self.assertEqual(results["target"]["token_ids"], [97, 98] * 8 + [97])
        for mode in ("baseline", "m1"):
            self.assertEqual(results[mode]["token_ids"], results["target"]["token_ids"])
        self.assertGreater(results["m1"]["cache_rounds"], 0)
        self.assertLess(results["m1"]["rounds"], results["baseline"]["rounds"])
        self.assertEqual(seeded["token_ids"], results["target"]["token_ids"])
        self.assertGreater(seeded["bridge_stats"]["seed_hits"], 0)


if __name__ == "__main__":
    unittest.main()
