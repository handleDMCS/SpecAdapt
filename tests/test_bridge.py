import unittest

from specadapt.bridge import ContinuationCache, FrozenContinuationCache, TokenTrie, VocabularyBridge, text_proposal
from specadapt.decoding import accepted_prefix, draft_prompt, target_prompt


class BridgeTests(unittest.TestCase):
    def test_frozen_cache_votes_and_does_not_join_training_examples(self):
        cache = FrozenContinuationCache([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                                         [1, 2, 3, 7, 8, 9]])
        self.assertEqual(cache.match([1, 2, 3]), ([4, 5, 6], 3))
        self.assertEqual(cache.match([5, 6, 1]), ([], 0))

    def test_trie_merge_and_boundary_keys(self):
        trie = TokenTrie()
        trie.put([10, -1, 20, 21], [99])
        self.assertEqual(trie.get([10, -1, 20, 21]), (99,))
        self.assertIsNone(trie.get([11, -1, 20, 21]))
        self.assertIsNone(trie.get([10, -1, 20]))

    def test_bounded_trie(self):
        trie = TokenTrie(capacity=1)
        trie.put([1], [2])
        trie.put([3], [4])
        self.assertIsNone(trie.get([1]))
        self.assertEqual(trie.get([3]), (4,))

    def test_bridge_merges_draft_subtokens_and_reuses_mapping(self):
        class Draft:
            all_special_ids = []
            def get_vocab(self):
                return {"x": 1, "st": 2, "amps": 3}
            def decode(self, ids, **kwargs):
                return "".join({1: "x", 2: "st", 3: "amps"}[i] for i in ids)
        class Target:
            def get_vocab(self):
                return {"x": 1, "stamps": 99}
            def decode(self, ids, **kwargs):
                return "".join({1: "x", 99: "stamps"}[i] for i in ids)
            def encode(self, text, **kwargs):
                return {"xstamps": [1, 99], "stamps": [99]}[text]
        bridge = VocabularyBridge(Draft(), Target())
        self.assertEqual(bridge.map([1], [2, 3], [1]), [99])
        self.assertEqual(bridge.map([1], [2, 3], [1]), [99])
        self.assertEqual(bridge.stats["trie_hits"], 1)

    def test_cache_incremental_and_longest_suffix(self):
        cache = ContinuationCache(k=3)
        cache.extend([1, 2, 3])
        self.assertEqual(cache.lookup([1, 2]), [3])
        cache.extend([4, 5])
        self.assertEqual(cache.lookup([1, 2]), [3, 4, 5])
        cache.extend([9, 2, 3, 7, 8, 6])
        self.assertEqual(cache.lookup([1, 2, 3]), [4, 5, 9])
        self.assertEqual(cache.lookup([9, 2, 3]), [7, 8, 6])

    def test_cache_no_unobserved_future_or_cross_request_state(self):
        cache = ContinuationCache()
        cache.extend([1, 2])
        self.assertEqual(cache.lookup([1, 2]), [])
        cache.extend([3])
        self.assertEqual(cache.lookup([1, 2]), [3])
        self.assertEqual(ContinuationCache().lookup([1, 2]), [])

    def test_verifier_rejects_entire_suffix_after_first_mismatch(self):
        self.assertEqual(accepted_prefix([1, 2, 3], [1, 9, 3, 4]), 1)
        self.assertEqual(accepted_prefix([1, 2], [1, 2, 3]), 2)
        self.assertEqual(accepted_prefix([], [1]), 0)

    def test_templates(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self_test.assertEqual(kwargs, dict(tokenize=False, add_generation_prompt=True))
                return "<user>" + messages[0]["content"] + "<assistant>"
        class VicunaTokenizer:
            chat_template = None
        self_test = self
        self.assertEqual(draft_prompt("hello"), "hello\n\n")
        self.assertEqual(target_prompt(Tokenizer(), "hello"), "<user>hello<assistant>")
        self.assertEqual(target_prompt(VicunaTokenizer(), "hello"),
                         "A chat between a curious user and an artificial intelligence assistant. "
                         "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                         "USER: hello ASSISTANT:")

    def test_text_bridge_preserves_left_whitespace(self):
        class Tokenizer:
            all_special_ids = []
            def decode(self, ids, **kwargs):
                return "".join(chr(x) for x in ids).lstrip(" ")
            def encode(self, text, **kwargs):
                return list(map(ord, text))
        tokenizer = Tokenizer()
        self.assertEqual(text_proposal(tokenizer, tokenizer, [ord("a")], [32, ord("b")],
                                       [ord("a")], 3), [32, ord("b")])
