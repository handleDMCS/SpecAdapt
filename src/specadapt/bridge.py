"""
Training-free, greedy cross-tokenizer proposal bridges.

All proposals still require target verification. The mutable continuation cache
contains only the current prompt and committed output. An optional frozen cache
contains target-generated continuations of disjoint training questions.
"""

from collections import Counter, OrderedDict


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def text_proposal(draft_tokenizer, target_tokenizer, context, proposed, target_context, k):
    # Decode WITH left context: SentencePiece otherwise drops a leading space.
    before = decode(draft_tokenizer, context)
    after = decode(draft_tokenizer, context + proposed)
    if not after.startswith(before):
        return []
    suffix = after[len(before):]
    # Retokenize across the boundary; never rewrite committed target tokens.
    tail = target_context[-16:]
    combined = target_tokenizer.encode(decode(target_tokenizer, tail) + suffix,
                                       add_special_tokens=False)
    if combined[:len(tail)] == tail:
        return combined[len(tail):][:k]
    return target_tokenizer.encode(suffix, add_special_tokens=False)[:k]


class TokenTrie:
    """Memoize draft-token n-grams to target-token sequences, with a hard cap."""

    def __init__(self, capacity=4096):
        self.root = {}
        self.size = 0
        self.capacity = capacity

    def put(self, key, value):
        if self.size >= self.capacity:
            self.root.clear()
            self.size = 0
        node = self.root
        for token in key:
            node = node.setdefault(token, {})
        if None not in node:
            self.size += 1
        node[None] = tuple(value)

    def get(self, key):
        node = self.root
        for token in key:
            node = node.get(token)
            if node is None:
                return None
        return node.get(None)


class ContinuationCache:
    """Longest matching suffix lookup, scoped to one request (no answer leakage)."""

    def __init__(self, k=3, min_ngram=2, max_ngram=8, capacity=32768):
        self.k, self.min_ngram, self.max_ngram = k, min_ngram, max_ngram
        self.capacity = capacity
        self.entries = OrderedDict()
        self.tokens = []

    def extend(self, tokens):
        old = len(self.tokens)
        self.tokens.extend(tokens)
        # Revisit the final K positions as their continuation becomes available.
        for end in range(max(self.min_ngram, old - self.k + 1), len(self.tokens)):
            continuation = tuple(self.tokens[end:end + self.k])
            for n in range(self.min_ngram, min(self.max_ngram, end) + 1):
                key = tuple(self.tokens[end - n:end])
                previous = self.entries.get(key, ())
                if len(continuation) >= len(previous):
                    self.entries[key] = continuation
                    self.entries.move_to_end(key)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)

    def lookup(self, context):
        return self.match(context)[0]

    def match(self, context):
        for n in range(min(self.max_ngram, len(context)), self.min_ngram - 1, -1):
            result = self.entries.get(tuple(context[-n:]))
            if result:
                return list(result[:self.k]), n
        return [], 0


class FrozenContinuationCache:
    """Read-only n-grams from disjoint training questions, never test answers."""

    def __init__(self, sequences, k=3, min_ngram=3, max_ngram=8):
        self.k, self.min_ngram, self.max_ngram = k, min_ngram, max_ngram
        counts = {}
        for sequence in sequences:
            for end in range(min_ngram, len(sequence) - k + 1):
                value = tuple(sequence[end:end + k])
                for n in range(min_ngram, min(max_ngram, end) + 1):
                    key = tuple(sequence[end - n:end])
                    counts.setdefault(key, Counter())[value] += 1
        self.entries = {key: votes.most_common(1)[0][0] for key, votes in counts.items()}

    def match(self, context):
        for n in range(min(self.max_ngram, len(context)), self.min_ngram - 1, -1):
            value = self.entries.get(tuple(context[-n:]))
            if value:
                return list(value), n
        return [], 0


class VocabularyBridge:
    """Direct overlap map + contextual n-gram trie + verified continuation cache.

    The trie keys include the tokenizer boundary context, so a cached mapping
    cannot silently lose whitespace or reuse a mapping in a different context.
    """

    def __init__(self, draft_tokenizer, target_tokenizer, k=3):
        self.draft, self.target, self.k = draft_tokenizer, target_tokenizer, k
        self.trie = TokenTrie()
        self.continuations = ContinuationCache(k)
        self.direct = {}
        target_vocab = target_tokenizer.get_vocab()
        special_ids = set(draft_tokenizer.all_special_ids)
        # ASCII overlap is unambiguous for Llama SentencePiece / Qwen byte BPE.
        # Other Unicode and byte-fallback cases take the tokenizer fallback.
        for piece, index in draft_tokenizer.get_vocab().items():
            if index in special_ids:
                continue
            surface = piece.replace("\u2581", " ")
            if surface.isascii() and not piece.startswith("<"):
                target_piece = surface.replace(" ", "\u0120").replace("\n", "\u010a")
                if target_piece in target_vocab:
                    self.direct[index] = target_vocab[target_piece]
        self.stats = dict(direct_hits=0, trie_hits=0, tokenizer_fallbacks=0, cache_hits=0, seed_hits=0)

    def map(self, context, proposed, target_context):
        key = tuple(target_context[-16:]) + (-1,) + tuple(context[-4:]) + (-2,) + tuple(proposed)
        cached = self.trie.get(key)
        if cached is not None:
            self.stats["trie_hits"] += 1
            return list(cached)
        result = text_proposal(self.draft, self.target, context, proposed, target_context, self.k)
        direct = [self.direct.get(token) for token in proposed]
        # Multi-token BPE merges must be resolved before installing the fast path.
        if direct and None not in direct and direct[:self.k] == result:
            self.stats["direct_hits"] += 1
            result = direct[:self.k]
        else:
            self.stats["tokenizer_fallbacks"] += 1
        self.trie.put(key, result)
        return result