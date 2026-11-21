"""A verifier for a fixed greedy trajectory; never a source of proposals."""


class GreedyOracle:
    def __init__(self, token_ids, eos_token_ids):
        self.token_ids = tuple(token_ids)
        self.eos_token_ids = frozenset(eos_token_ids)
        if not self.token_ids or not self.eos_token_ids:
            raise ValueError("Oracle tokens and EOS IDs must be nonempty.")
        if any(type(token) is not int or token < 0 for token in (*self.token_ids, *self.eos_token_ids)):
            raise ValueError("Oracle token IDs must be nonnegative integers.")
        if any(token in self.eos_token_ids for token in self.token_ids[:-1]):
            raise ValueError("Oracle contains tokens after EOS.")

    def validate_limit(self, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")
        if len(self.token_ids) < max_new_tokens and self.token_ids[-1] not in self.eos_token_ids:
            raise ValueError("Oracle is truncated before the requested token limit; record a longer reference.")

    def predictions(self, generated, candidates):
        position = len(generated)
        if tuple(generated) != self.token_ids[:position]:
            raise ValueError("Committed prefix differs from the oracle trajectory.")
        if position >= len(self.token_ids):
            raise ValueError("Oracle exhausted.")
        # After the first mismatch no other prediction is consulted. A slice
        # ending at EOS is sufficient even if the draft proposed beyond EOS.
        return self.token_ids[position:position + len(candidates) + 1]
