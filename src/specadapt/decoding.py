"""Greedy speculative decoding with rollback of both models' KV caches."""

import time

from specadapt.bridge import VocabularyBridge, decode, text_proposal


def accepted_prefix(proposed, predictions):
    count = 0
    for proposal, prediction in zip(proposed, predictions):
        if proposal != prediction:
            break
        count += 1
    return count


def draft_prompt(prompt):
    return prompt + "\n\n"


def target_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)


class CachedModel:
    def __init__(self, model):
        self.model = model
        self.ids = []
        self.cache = None
        self.last_logits = None
        self.steps = 0
        self.seconds = 0.0

    def sync(self, ids):
        import torch
        common = 0
        for a, b in zip(self.ids, ids):
            if a != b:
                break
            common += 1
        if common == len(ids) and common == len(self.ids):
            return self.last_logits
        # Need one input token to recover logits when the new prefix is shorter.
        common = min(common, len(ids) - 1)
        if self.cache is not None:
            self.cache.crop(common)
        inputs = torch.tensor([ids[common:]], dtype=torch.long, device=self.model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.model(input_ids=inputs, past_key_values=self.cache, use_cache=True)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - started
        self.steps += 1
        self.cache = output.past_key_values
        self.ids = list(ids)
        self.last_logits = output.logits[:, -1:, :]
        return output.logits

    def propose(self, ids, k, eos):
        proposed = []
        for _ in range(k):
            logits = self.sync(ids + proposed)
            token = int(logits[0, -1].argmax())
            if token in eos:
                break
            proposed.append(token)
        return proposed


def generate(target, draft, target_tokenizer, draft_tokenizer, prompt, *, mode,
             max_new_tokens=128, k=3, use_cache=True, seed_cache=None):
    import torch
    if mode not in {"target", "baseline", "m1"}:
        raise ValueError(mode)
    target_ids = target_tokenizer.encode(target_prompt(target_tokenizer, prompt), add_special_tokens=False)
    generated = []
    target_state, draft_state = CachedModel(target), CachedModel(draft)
    setup_start = time.perf_counter()
    bridge = VocabularyBridge(draft_tokenizer, target_tokenizer, k) if mode == "m1" else None
    if bridge:
        bridge.continuations.extend(target_tokenizer.encode(prompt, add_special_tokens=False))
    bridge_setup_seconds = time.perf_counter() - setup_start
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    draft_eos = {draft_tokenizer.eos_token_id}
    rounds = accepted = proposed_count = draft_tokens = cache_rounds = 0
    bridge_seconds = 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        # Cache all but the final token so each verification predicts its proposals.
        target_state.sync(target_ids[:-1])
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            candidates = []
            bridge_start = time.perf_counter()
            if bridge and use_cache:
                candidates, match_length = bridge.continuations.match(generated)
                if seed_cache is not None:
                    seeded, seed_length = seed_cache.match(generated)
                    if seed_length > match_length:
                        candidates = seeded
                        bridge.stats["seed_hits"] += 1
                if candidates:
                    bridge.stats["cache_hits"] += 1
                    cache_rounds += 1
            bridge_seconds += time.perf_counter() - bridge_start
            if mode != "target" and not candidates:
                bridge_start = time.perf_counter()
                context = draft_tokenizer.encode(draft_prompt(prompt) + decode(target_tokenizer, generated),
                                                  add_special_tokens=True)
                bridge_seconds += time.perf_counter() - bridge_start
                raw = draft_state.propose(context, k, draft_eos)
                draft_tokens += len(raw)
                bridge_start = time.perf_counter()
                if bridge:
                    candidates = bridge.map(context, raw, target_ids + generated)
                else:
                    candidates = text_proposal(draft_tokenizer, target_tokenizer, context, raw,
                                               target_ids + generated, k)
                bridge_seconds += time.perf_counter() - bridge_start
            candidates = candidates[:remaining]
            prefix = target_ids + generated
            # Crop away rejected tokens left over from the previous round.
            target_state.cache.crop(len(prefix) - 1)
            target_state.ids = prefix[:-1]
            logits = target_state.sync(prefix + candidates)
            predictions = logits[0].argmax(dim=-1).tolist()
            count = accepted_prefix(candidates, predictions)
            committed = candidates[:count]
            if len(committed) < remaining:
                committed.append(predictions[count])
            for position, token in enumerate(committed):
                if token in eos:
                    committed = committed[:position + 1]
                    count = min(count, len(committed))
                    break
            generated.extend(committed)
            bridge_start = time.perf_counter()
            if bridge:
                bridge.continuations.extend(committed)
            bridge_seconds += time.perf_counter() - bridge_start
            rounds += 1
            accepted += count
            proposed_count += len(candidates)
            if generated[-1] in eos:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started + bridge_setup_seconds
    return dict(mode=mode, token_ids=generated, text=decode(target_tokenizer, generated),
                generated_tokens=len(generated), seconds=elapsed, tokens_per_second=len(generated) / elapsed,
                rounds=rounds, accepted=accepted, proposed=proposed_count,
                acceptance_rate=accepted / proposed_count if proposed_count else 0.0,
                accepted_per_round=accepted / rounds, draft_tokens=draft_tokens,
                draft_steps=draft_state.steps, draft_seconds=draft_state.seconds,
                bridge_setup_seconds=bridge_setup_seconds,
                bridge_seconds=bridge_seconds, bridge_ms_per_token=1000 * bridge_seconds / len(generated),
                cache_rounds=cache_rounds, bridge_stats=bridge.stats if bridge else {})
