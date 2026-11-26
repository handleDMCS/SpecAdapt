"""Greedy speculative decoding with rollback of both models' KV caches."""

import time

from specadapt.adaptive_k import text_labels, token_features
from specadapt.bridge import VocabularyBridge, decode, text_proposal

# Left context used to decode a few tokens without losing a leading space.
LABEL_CONTEXT = 16


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
    if getattr(tokenizer, "chat_template", "") is None:
        return ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: {prompt} ASSISTANT:")
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

    def propose(self, ids, k, eos, policy=None, features=None):
        """Draft up to k tokens. A policy may stop after any drafted token; features collects each
        drafted token's features (FP32) when given."""
        proposed = []
        for _ in range(k):
            logits = self.sync(ids + proposed)
            token = int(logits[0, -1].argmax())
            if token in eos:
                break
            proposed.append(token)
            if policy is not None or features is not None:
                values = token_features(logits[0, -1])
                if features is not None:
                    features.append(values)
                if policy is not None and not policy.should_continue(values, len(proposed)):
                    break
        return proposed


def appended_text_length(tokenizer, context, ids):
    """Length of the text that ids add after context, decoded with left context."""
    for left in (context[-LABEL_CONTEXT:], context):
        before = decode(tokenizer, left)
        after = decode(tokenizer, left + ids)
        if after.startswith(before):
            return len(after) - len(before)
    return len(after) - len(before)


def generate(target, draft, target_tokenizer, draft_tokenizer, prompt, *, mode,
             max_new_tokens=128, k=3, use_cache=True, seed_cache=None, oracle=None, same_tokenizer=False,
             policy=None, cache_k=None, trace=False):
    """Speculative decoding, or its oracle replay.

    policy: decides after each drafted token whether to draft another (k must equal policy.k_max).
    cache_k: how many tokens a continuation-cache hit proposes (default k).
    trace: also return one record per round, with per drafted token its features and text label.
    In the policy and trace paths a mapped draft proposal is never cut to k target tokens.
    """
    import torch
    if mode not in {"target", "baseline", "m1"}:
        raise ValueError(mode)
    if type(k) is not int or k <= 0 or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("k and max_new_tokens must be positive integers.")
    if cache_k is not None and (type(cache_k) is not int or cache_k <= 0):
        raise ValueError("cache_k must be a positive integer.")
    if policy is not None and (mode == "target" or policy.k_max != k):
        raise ValueError("A policy needs a draft mode and k equal to its k_max.")
    # Never shorten a k_max-token draft proposal when a policy or a label trace is in use.
    cut = None if policy is not None or trace else k
    if oracle is not None:
        if target is not None or mode == "target":
            raise ValueError("Oracle replay requires a draft mode and no target model.")
        oracle.validate_limit(max_new_tokens)
    target_ids = target_tokenizer.encode(target_prompt(target_tokenizer, prompt), add_special_tokens=False)
    generated = []
    target_state = CachedModel(target) if oracle is None else None
    draft_state = CachedModel(draft)
    setup_start = time.perf_counter()
    bridge = (VocabularyBridge(draft_tokenizer, target_tokenizer, cut, k if cache_k is None else cache_k)
              if mode == "m1" else None)
    if bridge:
        bridge.continuations.extend(target_tokenizer.encode(prompt, add_special_tokens=False))
    bridge_setup_seconds = time.perf_counter() - setup_start
    if oracle is None:
        eos = target.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
    else:
        eos = oracle.eos_token_ids
    draft_eos = set() if mode == "target" else {draft_tokenizer.eos_token_id}
    rounds = accepted = proposed_count = draft_tokens = cache_rounds = 0
    cache_by_match = {}
    trace_rounds = []
    bridge_seconds = 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        # Cache all but the final token so each verification predicts its proposals.
        if target_state is not None:
            target_state.sync(target_ids[:-1])
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            candidates = []
            match_length = 0
            raw = None
            bridge_start = time.perf_counter()
            if bridge and use_cache:
                candidates, match_length = bridge.continuations.match(generated)
                if seed_cache is not None:
                    seeded, seed_length = seed_cache.match(generated)
                    if seed_length > match_length:
                        candidates, match_length = seeded, seed_length
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
                features = [] if trace else None
                steps_before = draft_state.steps
                raw = draft_state.propose(context, k, draft_eos, policy, features)
                draft_tokens += len(raw)
                bridge_start = time.perf_counter()
                if bridge:
                    candidates = bridge.map(context, raw, target_ids + generated)
                elif same_tokenizer:
                    candidates = raw
                else:
                    candidates = text_proposal(draft_tokenizer, target_tokenizer, context, raw,
                                               target_ids + generated, cut)
                bridge_seconds += time.perf_counter() - bridge_start
            limited = len(candidates) > remaining
            candidates = candidates[:remaining]
            if oracle is None:
                prefix = target_ids + generated
                # Crop away rejected tokens left over from the previous round.
                target_state.cache.crop(len(prefix) - 1)
                target_state.ids = prefix[:-1]
                logits = target_state.sync(prefix + candidates)
                predictions = logits[0].argmax(dim=-1).tolist()
            else:
                predictions = oracle.predictions(generated, candidates)
            count = accepted_prefix(candidates, predictions)
            committed = candidates[:count]
            if len(committed) < remaining and not any(token in eos for token in committed):
                committed.append(predictions[count])
            for position, token in enumerate(committed):
                if token in eos:
                    committed = committed[:position + 1]
                    count = min(count, len(committed))
                    break
            if raw is None and candidates:
                hits = cache_by_match.setdefault(str(match_length), dict(rounds=0, proposed=0, accepted=0))
                hits["rounds"] += 1
                hits["proposed"] += len(candidates)
                hits["accepted"] += count
            if trace:
                source = "draft" if raw is not None else "cache" if candidates else "target"
                record = dict(source=source, match=match_length,
                              proposed=len(candidates), accepted=count)
                if raw is not None:
                    prefix = target_ids + generated
                    lengths = [appended_text_length(draft_tokenizer, context, raw[:i])
                               for i in range(1, len(raw) + 1)]
                    accepted_length = appended_text_length(target_tokenizer, prefix, candidates[:count])
                    verified_length = appended_text_length(target_tokenizer, prefix, candidates) if limited else None
                    labels = text_labels(lengths, accepted_length, verified_length)
                    record.update(steps=draft_state.steps - steps_before, drafted=len(raw),
                                  tokens=[dict(features[i], label=label) for i, label in enumerate(labels)])
                trace_rounds.append(record)
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
    result = dict(mode=mode, token_ids=generated, text=decode(target_tokenizer, generated),
                generated_tokens=len(generated), seconds=elapsed, tokens_per_second=len(generated) / elapsed,
                rounds=rounds, accepted=accepted, proposed=proposed_count,
                acceptance_rate=accepted / proposed_count if proposed_count else 0.0,
                accepted_per_round=accepted / rounds, draft_tokens=draft_tokens,
                draft_steps=draft_state.steps, draft_seconds=draft_state.seconds,
                bridge_setup_seconds=bridge_setup_seconds,
                bridge_seconds=bridge_seconds, bridge_ms_per_token=1000 * bridge_seconds / len(generated),
                cache_rounds=cache_rounds, bridge_stats=bridge.stats if bridge else {},
                cache_by_match=cache_by_match)
    if trace:
        result["trace"] = trace_rounds
    if oracle is not None:
        result["verification"] = "oracle"
        result["replay_seconds"] = result.pop("seconds")
        result.pop("tokens_per_second")
    else:
        result.update(target_model_seconds=target_state.seconds, target_steps=target_state.steps)
    return result
