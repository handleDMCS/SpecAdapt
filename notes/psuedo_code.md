**Algorithm 1:** One speculative round

**Input:**
- `prompt` (`str`): the user's question as plain text.
- `x` (`list[int]`): target token ids of the prompt in chat format, from `target_tok.encode(chat_template(prompt), add_special_tokens=False)`.
- `y` (`list[int]`): target token ids committed so far. It is empty in the first round.
- `store` (`ContinuationCache`): continuations remembered from this request, filled from the plain prompt plus `y`.
- `trie` (`TokenTrie`): saved draft-to-target conversions for this request. Its keys are `tuple[int, ...]` and its values are `tuple[int, ...]`.
- `frozen` (`FrozenContinuationCache | None`): optional read-only continuations built from training answers.
- `remaining` (`int`): how many output tokens are still allowed, equal to `N - len(y)`.
- `K` (`int`): the maximum number of suggested tokens per round (default `3`).
- Shared objects, fixed for the whole run:
  - `target`, `draft` (`CachedModel`): wrappers around Hugging Face `PreTrainedModel` that keep their own KV cache.
  - `target_tok`, `draft_tok` (`PreTrainedTokenizerBase`).
  - `eos` (`set[int]`): the target's stop token ids.
- Helper `convert(c, d, h, K)`:
  - `c` (`list[int]`): the draft context ids.
  - `d` (`list[int]`): the draft's suggested ids (at most `K`).
  - `h` (`list[int]`): the target history `x + y`.
  - It returns `list[int]`: at most `K` target ids, possibly empty.

**Output:**
- `u` (`list[int]`): the target token ids newly committed this round, with `1 <= len(u) <= min(K + 1, remaining)`. It includes the target's correction or bonus token, and ends at the first stop token if one appears.

**Pseudo code**
```python
 1  s, n = store.match(y)                                  # (list[int], int)
 2  if frozen is not None:
 3      s2, n2 = frozen.match(y)
 4      if n2 > n: s = s2
 5  if not s:
 6      c = draft_tok.encode(prompt + "\n\n" + target_tok.decode(y))
 7      d = draft.greedy(c, max_tokens=K, stop_before=draft_tok.eos_token_id)
 8      key = (*(x + y)[-16:], -1, *c[-4:], -2, *d)          # tuple[int, ...]
 9      cached = trie.get(key)
10      if cached is not None: s = list(cached)
11      else: s = convert(c, d, x + y, K);  trie.put(key, s)
12  s = s[:remaining]
13  target.crop_cache(len(x + y) - 1)
14  pred = target.argmax((x + y)[-1:] + s)                 # list[int], len(s) + 1
15  a = count_matching_prefix(s, pred)
16  u = s[:a] + ([pred[a]] if a < remaining else [])
17  return cut_after_first_eos(u, eos)


def convert(c, d, h, K):
 1  before = draft_tok.decode(c);  after = draft_tok.decode(c + d)
 2  if not after.startswith(before): return []
 3  new_text = after[len(before):]
 4  tail = h[-16:]
 5  z = target_tok.encode(target_tok.decode(tail) + new_text)
 6  if z[:len(tail)] == tail: return z[len(tail):][:K]
 7  return target_tok.encode(new_text)[:K]
```

**Algorithm 2:** Full generation

**Input:**
- `prompt` (`str`): the user's question as plain text.
- `N` (`int`): the maximum number of new tokens (`max_new_tokens`, default `128`).
- `K` (`int`): the maximum number of suggested tokens per round (default `3`).
- `frozen` (`FrozenContinuationCache | None`): optional read-only continuations (`seed_cache`, default `None`).
- Shared objects: `target`, `target_tok` and `eos`, with the same types as in Algorithm 1.

**Output:**
- `y` (`list[int]`): the generated target token ids, with `len(y) <= N`. If generation stopped early, the last id is in `eos`. The real `generate()` returns this inside a `dict` along with the decoded text and timing statistics.

**Pseudo code**
```python
1  x = target_tok.encode(chat_template(prompt), add_special_tokens=False)
2  y = []
3  store = ContinuationCache(K);  trie = TokenTrie()
4  store.extend(target_tok.encode(prompt, add_special_tokens=False))
5  target.prefill(x[:-1])
6  while len(y) < N:
7      u = round(prompt, x, y, store, trie, frozen, N - len(y), K)
8      y += u;  store.extend(u)
9      if y[-1] in eos: break
10 return y
```

I made two changes so the types match the real classes:

- Algorithm 1 now uses `trie.get` / `trie.put`, since `TokenTrie` isn't a dict. The key is now the flat tuple the code actually builds, with `-1` and `-2` as separators.
- Algorithm 2 now creates `store` and `trie` explicitly in line 3. In `generate()` they are created inside `VocabularyBridge`.