**Overview**

tl;dr: ContinuationCache and FrozenContinuationCache remember which target pieces followed earlier sequences. A matching ending can supply the next suggestion without running the draft model; the target still checks every suggested piece.

Core sources: [both cache classes](../../src/specadapt/bridge.py) and [selection and verification in generate](../../src/specadapt/decoding.py). K defaults to 3 suggested pieces. These caches store target-to-target continuations, whereas TokenTrie stores draft-to-target conversions.

**Learn continuations within the current request**

**tl;dr**: Record what followed short sequences in the prompt and accepted output.

**Why**: Repeated wording can supply useful future suggestions. The decoder first adds the plain prompt encoded as target pieces, then adds only committed output. By default, the cache records endings of 2–8 pieces and up to K following pieces, revisiting recent positions as more output arrives. An equally long or longer continuation replaces an older one; above 32,768 entries, the least recently written entries are removed. The accumulated piece history itself is not capped.

**Input**:
```text
- The prompt pieces, followed over time by newly committed output pieces
- The existing history and remembered continuations for this request
```

**Output**:
```text
- Updated history and a bounded dictionary of endings and their following pieces
```

**Technical keywords**: ContinuationCache.extend, n-gram, OrderedDict, incremental update

**Build the optional frozen reference store**

**tl;dr**: Count continuations in supplied example sequences and keep the most frequent one for each ending.

**Why**: Previously generated examples can provide useful continuations before the current request repeats itself. By default, this store uses endings of 3–8 pieces and requires a full K-piece continuation. It has no entry cap and stays unchanged during generation. The intended examples are target-generated answers to separate training questions; the constructor receives only numbers and cannot enforce that separation itself.

**Input**:
```text
- Separate example sequences already expressed as target pieces
- The ending lengths and suggestion length to retain
```

**Output**:
```text
- A fixed dictionary containing the most frequent continuation for each ending
```

**Technical keywords**: FrozenContinuationCache.__init__, Counter, most_common

**Find the longest remembered ending**

**tl;dr**: Compare the end of the generated answer with remembered endings, longest first.

**Why**: A longer match uses more preceding information. Each cache returns its first matching continuation and the match length. The decoder searches using generated output only, even though the mutable store was initially populated from the prompt; an empty generated answer therefore cannot match yet.

**Input**:
```text
- The target pieces generated so far
- A store of remembered endings and continuations
```

**Output**:
```text
- Suggested following pieces and the matched ending length, or no suggestion and zero
```

**Technical keywords**: match, longest-suffix lookup, generated

**Choose a cached suggestion**

**tl;dr**: Use the frozen suggestion only when it matches a longer ending than the current-request suggestion.

**Why**: The decoder needs one candidate sequence per round. Ties favor the mutable cache. A cache hit skips drafting and vocabulary conversion for that round; no hit leads to draft generation and bridge mapping. Setting use_cache to false skips continuation lookup, while bridge mapping remains available.

**Input**:
```text
- The current-request suggestion and match length
- The optional frozen suggestion and match length
- Whether continuation lookup is enabled
```

**Output**:
```text
- One cached suggestion, or a decision to ask the draft model
```

**Technical keywords**: generate, seed_cache, use_cache, seed_length

**Verify before committing**

**tl;dr**: Keep cached pieces only while they agree with the target's next choices.

**Why**: Remembering a continuation does not make it correct in a new context. The greedy decoder accepts the matching prefix, then adds the target's correction or next piece when space remains, respecting the stop token. Only this committed output is added to the mutable cache; the frozen store is unchanged.

**Input**:
```text
- The selected suggestion, capped by the remaining output allowance
- The target model's predictions for this round
```

**Output**:
```text
- Committed output pieces that update the answer and current-request cache
```

**Technical keywords**: accepted_prefix, greedy verification, committed, ContinuationCache.extend
