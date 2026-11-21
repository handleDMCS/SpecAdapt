**Overview**

tl;dr: Each M1 round first tries a remembered target continuation. If none fits, the draft model suggests pieces and VocabularyBridge converts them into target pieces. The target verifies the suggestion, and committed output feeds the next round.

The loop lives in [generate](../../src/specadapt/decoding.py); the four components live in [bridge.py](../../src/specadapt/bridge.py). This describes greedy M1 with continuation lookup enabled and a fixed proposal limit K, normally 3.

| Route through one round | Components used, in order |
|---|---|
| A remembered continuation matches | Continuation caches → target verification |
| A draft conversion is already saved | Continuation cache miss → draft model → VocabularyBridge → TokenTrie hit → target verification |
| A draft conversion is new | Continuation cache miss → draft model → VocabularyBridge → TokenTrie miss → text_proposal → direct-overlap check → save conversion → target verification |

All routes finish by adding committed output to the current-request continuation cache. A continuation hit avoids drafting; a trie hit avoids conversion after drafting has already happened.

**1. Prepare the bridge for this request**

**tl;dr**: Build the shared-piece dictionary and create places to remember conversions and continuations.

**Why**: [VocabularyBridge](vocabulary-bridge.md) owns the direct overlap map, an empty TokenTrie, and a new ContinuationCache. The decoder fills the continuation cache from the plain prompt expressed as target pieces. An optional FrozenContinuationCache arrives separately, already built from supplied examples.

**Input**:
```text
- Both models' text readers, splitters, and piece dictionaries
- The prompt and proposal length limit
- An optional prepared store of example continuations
```

**Output**:
```text
- A request-specific bridge and access to the optional frozen store
```

**Technical keywords**: VocabularyBridge.__init__, generate, seed_cache

**2. Look for a remembered continuation**

**tl;dr**: Find the longest remembered ending that matches the answer generated so far.

**Why**: [Continuation caches](continuation-caches.md) already return target pieces. The decoder compares both stores, choosing the frozen suggestion only for a strictly longer match; ties favor the current-request store. A hit goes straight to step 8. With no generated output yet, neither store can match.

**Input**:
```text
- The generated answer's target pieces
- The current-request and optional frozen continuation stores
```

**Output**:
```text
- A target suggestion for verification, or no match
```

**Technical keywords**: ContinuationCache.match, FrozenContinuationCache.match, longest-suffix lookup

**3. Ask the draft model for a suggestion**

**tl;dr**: When no continuation matches, let the draft model extend the prompt and current answer.

**Why**: The draft needs its own representation of the shared text. The decoder splits the plain prompt plus two newlines and the decoded answer using the draft's splitter, then generates up to K draft pieces, stopping before a draft stop token.

**Input**:
```text
- The original prompt and committed answer text
- The draft model and proposal length limit
```

**Output**:
```text
- Draft context and newly suggested draft pieces
```

**Technical keywords**: draft_prompt, CachedModel.propose

**4. Look for an existing conversion**

**tl;dr**: Check whether this draft suggestion has already been converted with the same nearby pieces.

**Why**: VocabularyBridge.map asks [TokenTrie](token-trie.md) for an exact key containing the last 16 target pieces, last 4 draft pieces, and full suggestion. A saved result, including an empty result, goes straight to step 8. Only a missing key requires fresh conversion.

**Input**:
```text
- The draft suggestion and nearby draft and target pieces
- The saved conversion tree
```

**Output**:
```text
- Saved target pieces for verification, or a conversion miss
```

**Technical keywords**: VocabularyBridge.map, TokenTrie.get

**5. Convert a new suggestion through text**

**tl;dr**: Read the new draft text and split it into target pieces without replacing committed pieces.

**Why**: [text_proposal](text-proposal.md) decodes with draft context to preserve spacing, then splits with nearby target text to resolve boundaries. If that changes the old target pieces, it splits only the appended text. If decoding changes the earlier draft text, it returns an empty proposal.

**Input**:
```text
- Draft context and suggested pieces
- Committed target context and both models' text readers and splitters
```

**Output**:
```text
- Up to K target pieces, possibly an empty list
```

**Technical keywords**: text_proposal, retokenization

**6. Check the direct overlap map**

**tl;dr**: Compare the shared-piece dictionary's answer with the text-based answer.

**Why**: VocabularyBridge counts a direct hit only if every draft piece has a mapping and the first K mapped pieces equal the converted result. Otherwise it counts a fallback. This check happens after text_proposal, so direct overlap currently does not avoid retokenization.

**Input**:
```text
- The draft suggestion and shared-piece dictionary
- The text-based conversion
```

**Output**:
```text
- The same target suggestion, classified as a direct match or fallback
```

**Technical keywords**: direct, direct_hits, tokenizer_fallbacks

**7. Save the new conversion**

**tl;dr**: Remember the converted pieces under this suggestion's context key.

**Why**: TokenTrie can reuse this mapping on a later exact match. Saving happens before target verification: it records how to represent the suggestion, not whether the target approves it.

**Input**:
```text
- The complete conversion key and converted target pieces
```

**Output**:
```text
- An updated conversion tree and a suggestion ready for verification
```

**Technical keywords**: TokenTrie.put

**8. Verify and commit the next output**

**tl;dr**: Keep suggested pieces while they match the target, then use the target's own next choice.

**Why**: Every route supplies candidates to the same greedy verification step. The decoder caps them by the remaining output allowance, removes stale rejected target state, accepts the matching prefix, and adds a correction or bonus piece if room remains. An empty proposal still allows one target-generated piece; a stop token ends output.

**Input**:
```text
- Candidate target pieces and committed target history
- The target model and remaining output allowance
```

**Output**:
```text
- Newly committed answer pieces
```

**Technical keywords**: accepted_prefix, CachedModel.sync, KV-cache rollback, greedy verification

**9. Feed committed output into the next round**

**tl;dr**: Teach the current-request cache what followed, then repeat if the answer is unfinished.

**Why**: ContinuationCache.extend records the committed pieces, including target corrections and bonus pieces. Rejected suggestions never enter this continuation history; the frozen store stays unchanged. The next round returns to step 2, and any later draft call rebuilds its context from the committed answer.

**Input**:
```text
- Newly committed pieces and the current-request continuation history
- The stop condition and output length limit
```

**Output**:
```text
- Updated remembered continuations and either another round or the finished answer
```

**Technical keywords**: ContinuationCache.extend, generated, eos
