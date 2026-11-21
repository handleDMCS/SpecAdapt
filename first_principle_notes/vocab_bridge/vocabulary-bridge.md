**Overview**

tl;dr: The draft and target models can assign different numbers and boundaries to the same text. VocabularyBridge converts draft suggestions into target-readable pieces, reusing saved conversions where possible. Every suggestion still needs target verification.

Core source: [VocabularyBridge](../../src/specadapt/bridge.py). The proposal length limit, K, defaults to 3 target pieces.

**Build a shared-piece dictionary**

**tl;dr**: Find text pieces that both models already know.

**Why**: A shared spelling can provide a direct draft-to-target number mapping. The current rules specifically translate Llama-style space markers into Qwen-style space/newline markers; they accept ASCII text, skip draft special pieces and spellings starting with `<`, and leave other cases to text conversion.

**Input**:
```text
- Both models' lists of text pieces and their assigned numbers
- The draft model's list of special-purpose pieces
```

**Output**:
```text
- A dictionary linking eligible draft piece numbers to target piece numbers
```

**Technical keywords**: VocabularyBridge.__init__, direct, SentencePiece, byte-level BPE, ASCII

**Look for a saved conversion**

**tl;dr**: Check whether this suggestion with the same nearby pieces has been converted before.

**Why**: Reusing a conversion avoids decoding and retokenizing again. The lookup includes the last 16 target pieces, the last 4 draft pieces, and the complete draft suggestion, separated by markers. A saved empty result also counts as a hit.

**Input**:
```text
- The draft suggestion
- Nearby draft and target pieces
- Previously saved conversions
```

**Output**:
```text
- The saved target suggestion, returned immediately, or a missing-match result
```

**Technical keywords**: VocabularyBridge.map, TokenTrie, contextual cache key

**Resolve a new conversion**

**tl;dr**: Convert the suggestion through text, then check whether the shared-piece dictionary gives the same answer.

**Why**: Adjacent draft pieces may become one target piece, so individual dictionary entries are insufficient. On every cache miss, text_proposal runs first. Direct overlap is counted only when every proposed piece has a mapping and the first K mapped pieces equal the text-based result; it currently saves no retokenization work.

**Input**:
```text
- The draft suggestion and surrounding text pieces
- Both models' text readers and splitters
- The shared-piece dictionary and length limit
```

**Output**:
```text
- At most K target pieces, possibly an empty list
- A count of either a direct match or a text-conversion fallback
```

**Technical keywords**: text_proposal, direct_hits, tokenizer_fallbacks

**Remember the conversion**

**tl;dr**: Save the answer for reuse and return it to the decoder.

**Why**: Repeated suggestions can take the saved-conversion path next time. This stores a vocabulary conversion, not proof that the target model will accept the suggested text. The decoder verifies it separately.

**Input**:
```text
- The suggestion's lookup key
- The converted target pieces
```

**Output**:
```text
- A saved conversion
- Target pieces ready for verification
```

**Technical keywords**: TokenTrie.put, target verification
