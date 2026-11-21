**Overview**

tl;dr: text_proposal reads the draft suggestion as text, then splits that text into pieces the target understands. It uses surrounding text to preserve spaces and avoids replacing target pieces that have already been committed.

Core source: [text_proposal and decode](../../src/specadapt/bridge.py). This function serves both the naive baseline and M1 when no saved conversion is available. K is the maximum number of returned target pieces.

**Isolate the newly suggested text**

**tl;dr**: Read the draft context before and after adding the suggestion, then keep only the added text.

**Why**: Reading the suggestion alone can drop a leading space. Both readings omit special-purpose pieces and disable spacing cleanup. If the second reading does not begin with the first, the function returns an empty proposal because it cannot isolate a purely appended continuation.

**Input**:
```text
- The draft pieces already present
- The newly suggested draft pieces
- The draft model's text reader
```

**Output**:
```text
- Only the appended text, or an empty proposal if the earlier text changed
```

**Technical keywords**: decode, SentencePiece, startswith, suffix extraction

**Split the new text with nearby target text**

**tl;dr**: Join the new text to the text of the last 16 target pieces and split the combined text again.

**Why**: Piece boundaries can change where old and new text meet. Using the target's own splitter resolves target-side merges, including cases where multiple draft pieces become one target piece. No special-purpose pieces are added during this split.

**Input**:
```text
- The appended text
- Up to 16 final pieces of the committed target context
- The target model's text reader and splitter
```

**Output**:
```text
- Target pieces covering the nearby old text and the new text together
```

**Technical keywords**: target_context[-16:], encode, retokenization, boundary merges

**Choose a continuation that leaves committed pieces intact**

**tl;dr**: Keep the new pieces if the old pieces stayed identical; otherwise split the new text by itself.

**Why**: The function cannot rewrite committed target history. It checks the exact old piece numbers at the start of the combined result. If they match, it removes them; if they differ, it falls back to splitting only the appended text. Either result is capped at K pieces and remains subject to target verification.

**Input**:
```text
- The combined target pieces
- The original final target pieces
- The appended text and maximum suggestion length
```

**Output**:
```text
- At most K suggested target pieces, without replacing committed pieces
```

**Technical keywords**: prefix equality, suffix-only fallback, proposal truncation
