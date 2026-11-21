**Overview**

tl;dr: TokenTrie remembers conversions from draft pieces to target pieces. It follows a path of numbers to retrieve an exact saved result; the actual text conversion happens elsewhere.

Core source: [TokenTrie and VocabularyBridge.map](../../src/specadapt/bridge.py). The bridge supplies the context-aware keys; TokenTrie itself accepts arbitrary sequences of numbers.

**Follow an exact lookup path**

**tl;dr**: Walk through the saved branches using each number in the lookup key.

**Why**: Suggestions can share an initial sequence but require different answers. Retrieval requires the entire key and a stored result at its endpoint; this is not a longest-prefix search. In the bridge, the key contains recent target context, recent draft context, and the draft suggestion.

**Input**:
```text
- An ordered list of numbers identifying a conversion
- The tree of previously saved conversions
```

**Output**:
```text
- The exact saved result, including an empty result, if present
- A missing-match result otherwise
```

**Technical keywords**: TokenTrie.get, trie, exact-key lookup, None sentinel

**Reset a full store before writing**

**tl;dr**: Clear all saved conversions when the store has reached its entry limit.

**Why**: The default limit is 4,096 stored keys. The check happens before every write, including an attempted replacement of an existing key. Clearing limits retained entries, but does not impose a fixed byte limit because keys can have different lengths.

**Input**:
```text
- The current number of saved keys
- The configured entry limit
```

**Output**:
```text
- An empty store if the limit was reached, otherwise the existing store
```

**Technical keywords**: TokenTrie.put, capacity, full-cache reset

**Store the conversion at the path's endpoint**

**tl;dr**: Create any missing branches and attach the converted pieces at the end.

**Why**: Branches share common beginnings, while an endpoint marker separates the saved answer from child branches. Writing an existing endpoint replaces its answer; writing a new endpoint increases the entry count. Empty conversions can be stored too.

**Input**:
```text
- The complete lookup key
- The converted target pieces
```

**Output**:
```text
- A retrievable conversion stored at the complete key
- An updated count of stored keys
```

**Technical keywords**: TokenTrie.put, nested dictionaries, tuple, terminal marker
