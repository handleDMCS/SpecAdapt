# Draft-only greedy oracle replay

Implemented and measured on 2026-11-21. Reusing the recorded greedy target
trajectory cut Kaggle script execution by **90–91%** across two fresh T4
submissions. The time from command start to result download fell by **60%** in
the first replay and **7%** in the second. Kaggle overhead dominated the slower
repeat. These timings measure the adapter development workflow.

## What changed

- `src/specadapt/oracle.py`: a greedy token-ID verifier with prefix, EOS, and
  reference-length guards.
- `src/specadapt/decoding.py`: optional oracle verification in the existing
  decoding loop. Draft proposals, tokenizer bridging, training-cache lookups,
  accepted-prefix counting, and committed-prefix updates remain shared with
  live verification. Future reference tokens are never passed to the proposer
  or used to initialize the M1 continuation cache.
- `src/demo/oracle.py`: versioned, fingerprinted TOML reference files, with
  target/dataset revisions, tokenizer backend/vocabulary/chat-template binding,
  target numerical settings, package versions, GPU, exact prompt hashes and
  input token IDs, output IDs, recording limits, and termination reasons.
- `src/demo/benchmark.py`: records references during ordinary full demos;
  supports target-only recording and draft-only replay; records setup-inclusive
  benchmark wall time and decoding wall time. Replay records explicitly list
  `loaded_models = ["llama_68m"]`.
- `src/demo/record_oracle.py`: target-only reference recording, without loading
  draft weights.
- `src/demo/extract_oracle.py`: extracts a validated reference TOML from a
  completed submission JSON without loading models or importing test modules.
- `src/demo/demo_oracle.py` and `oracle_demo.toml`: replay entry point and settings.
- `tests/test_oracle.py`, `tests/test_decoding.py`: synthetic guards and
  live-versus-replay equivalence tests, including rejected proposals, empty
  proposals, EOS, token limits, continuation-cache hits, and incorrect seed proposals.

The input project bundle and single-JSON submission result format are unchanged.
The existing uncommitted JSON-result refactor is retained.

## First recording

A normal full run now exports `demo_results/target_references.toml` in its result:

```powershell
uv run python -m demo.kaggle_run src/demo/prepare_data.py src/demo/demo_m1.py
```

If only reference recording is needed, use this instead; it loads the target
model, but not the draft:

```powershell
uv run python -m demo.kaggle_run src/demo/prepare_data.py src/demo/record_oracle.py
```

Both use the prompt IDs and token limit in `pair2_demo.toml`. Downloaded output
is `results/submission-<timestamp>.json`. Extract the TOML using the actual result path:

```powershell
uv run python -m demo.extract_oracle results/submission-<timestamp>.json
```

This creates root-level `target_references.toml`. It is ignored by Git, included
by the existing TOML submission bundler, and excluded from the public Repomix
source page's allowlist. Extraction refuses to overwrite an existing reference;
use `--output another_references.toml` and change `reference_file` when recording
another version. Alternative TOML names are not automatically Git-ignored.

Older results containing only `target_reference.token_ids` lack the exact prompt
and tokenizer bindings. The extractor rejects these rather than inferring their
provenance. Run a new recording/full demo once.

## Subsequent runs

```powershell
uv run python -m demo.kaggle_run src/demo/prepare_data.py src/demo/demo_oracle.py
```

`oracle_demo.toml` defaults to M1 and inherits the full benchmark workload:

```toml
mode = "m1"  # also "baseline" or "compare"
reference_file = "target_references.toml"
# Optional development overrides:
# max_new_tokens = 128
# k = 5
# prompt_ids = [0, 1, 2]
```

Changing K or the draft implementation does not invalidate target references.
Shorter outputs and subsets of recorded prompts are allowed. A longer token
limit is allowed only when the reference already ended in EOS; otherwise a
longer recording is required. Changed target/tokenizer/dataset/environment
provenance or changed prompt tokens is rejected before model weight loading.
Replay still needs target tokenizer/config files and cached dataset inputs.
`prepare_data.py` checks the existing caches; it does not load target weights.

M1 replay requires the separate training-derived `m1_seed_cache.toml`. It never
generates that cache with the target during replay and never substitutes test
references for training-cache data. The repository already contains this cache;
its existing provenance check remains enabled.

## Interpreting results

Replay artifacts use `demo_results/oracle_m1.jsonl`,
`oracle_m1_metadata.json`, and `oracle_m1_summary.json` (or `oracle_baseline`).
They contain acceptance, accepted tokens per round, proposal/round counts,
bridge/cache counters, draft time, and bridge overhead. Timing is labeled
`replay_seconds`; the results omit `tokens_per_second`, target speedup, and
`exact_match`. These omissions prevent oracle timing from being mistaken for
two-model performance or replay agreement from being mistaken for a live
correctness check.

Acceptance is measured against the recorded greedy trajectory. Quantized live
verification may take a different trajectory and consequently change later
proposals and cache hits. Draft-only timing also runs under different GPU memory
pressure. Use `demo_compare.py` for final two-model throughput and agreement
checks. This oracle does not support stochastic verification or target-dependent
adaptive policies.

## Validation and runtime comparison

Local validation is limited to syntax checks and staging dry runs; tests and
model execution run on Kaggle. All 44 synthetic tests passed on Kaggle CPU in
12.680 seconds: [submission 20261121T045412438748Z](../results/submission-20261121T045412438748Z.json).
This includes the previous JSON-result refactor's tests. Source syntax checks,
the test-suite staging dry run, and `git diff --check` also passed.

The runtime comparison used the same 12 GSM8K test prompts,
`max_new_tokens = 512`, K=3, seed 42, pinned Tesla T4 environment, and bundled
training cache. All three submissions used identical sources and Kaggle
settings. The replay bundles added only the reference TOML.

| Run | Command to result | Kaggle script execution | Benchmark with setup | Decoding loop |
|---|---:|---:|---:|---:|
| [Full M1 and reference recording](../results/submission-20261121T045555104115Z.json) | 842.6 s | 807.0 s | 793.7 s | 571.2 s |
| [First M1 replay](../results/submission-20261121T052243824146Z.json) | 340.3 s | 80.2 s | 75.9 s | 12.6 s |
| [Second M1 replay](../results/submission-20261121T052835364777Z.json) | 780.2 s | 71.4 s | 68.4 s | 12.6 s |

Command to result includes cache discovery, queueing, startup, polling, and
download. Kaggle script execution includes `prepare_data.py` and the benchmark.
Decoding time for the full run includes 228.1 s of target-only generation and
343.1 s of live M1 decoding. The second replay spent 708.8 s outside the scripts,
versus 260.1 s for the first. Both replays saved more than 12 minutes of script
execution, with command-to-result savings of 8m 22s and 1m 02s, respectively,
against the measured 14m 03s full run.

Both replays loaded only `llama_68m` weights and reproduced all 3,423 recorded
target tokens. They returned identical per-prompt acceptance and bridge counters.
Acceptance was 27.89% for replay and 27.87% for live M1. On prompt 10, live M1
diverged from the target-only output, which also changed its proposal counts.

The extracted `target_references.toml` is ready for repeated runs in this
workspace. The target-only entry point passed staging validation. The timed
comparison recorded its references as part of the full M1 run.

[Machine-readable runtime comparison](../results/oracle_runtime_comparison.json)
