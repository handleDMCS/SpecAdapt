This note merges the spec with all fixes from my review. Three defaults are my picks, since they were still open: the stochastic check runs on P1, Vicuna uses the standard v1.1 prompt, and the Llama-3.1 pairs start only if spend is at most $15 after the first three pairs. The Alpaca slices also moved so that test rows are fixed first.

---

# SpecAdapt hand-off note

**Base:** commit `8c3dea71`. Code lives in `src/specadapt/` (library) and `src/demo/` (harness).

**Compute contract**

| Platform | Use | Limit |
| --- | --- | --- |
| Local (Windows, `uv`, Python 3.12) | Editing, job submission, statistics. No model runs. | |
| Kaggle T4 | Tests, all replay, all draft training and ablations | 30 GPU-hours/week |
| Modal A10, FP16 unquantized | 7B–8B target passes only | Hard cap $25.00 of $30 credit |

## 1. Tiers and pairs

| Tier | Configuration |
| --- | --- |
| B0 | Target only |
| B1 | Plain speculative decoding (P1) or naive `text_proposal` bridge (P2–P5), fixed `K = 3` |
| T0 | Continuation caches + `VocabularyBridge` + `EntropyKPolicy` |
| T1 | T0 + `OutputLogitAdapter` (logit bias + low-rank `lm_head` delta, rank 8 or 16, merged) + `LogisticKPolicy` |
| T2 | T0 + `LoRAAdapter` (rank 16 on `q,k,v,o,gate,up,down`, merged with `merge_and_unload()`) + `LogisticKPolicy` |

Total adapter overhead for T0–T2 must be at most 10% of one draft step.

| Pair | Draft → Target | Wave |
| --- | --- | --- |
| P1 | `llama_68m` → `vicuna_7b` (same tokenizer) | 1 |
| P2 | `llama_68m` → `qwen_7b` (primary) | 1 |
| P4 | `llama_160m` → `qwen_7b` | 1 |
| P3, P5 | `llama_68m`, `llama_160m` → `llama_31_8b` | 2: only if access is approved and ledger spend is at most $15.00 after Wave 1 stage 5 |

## 2. Work items (in order)

Each item is done when `uv run python -m demo.kaggle_run --cpu --no-cache tests/run_tests.py` passes.

| # | Work | Done when |
| --- | --- | --- |
| W1 | `decoding.py`: fall back to the Vicuna v1.1 prompt when `tokenizer.chat_template is None` | P1 prompt test passes |
| W2 | `bridge.py`: exact-piece matching and symmetric space markers | P1 proposals resolve as `direct` |
| W3 | `oracle.py`: FP16 option, `split in {"val", "test"}`, multi-target manifests, separate `recording_gpu` and `replay_gpu` | T4 replays an A10 oracle without a provenance error |
| W4 | `benchmark.py`: remove the `"T4" not in gpu` and `/kaggle/input` hard-codes | Runs from generic paths |
| W5 | `build_splits.py`, `test_splits.py`, committed `splits.toml` | Section 3 gate passes |
| W6 | Modal harness: `modal_app.py`, image pinned to the Kaggle package versions, Volume `/vol/specadapt`, Hugging Face secret, one function per stage, spend ledger | Dry run on Modal CPU |
| W7 | Recorders: batched (train) and `batch = 1` (test), resume by `prompt_id` | Resume test passes |
| W8 | `adaptive_k.py`: `EntropyKPolicy`, `LogisticKPolicy` | Unit tests pass |
| W9 | `alignment.py`: `OutputLogitAdapter`, `LoRAAdapter`, `M2Trainer` | Merged and unmerged logits differ by less than `1e-3` |
| W10 | Ratio bootstrap and breakdown reports | Synthetic test passes |
| W11 | Stochastic verification path (only needed for stage 6) | Synthetic test passes |

Vicuna v1.1 prompt: `"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"`

## 3. Dataset (`splits.toml`)

| Split | Math | Code | Instruction | Summarization | Count |
| --- | --- | --- | --- | --- | --- |
| `dev/` | 12 `prompt_ids` from `pair2_demo.toml` | | | | 12 |
| `test/core/` | `gsm8k_test[0:50]` | `humaneval_test[0:25]` + `mbpp_test[0:25]` | `alpaca_train[0:50]` | `xsum_test[0:50]` | 200 |
| `test/expansion/` | `gsm8k_test[50:100]` | `humaneval_test[25:50]` + `mbpp_test[25:50]` | `alpaca_train[50:100]` | `xsum_test[50:100]` | 200 |
| `smoke/` | `gsm8k_train[0:10]` | `mbpp_prompt[0:10]` | `alpaca_train[100:110]` | `xsum_validation[0:10]` | up to 40 |
| `train/fit/base/` | `gsm8k_train[10:370]` | `mbpp_train[0:360]` | `alpaca_train[110:470]` | `xsum_train[0:360]` | up to 1,440 |
| `train/val/` | `gsm8k_train[370:450]` | `mbpp_validation[0:80]` | `alpaca_train[470:550]` | `xsum_train[360:440]` | up to 320 |
| `train/fit/extra/` | `gsm8k_train[450:1530]` | `mbpp_train[360:]` | `alpaca_train[550:1630]` | `xsum_train[440:1520]` | up to 3,254 |

**Prompt formats (frozen)**

| Source | Format |
| --- | --- |
| GSM8K | `row["question"].strip()` |
| HumanEval | `row["prompt"].strip()` |
| MBPP | `f"{row['text'].strip()}\n\nExample test:\n{row['test_list'][0].strip()}"` |
| Alpaca | `f"{instruction}\n\nInput:\n{input}"` if input is non-empty, else `instruction` |
| XSum | `f"Summarize the following article in one sentence.\n\n{row['document'].strip()}"` |
| Draft side | `f"{prompt}\n\n"` + decoded committed target text |

**Build order (`build_splits.py`, deterministic)**

1. Move the 12 demo rows to `dev/math/` and remove them from `gsm8k_test`.
2. Render prompts. Drop empty ones and any with `len(draft_tok.encode(draft_prompt(p))) > min_ctx - 512`, where `min_ctx` is the smaller `max_position_embeddings` of the two drafters.
3. Exact dedup:
   - Hash with `sha256(re.sub(r"\s+", " ", text.strip().lower()))`.
   - Keep priority: `dev > test > validation > train > prompt`, with HumanEval above MBPP.
   - Ties keep the lowest row ID.
4. Shuffle each source split with `seed = 42`.
5. Fix the 400 test rows using the slices above.
6. Near-duplicate filter, on the first 2,500 rows of each non-test shuffled split:
   - Drop a row when `len(a & b) / len(a | b) >= 0.8` against any same-task test row.
   - `a` and `b` are `set(normalized_text.split())`.
7. Slice smoke, fit and val from what remains. Write row IDs, dataset revisions and hashes to `splits.toml`. Never edit it after commit.

**Gate (`test_splits.py`)**

- Zero overlap of `(dataset, split, row_id)` and zero hash overlap across all splits.
- Exact counts: `dev = 12`, `test/core = 200`, `test/expansion = 200`.
- Exact counts per non-code task: `smoke = 10`, `fit/base = 360`, `val = 80`, `fit/extra = 1080`.
- Code floors: `smoke >= 6`, `fit/base >= 340`, `val >= 70`.
- If a floor fails, stop and report.

## 4. Recording contract

| Item | Value |
| --- | --- |
| Decoding | Greedy, `max_new_tokens = 256`, `stop_reason in {"eos", "length"}` |
| `test/` (400) | `batch = 1`, to match live verification |
| `train/` (up to 5,014) | Batched 16–32, left-padded, length-sorted. Use for training and selection only, never for reported numbers. |
| JSONL line | `{prompt_id, task, source_dataset, prompt_sha256, input_ids, token_ids, text, max_new_tokens, stop_reason}` |
| Manifest | Target revision, `tokenizer_sha256`, package versions, `recording_gpu`, Git SHA |
| Resume | Append-only. On startup, skip every `prompt_id` already written. |

## 5. Modal workflow and budget

Estimates assume $1.32/hr, 28 tok/s at `batch = 1`, 250 tok/s batched, and 175 output tokens per prompt. None of these is measured. Stage 0 reads the real hourly rate and stage 1 measures the real speeds.

| Stage | Work | Est. cost (Wave 1 / all 5 pairs) | Gate to continue |
| --- | --- | --- | --- |
| 0. Asset sync (Modal CPU) | Download pinned models and datasets to the Volume | $0.10 / $0.15 | Manifests validated |
| 1. Smoke | `smoke/` with B0 and B1 at `batch = 1`, plus a 32-prompt batched speed probe | $0.50 / $0.75 | See below |
| 2. Test oracle | B0 on all 400 `test/` prompts per target | $1.85 / $2.75 | 400 valid records per target |
| 3. Train recording | Up to 5,014 `train/` prompts per target | $2.60 / $3.90 | If batched speed is under 250 tok/s, record `fit/base` + `val` only (1,760 prompts) |
| 4. Fit and select (Kaggle) | Section 6 | $0 | See below |
| 5. Live sweep | `test/core/` at `batch = 1`: B0, B1, T0, T1, T2 in one session per pair, mode order rotated per prompt | $6.45 / $10.60 | See below |
| 6. Stochastic check | P1, `test/core/math` (50), `T = 1`, 3 seeds | $1.85 | Only if W11 passes and `spent + 2.10 <= 25` |
| Cold starts (about 10%) | | $1.35 / $2.00 | |
| **Total** | | **$14.70 / $22.00** | |

**Gates**

- **Stage 1:**
  - Drop a pair if `(t_draft + t_bridge) / t_target > 0.5`.
  - B1 replay acceptance on Kaggle must match B1 live acceptance: `abs(replay - live) <= 0.005`. If not, stop and report.
  - Recompute the full projection from the measured speeds. It must be at most $25.
- **Stage 4:**
  - The best checkpoint of every tier advances, even if it does not beat the tier below.
  - Block a checkpoint only if overhead is above 10% or the merge test fails.
- **Stage 5:**
  - At least 99% of prompts must have a full token sequence equal to same-session B0.
  - Log the first divergent token index per prompt.
  - Report replay and live acceptance on `test/core/` side by side.

**Spend rules**

- Every Modal function appends `{stage, target, gpu_seconds, usd}` to `/vol/specadapt/ledger.jsonl`.
- Every paid script reads the ledger on startup and refuses to run if `spent + projected > 25.00`.
- Run order: Wave 1 stages 0–5, then Wave 2 stages 0–5, then stage 6.
- If the budget gets tight, cut in this order: stage 6, Wave 2, `fit/extra` recording.

## 6. Training and selection (Kaggle T4)

All selection uses `train/val/` only.

| Step | Config |
| --- | --- |
| Training text | `draft_prompt(prompt) + answer_text`, plus draft EOS only when `stop_reason == "eos"`. Re-tokenize with the draft tokenizer. Cross-entropy on answer tokens only (prompt labels set to `-100`). |
| T0 | Sweep the entropy threshold. Ablations: no continuation caches, and fixed `K = 3`. |
| T1 | Train the bias and low-rank delta, choose rank 8 or 16, merge into `lm_head` before timing. |
| T2 | LoRA rank 16, choose learning rate and epochs, `merge_and_unload()` before timing. |
| `LogisticKPolicy` | L2-regularized, features `entropy`, `top1_prob`, `top1_top2_margin`, `cache_hit`. Labels come from one replay pass of the trained drafter over `fit/base`. Refit per tier, and tune the threshold on val. |
| Learning curve | 25%, 50% and 100% of `fit/base`. Use `fit/extra` only if val acceptance gains at least 1.0 percentage point from 50% to 100%. With `fit/extra`, set code sampling weight to 2x–3x and stop early if `val/code` degrades. |
| Frozen cache | Default 16 prompts (first 4 per task of `fit/base`). Ablation at 160 and 1,440. |
| Leave-one-task-out | T1 and T2: train on 3 tasks, replay on the fourth. |
| Kaggle limits | Learning curve, cache ablation and leave-one-task-out run on P2 only. Other pairs get the final selected configuration. At most 6 hyperparameter configurations per tier per pair. |
| Test lock | Commit `checkpoints_frozen.toml` (hash of every selected checkpoint) before any `test/` replay. Test scripts refuse to run without it. |

## 7. Evaluation

| Metric | Data | Source |
| --- | --- | --- |
| Acceptance rate, accepted tokens per round | All 400 `test/` | Kaggle replay against the stage 2 oracle |
| Speedup, tok/s, adapter overhead (ms/token), exact match with B0 | `test/core/` (200) | Stage 5 |
| 95% confidence intervals | `prompt_id` | Paired bootstrap, at least 5,000 resamples: resample IDs, then `sum(accepted) / sum(proposed)` and tier-minus-baseline differences |
| Breakdowns | | Per task, HumanEval separate from MBPP, 128-token prefix against 256, and `stop_reason == "eos"` rate per task |

## 8. Guardrails

1. **Paid jobs:** the agent never launches a Modal GPU job. It prints the exact command and the projected cost, then stops. The user runs it.
2. **No deletes:** no delete or overwrite of anything on the Volume, in `results/`, or in recorded JSONL without the user's approval.
3. **Tests first:** the test command in section 2 must pass before any GPU job is proposed.
4. **Secrets:** never print, log or commit `.env`, `HF_TOKEN`, `KAGGLE_API_TOKEN` or Modal tokens.
5. **Replay is not timing:** replay records never contain `tokens_per_second`, `speedup_vs_target` or `exact_match`. Those come from stage 5 only.
6. **Audit trail:** every result artifact logs the Git SHA, `asset_plan.toml` revisions, `recording_gpu` and `replay_gpu`.
7. **On any failed gate:** stop and report. Do not work around it.