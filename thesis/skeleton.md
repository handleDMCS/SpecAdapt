# Thesis skeleton

Snapshot: 2026-11-24, repo at `1812446`. Language: **English**. First PDF due 2026-12-06, final PDF due 2027-01-04.

This file is the content tree for the thesis and the only source of facts for whoever writes the PDF. Format and build rules are in the `thesis-pdf` skill (`.claude/skills/thesis-pdf/`).

## How to use this file

1. Write only what a section's **Facts** and **Evidence** support. Open the evidence file and re-read the number before putting it in the PDF.
2. Every section has a status:
   - `READY`: evidence exists; write it.
   - `PARTIAL`: write the supported part; leave the rest as a visible `[PENDING: ...]` note in the draft.
   - `BLANK`: no evidence yet. Keep the heading, write one sentence saying what will go here, and nothing else. Do not estimate, extrapolate or borrow numbers from other pairs.
3. Never copy a number from a note when a result file exists. Notes say where to look; result files are the evidence.
4. Citations: only works listed under [Literature](#literature) may be cited, and each still needs its bibliographic record checked against the source. Do not cite from memory.
5. `results/` and `build/` are **git-ignored**. The evidence files below exist only on this machine. Back them up before the first PDF.
6. Where notes disagree, `training_notes/training.md` is the current plan. `notes/experiment.md` is the earlier plan (4-bit targets on Kaggle, `max_new_tokens = 128`) and describes only the October/21 Nov demos.

## Structure

The thesis follows the standard English thesis structure, set by the author on 2026-11-24. The university's fixed pages sit inside it.

| Part | In this thesis |
| --- | --- |
| Front matter | cover, title page, authorship declaration, supervisor's approval, acknowledgement, abstract, table of contents, lists of figures, tables and algorithms, abbreviations |
| Introduction | Chapter 1 |
| Literature review | Chapter 2, a full chapter |
| Methodology | Chapter 3, a full chapter: the method (3.2) and the experimental design (3.3) |
| Results | Chapter 4 (baselines and logit alignment) and Chapter 5 (vocabulary bridge and decoding cost) |
| Discussion | Chapter 6 |
| Conclusion and future work | Chapter 7 |
| References | IEEE style, before the appendices |
| Appendices | A to D |

Results chapters state what was measured and what the tables show. Interpretation, relation to prior work and limits go in the Discussion. Keep the two apart.

## Front matter (to fill by the author)

Do not guess any of these. Leave the template placeholder if unknown.

| Field | Value | Source |
| --- | --- | --- |
| Thesis title | `[TO CONFIRM]`. Working title in the notes: "SpecAdapt: Lightweight Draft-Side Adaptation for Speculative Decoding" | `notes/idea.md` |
| Author, student ID | `[TO FILL]` | |
| Major (exact programme name) | `[TO FILL]` | |
| Supervisor, co-supervisor | `[TO FILL]` | |
| University block | As printed in the format documents: "VIETNAM NATIONAL UNIVERSITY, HANOI / UNIVERSITY OF ENGINEERING AND TECHNOLOGY". `[CONFIRM this is the right institution]` | `thesis/sample/format.md` links |
| Defense year on the cover | `[TO FILL]` | |
| Acknowledgement, funding line | `[TO FILL]` | |
| Related publications | None recorded in the repo. Omit the page unless the author supplies one. | |

## Results at a glance (verified today)

All acceptance numbers are **oracle replay on validation prompts**, greedy, fixed `K = 3`, `max_new_tokens = 256`, target tokens accepted ÷ target tokens proposed. None is a test-set number and none is a measured speed.

| Result | Value | Evidence |
| --- | --- | --- |
| P2 untrained drafter (B1), 320 val prompts | 18,316 / 98,317 = 18.63% | `results/runs/20261124T052050221490Z/summary.json` → `tiers.B1` |
| P2 LoRA (T2), same prompts | 24,772 / 78,735 = 31.46%, **+12.83 pp** | same file → `tiers.T2`, `t2_gain_pp` |
| P2 output adapter (T1), same prompts | 20,836 / 89,120 = 23.38%, **+4.75 pp** | same file → `tiers.T1`, `t1_gain_pp` |
| P2, 80-prompt subset: vocabulary bridge + caches (M1) on the untrained drafter | acceptance 18.73% → 20.28%; tokens per target call 1.512 → 1.564 | `results/runs/20261124T063633692969Z/summary.json` → `arms` |
| P2, 80-prompt subset: M1 on top of LoRA | tokens per target call 1.824 (T2) vs 1.821 (T2 + M1); draft steps 24,514 → 19,662 | same file |
| Stage 1 live check, 40 smoke prompts, A10 FP16 | live B1 acceptance P1 25.2%, P2 20.2%, P4 26.4%; draft-to-target step ratio 0.057 / 0.062 / 0.231 | `results/modal/20261123T141520817523Z/smoke-{vicuna_7b,qwen_7b}.json` → `pairs` |

---

## Chapter 1. Introduction

### 1.1 Motivation · `PARTIAL`

- **Content:** large language models decode one token per forward pass; speculative decoding (SpD) lets a small draft model propose `K` tokens that the target verifies in one pass; the gain depends on how many proposals are accepted; with different tokenizers or mismatched distributions acceptance collapses.
- **Facts available from this project (use as the concrete motivation):** with an untrained 68M drafter and a 7B target using a different tokenizer (P2), only 18.63% of proposed tokens are accepted at `K = 3` on 320 validation prompts.
- **Evidence:** `results/runs/20261124T052050221490Z/summary.json`; problem framing in `notes/idea.md` ("Core Idea").
- **Pending:** every general claim about SpD (speed-ups reported in the literature, why heterogeneous pairs matter in deployment) needs a citation from the [Literature](#literature) list. Nothing in the repo supports those sentences yet.

### 1.2 Problem statement and research question · `READY`

- **Content:** draft-side-only adaptation; the target is never modified; adapters must be cheap at run time.
- **Research question (verbatim from the notes):** "How much of the acceptance-rate and speedup improvement of adaptive speculative decoding can be recovered using training-free or very lightweight draft-side adaptation?"
- **Design constraint:** adapter overhead at most 10% of one draft step.
- **Evidence:** `notes/idea.md` ("Research Hypothesis", "Proposed Adapter"); `training_notes/training.md` §1.
- **Framing decision for the author (do not resolve silently):** the notes hypothesise that training-free and lightweight tiers recover most of the gain. The P2 validation result so far says otherwise: T1 recovers 4.75 of T2's 12.83 points, and M1 without a seed cache adds 1.55 points of acceptance on the untrained drafter and no extra tokens per target call on top of T2. State the question as open and report what was measured. Note also that T1 is trained offline with gradients; only T0 is training-free. §6.1 is where the thesis answers the question.

### 1.3 Contributions · `PARTIAL`

Claim only what exists:

1. A draft-only **oracle replay** evaluation method: record the target's greedy output once, then measure acceptance for any drafter without loading the target. `READY`. Evidence: `src/specadapt/oracle.py`, `src/specadapt/decoding.py` (`generate(..., oracle=...)`), `notes/oracle-replay.md`.
2. A **vocabulary bridge with continuation caches** (M1) for cross-tokenizer SpD. `READY` as an implementation; results are limited to §5.1 and §5.2.
3. **Draft-side logit alignment** in two sizes (T1 output adapter, T2 LoRA), merged into the drafter so they add no run-time cost. `READY` for P2 validation.
4. **Adaptive draft length** (M3). `BLANK`: `src/specadapt/adaptive_k.py` does not exist. Do not list as a contribution until block 5 of `training_notes/iteration.md` has run.
5. A reproducible, low-cost experiment pipeline (pinned revisions, provenance in every result, fire-and-forget Kaggle jobs). `READY`. Evidence: `asset_plan.toml`, `src/demo/kaggle_run.py`, `src/demo/kaggle_runtime.py`, `training_notes/progress.md` Log.

### 1.4 Thesis overview · `READY`

One paragraph per chapter, written last from the final chapter list.

---

## Chapter 2. Literature review

A full chapter, and all of it needs literature: the repo holds code and measurements, not a literature review. Sections marked `PARTIAL` can describe a mechanism from the code but still need citations.

### 2.1 Autoregressive decoding and its cost · `BLANK` (needs citations)



### 2.2 Speculative decoding · `PARTIAL`

- **Supported by the code (can be described exactly):** the greedy draft-then-verify round. The drafter proposes up to `K` tokens; the target scores them in one pass; the longest prefix matching the target's own greedy choice is accepted; the target's token at the first mismatch is appended; both KV caches roll back.
- **Evidence:** `src/specadapt/decoding.py` (`accepted_prefix`, `CachedModel`, `generate`); `notes/psuedo_code.md` (Algorithm 1: one speculative round).
- **Pending:** the original SpD formulation, stochastic (sampling) verification and the standard speed-up model need citations. The stochastic path is not implemented here (W11 not started, `training_notes/progress.md`).

### 2.3 Heterogeneous pairs: tokenizer and distribution mismatch · `PARTIAL`

- **Supported:** which pairs share a tokenizer (P1) and which do not (P2, P4); why a draft token sequence cannot be verified directly by a target with another vocabulary.
- **Evidence:** `first_principle_notes/vocab_bridge/vocabulary-bridge.md`; `asset_plan.toml`.
- **Pending:** citations for prior cross-vocabulary SpD.

### 2.4 Closest prior work: OmniDraft · `BLANK` (needs the paper)

- The notes name it as the main reference: Ramakrishnan et al., "OmniDraft: A Cross-vocabulary, Online Adaptive Drafter", arXiv:2507.02659 (`notes/idea.md`).
- Nobody has recorded its method details or numbers in this repo. Read the paper and fill this section from it. Do not write a numerical comparison: the setups differ and no matched run exists.

### 2.5 Parameter-efficient adaptation and distillation · `BLANK` (needs citations)

LoRA, low-rank output heads, knowledge distillation from teacher outputs.

### 2.6 Adaptive draft length · `BLANK` (needs citations)



---

## Chapter 3. Methodology

A full chapter in two parts: what the method is (3.2) and how it is evaluated (3.3).

### 3.1 Overview and tiers · `READY`

- **Content:** three modules on the draft side (M1 vocabulary bridge, M2 logit alignment, M3 adaptive `K`) and the run ladder B0, B1, T0, T1, T2.
- **Facts:** tier table in `training_notes/training.md` §1. B1 = plain SpD for the same-tokenizer pair, naive text bridge for cross-tokenizer pairs, fixed `K = 3`.
- **Figure to draw:** system diagram (drafter → bridge → target verification → commit). No such figure exists; draw from `first_principle_notes/vocab_bridge/pipeline.md`.

### 3.2 The SpecAdapt method

#### 3.2.1 Notation and prompt formats · `READY`

- Target sees its own chat template; the drafter sees `"{prompt}\n\n"` followed by the decoded committed target text, re-encoded every round.
- Vicuna uses the v1.1 system prompt when the tokenizer has no chat template.
- **Evidence:** `src/specadapt/decoding.py` (`draft_prompt`, `target_prompt`, line 126); `training_notes/training.md` §2 (Vicuna prompt) and §3 (prompt formats per dataset).

#### 3.2.2 Baseline: naive text bridge (B1) · `READY`

- Decode the draft proposal to text with its left context, re-tokenize with the target tokenizer, keep up to `K` target tokens.
- **Evidence:** `src/specadapt/bridge.py` (`text_proposal`, `decode`); `first_principle_notes/vocab_bridge/text-proposal.md`.

#### 3.2.3 M1: vocabulary bridge and continuation caches · `READY`

Subsections and their sources:

| Part | What it does | Evidence |
| --- | --- | --- |
| Direct overlap map | Draft token → target token when the surface piece is identical (ASCII pieces) | `src/specadapt/bridge.py` `VocabularyBridge.__init__`; `first_principle_notes/vocab_bridge/vocabulary-bridge.md` |
| Contextual n-gram trie | Remembers draft-sequence → target-sequence conversions, keyed with boundary context | `TokenTrie`, `VocabularyBridge.map`; `first_principle_notes/vocab_bridge/token-trie.md` |
| Continuation cache | Longest-suffix lookup in the prompt and in already verified output; a hit proposes tokens without calling the drafter; proposals are still verified by the target | `ContinuationCache`; `first_principle_notes/vocab_bridge/continuation-caches.md` |
| Frozen seed cache | Optional read-only continuations built from target outputs on training questions | `FrozenContinuationCache`, `README.md` "Pair 2 M1 demo" |
| One full round | Order of the lookups and the fallback | `first_principle_notes/vocab_bridge/pipeline.md`; `notes/psuedo_code.md` |

- **Algorithm listing:** Algorithm 1 in `notes/psuedo_code.md` can be typeset as the thesis algorithm.
- **Caveat to state:** work item W2 (exact-piece matching resolving P1 proposals as `direct`) is implemented but its P1 check is still pending (`training_notes/progress.md`).

#### 3.2.4 M2: logit alignment · `READY`

- **Training signal:** the target's own greedy answer text for each training prompt (recorded once), not dataset reference answers. Training text is `draft_prompt(prompt) + answer_text`, re-tokenized with the draft tokenizer; cross-entropy on answer tokens only (prompt positions masked with `-100`); draft EOS appended only when the target stopped at EOS.
- **T1, output logit adapter:** frozen drafter; a learned vocabulary bias plus a low-rank delta on the LM head, `logits = W h + U(D h) + b`; rank 8 or 16; merged into a new LM-head weight (input embeddings stay untouched when tied).
- **T2, LoRA:** low-rank updates on all seven projections per layer (`q, k, v, o, gate, up, down`), rank 16, no scaling factor (`W + U D`), `U` initialised to zero; merged before replay.
- **Optimiser:** AdamW, no weight decay, linear warm-up over 5% of steps then cosine decay, FP32, length-sorted batches capped at 16 sequences / 16,000 tokens, seed 42.
- **Merge check:** merged and unmerged logits must differ by less than `1e-3` or the job aborts. Measured on P2: 4.2e-05 (T2), 3.8e-05 (T1).
- **Adapter sizes on `llama_68m` (counted from the saved files):** T2 28 tensors, 565,248 parameters; T1 3 tensors, 556,288 parameters.
- **Evidence:** `src/specadapt/alignment.py` (`LowRankLinear`, `OutputLogitAdapter`, `LoRAAdapter`, `encode_example`, `M2Trainer.fit`); `training/kaggle/train_pair.py`; `training/jobs/p2.toml`; `results/runs/20261124T052050221490Z/artifacts/{t1,t2}.safetensors` and `summary.json` → `tiers.*.merge_error`; `training_notes/training.md` §6.

#### 3.2.5 M3: adaptive draft length · `BLANK`

- Planned: entropy threshold policy (T0) and an L2-regularised logistic policy on `entropy`, `top1_prob`, `top1_top2_margin`, `cache_hit` (T1, T2), selected by modeled cost per token, not acceptance.
- Not implemented: `src/specadapt/adaptive_k.py` does not exist. Plan: `training_notes/iteration.md` block 5; `training_notes/training.md` §6.
- Until it runs, describe M3 only in Future Work.

#### 3.2.6 Run-time cost of the adapters · `PARTIAL`

- **Supported:** T1 and T2 are merged, so the drafter's architecture and step cost are unchanged by construction (merge check above).
- **Pending:** measured bridge overhead for the current setup against the 10% budget. The only bridge timing on file is the October demo (§5.2), where bridge time per draft step was 41% (baseline) and 28% (M1) of a draft step, far above the budget. State this as an open issue, not as met.

### 3.3 Experimental design

#### 3.3.1 Model pairs · `READY`

| Pair | Drafter → Target | Role | Status |
| --- | --- | --- | --- |
| P1 | `JackFram/llama-68m` → `lmsys/vicuna-7b-v1.5` | same tokenizer, control | oracle recorded; training not run |
| P2 | `JackFram/llama-68m` → `Qwen/Qwen2.5-7B-Instruct` | cross-tokenizer, primary | trained and replayed on validation |
| P4 | `JackFram/llama-160m` → `Qwen/Qwen2.5-7B-Instruct` | drafter size | oracle recorded; training not run |
| P3, P5 | drafters → `meta-llama/Llama-3.1-8B-Instruct` | cross-tokenizer | target not recorded (Wave 2, conditional) |

- **Evidence:** `asset_plan.toml` (repositories and pinned revisions); `training_notes/training.md` §1; `training_notes/iteration.md` header.
- Write P3 and P5 as planned, not as evaluated.

#### 3.3.2 Datasets and splits · `READY`

- Four tasks: math (GSM8K), code (HumanEval, MBPP), instruction (Alpaca), summarization (XSum).
- Splits: `dev` 12, `smoke` up to 40, `train/fit/base` up to 1,440, `train/val` up to 320, `test/core` 200, `test/expansion` 200. Recorded sizes per target: 1,440 fit, 320 validation, 400 test.
- Build order: dedup by normalised hash, seed-42 shuffle, fix test rows first, near-duplicate filter (Jaccard ≥ 0.8 against same-task test rows), then slice.
- **Evidence:** `training_notes/training.md` §3; `training_notes/data_layout.md`; `training/build_splits.py`; `training/dataset/splits.toml`; oracle manifests under `results/modal/20261123T154925641802Z/oracle-dump/{qwen_7b,vicuna_7b}/`.
- **Caveats to state:** the split gate `test_splits.py` is missing (`training_notes/progress.md`, W5). The trainer does check at load time that fit and validation share no prompt ID (`src/specadapt/training_inputs.py`, `load_target`). `fit/extra` (3,254 prompts) was not recorded.

#### 3.3.3 Oracle recording and replay · `READY`

- **Recording:** each target generates greedily once per prompt, FP16 unquantised, `max_new_tokens = 256`, on a Modal A10; test prompts at batch 1, training prompts batched. Each record stores input IDs, output token IDs, text and stop reason; manifests store the target revision, tokenizer fingerprint, package versions, GPU and Git SHA.
- **Replay:** on a Kaggle T4 only the drafter is loaded. Each round's proposal is checked against the recorded output. The job aborts if the replayed sequence ever differs from the record, so the output is identical by construction and only the acceptance counts vary.
- **What replay does not measure:** wall-clock speed, speed-up, exact match with a live target. Those need the live sweep (§5.4).
- **Evidence:** `training_notes/training.md` §4 and §8 rule 5; `notes/oracle-replay.md`; `src/specadapt/oracle.py`; `training/kaggle/train_pair.py` (`replay`); `modal/overnight_oracles.py`, `modal/stage2_test_oracle.py`.

#### 3.3.4 Training and selection protocol · `READY` for P2

- Grid for P2: T2 rank 16 × learning rates {1e-4, 3e-4, 1e-3}, scored at epochs 2 and 4 of a 4-epoch schedule; T1 ranks {8, 16} × learning rates {1e-3, 3e-3, 1e-2}, 2 epochs. At most 6 configurations per tier.
- Selection metric: teacher-forced top-1 agreement with the target's answer tokens on the 320 validation prompts. The selected checkpoint is then merged and replayed on the same validation prompts.
- **State plainly:** selection and the reported acceptance use the same validation prompts. The test oracle has not been read.
- **Evidence:** `training/jobs/p2.toml`; `training/kaggle/train_pair.py` (`grid`, `fit_tier`); `results/runs/20261124T052050221490Z/summary.json` → `trials`.

#### 3.3.5 Metrics · `READY`

| Metric | Definition | Where computed |
| --- | --- | --- |
| Acceptance rate | accepted target tokens ÷ proposed target tokens, summed over prompts | `train_pair.py` `acceptance` |
| Tokens per target call | generated tokens ÷ verification rounds | `training/kaggle/m1_combo.py` `rates` |
| Modeled cost per token | (rounds + c · draft steps) ÷ generated tokens, with c the measured draft-to-target step ratio (P2: 0.0623) | same; c from the stage 1 smoke file |
| Top-1 agreement | share of the target's answer tokens the drafter ranks first under teacher forcing | `alignment.py` `M2Trainer.evaluate` |

- Confidence intervals: planned as a paired bootstrap over prompt IDs (`training_notes/training.md` §7). `BLANK` for the current results: none has been computed for the 24 Nov runs. Per-prompt counts for the 80-prompt run are saved (`results/runs/20261124T063633692969Z/artifacts/*.json`); the P2 full run saved totals and per-task counts only.

#### 3.3.6 Infrastructure and reproducibility · `READY`

- Kaggle T4 (16 GB), pinned Docker image digest, `torch 2.10.0+cu128`, `transformers 4.57.1`, `accelerate 1.11.0`, `sentencepiece 0.2.1`; replay in FP16 with SDPA attention, batch 1.
- Modal A10 for target passes; recorded spend $3.52 for asset sync, smoke and overnight recording.
- Every result records Git SHA, model and dataset revisions, tokenizer fingerprint, recording GPU and replay GPU.
- Test suite: 87 CPU tests passed on Kaggle (run `20261124T063355273658Z`).
- **Evidence:** `kernel-metadata.json`; `training_notes/iteration.md` "GPU config"; `results/runs/20261124T052050221490Z/summary.json` → `gpu_config`, `provenance`; `training_notes/progress.md` ("Spend and open checks"); `results/submission-20261124T063355273658Z.json`; `tests/`.
- **Caveat:** the P2 full run cites Git `f5f1767` with `git_dirty: true`. The only uncommitted file was a notes file (`training_notes/babysitter.md`); the trainer code is unchanged from `7004468`.

---

## Chapter 4. Results: baselines and logit alignment

Reports the numbers and what the tables show. Interpretation belongs to Chapter 6.

### 4.1 Replay validity and cost ratios (stage 1) · `PARTIAL`

- **Facts (live, Modal A10, FP16, 40 smoke prompts, `K = 3`):**

| Pair | Live B1 acceptance | Draft-to-target step ratio | Prompts where live B1 output equals target-only output |
| --- | --- | --- | --- |
| P1 | 2,582 / 10,250 = 25.19% | 0.0575 | 39 / 40 |
| P2 | 2,454 / 12,146 = 20.20% | 0.0623 | 34 / 40 |
| P4 | 2,878 / 10,921 = 26.35% | 0.2315 | 36 / 40 |

- Target speed on the A10: 28.3 tok/s (Qwen) and 28.7 tok/s (Vicuna) at batch 1; 199.5 and 79.0 tok/s batched.
- **Evidence:** `results/modal/20261123T141520817523Z/smoke-qwen_7b.json`, `smoke-vicuna_7b.json`.
- **Pending evidence:** the replay-versus-live acceptance deltas (P1 0.00112, P2 0.00183, P4 0.00053, gate ≤ 0.005) are stated in `training_notes/progress.md`, but the result file `results/submission-20261123T144002663579Z.json` is **not on disk**. Re-collect it (`uv run python -m demo.kaggle_run --resume 20261123T144002663579Z`) or cite the numbers as "recorded in the project log" until it is.
- **Report here, discuss in §6.5:** live B1 output differed from target-only output on 1 to 6 of 40 prompts. The stage 5 gate asks for 99% (`training_notes/iteration.md`, "Not in this plan"). This is an open issue for the live sweep.

### 4.2 Untrained baseline (B1) · `PARTIAL`

- **P2, 320 validation prompts:** 18.63% overall; math 7,000 / 31,948 = 21.91%, code 6,006 / 36,951 = 16.25%, instruction 4,118 / 22,468 = 18.33%, summarization 1,192 / 6,950 = 17.15%.
- **Evidence:** `results/runs/20261124T052050221490Z/summary.json` → `tiers.B1`.
- **P1, P4 on 40 validation prompts each (untrained replay):** 28.2% and 24.3% (P2 on the same 40: 18.7%). Evidence: `results/submission-20261124T014249246739Z.json`, summarised in `training_notes/progress.md`.
- **Pending:** B1 for P1 and P4 on all 320 validation prompts (comes with their training runs).

### 4.3 Logit alignment on P2 (T1, T2) · `READY`

Headline table (320 validation prompts, `K = 3`):

| Tier | Selected config | Accepted / proposed | Acceptance | Gain over B1 |
| --- | --- | --- | --- | --- |
| B1 | untrained | 18,316 / 98,317 | 18.63% | |
| T1 | rank 16, lr 1e-2, 2 epochs | 20,836 / 89,120 | 23.38% | +4.75 pp |
| T2 | rank 16, lr 1e-3, epoch 4 | 24,772 / 78,735 | 31.46% | +12.83 pp |

Per task:

| Tier | math | code | instruction | summarization |
| --- | --- | --- | --- | --- |
| B1 | 21.91% | 16.25% | 18.33% | 17.15% |
| T1 | 29.24% (8,173 / 27,956) | 22.22% (7,257 / 32,659) | 19.17% (4,162 / 21,707) | 18.30% (1,244 / 6,798) |
| T2 | 39.85% (9,634 / 24,173) | 32.91% (9,152 / 27,806) | 22.39% (4,580 / 20,452) | 22.30% (1,406 / 6,304) |

Sweep (top-1 agreement on validation answer tokens, 62,972 tokens):

| T2, rank 16 | epoch 2 | epoch 4 |
| --- | --- | --- |
| lr 1e-4 | 0.5049 | 0.5186 |
| lr 3e-4 | 0.5503 | 0.5638 |
| lr 1e-3 | 0.5889 | **0.5995** |

| T1, 2 epochs | lr 1e-3 | lr 3e-3 | lr 1e-2 |
| --- | --- | --- | --- |
| rank 8 | 0.4462 | 0.4695 | 0.4874 |
| rank 16 | 0.4546 | 0.4802 | **0.4995** |

- Run time on one T4: 29.8 minutes in total (B1 replay 6.6, T2 training 6.7, T2 replay 5.3, T1 training 3.7, T1 replay 6.2, preflight 1.3).
- **Evidence:** `results/runs/20261124T052050221490Z/summary.json` (`tiers`, `trials`, `minutes_per_phase`); figure `build/red-flag/20261124T052050221490Z.png` (regenerate with `uv run python -m training.training_red_flag 20261124T052050221490Z --visual`); smoke run `results/runs/20261124T050647634040Z/summary.json` (B1 gate 2,610 / 13,938 matched exactly).
- **Report the numbers and the sweep only.** Interpretation goes in §6.2.
- **Do not claim:** that these are test results; that 31.46% is the best achievable; any speed-up.
- **Untrained top-1 agreement reference:** the only value on file is 0.433 from the 40-prompt smoke after 10 training steps (`results/runs/20261124T050647634040Z/summary.json`), which is not a true untrained measurement. Do not print it as the untrained baseline.

### 4.4 Other pairs: P1 and P4 · `BLANK`

Planned: `training/jobs/p1.toml`, `training/jobs/p4.toml` (`training_notes/iteration.md` block 6). No training result exists.

### 4.5 Held-out test results · `BLANK`

Planned: freeze selected checkpoints (`checkpoints_frozen.toml`), then replay the 400 test prompts per pair (`training_notes/training.md` §6 "Test lock", §7). The test oracle is recorded (`results/modal/20261123T154925641802Z/oracle-dump/*/test/`) and has not been read by any training or selection job.

### 4.6 Ablations · `BLANK`

Planned on P2: learning curve (25%, 50%, 100% of fit), leave-one-task-out, frozen-cache size, T0 without caches (`training_notes/training.md` §6). None has run.

---

## Chapter 5. Results: vocabulary bridge and decoding cost

Same rule as Chapter 4: report, do not interpret.

### 5.1 Combining the vocabulary bridge with LoRA (P2) · `READY`

80 validation prompts (first 20 per task), `K = 3`, no seed cache, 14,976 generated tokens per arm:

| Arm | Accepted / proposed | Acceptance | Rounds | Tokens per target call | Draft steps | Rounds served by cache | Modeled cost per token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1, baseline bridge | 5,092 / 27,186 | 18.73% | 9,905 | 1.512 | 29,686 | 0 | 0.785 |
| B1 + M1 | 5,421 / 26,729 | 20.28% | 9,578 | 1.564 | 23,397 | 1,773 (18.5%) | 0.737 |
| T2, baseline bridge | 6,793 / 21,929 | 30.98% | 8,209 | 1.824 | 24,514 | 0 | 0.650 |
| T2 + M1 | 6,776 / 22,411 | 30.24% | 8,224 | 1.821 | 19,662 | 1,636 (19.9%) | 0.631 |

- **Evidence:** `results/runs/20261124T063633692969Z/summary.json` (`arms`, `vs_b1_baseline`, per task under `arms.*.per_task`); per-prompt counts in `artifacts/{B1-baseline,B1-m1,T2-baseline,T2-m1}.json`; job `training/jobs/p2-m1-combo.toml`; script `training/kaggle/m1_combo.py`; Git `1812446`, clean.
- **What the table shows (interpretation goes in §6.3):** M1 raises tokens per target call for the untrained drafter (1.512 → 1.564) but not for the trained one (1.824 vs 1.821). With either drafter it removes about a fifth of drafter calls, which lowers the modeled cost (0.650 → 0.631 for T2).
- **Limits to state:** 80 prompts, no confidence interval yet, no seed cache (it is built from GSM8K training answers, the same source as the math validation prompts), modeled cost is built from counts and a ratio measured on another GPU, not a timing.

### 5.2 Early M1 demonstration (October setup) · `READY`, different setup

- P2, 12 GSM8K test prompts, 4-bit NF4 target on a T4, `K = 3`, `max_new_tokens = 128`, warm frozen cache from 16 training questions.
- Acceptance 21.88% (580 / 2,651) → 23.82% (623 / 2,615); paired mean gain 1.92 pp, bootstrap 95% CI [0.73, 3.15]; 8.94 → 9.36 tokens/s (ratio 1.048); all 12 outputs equal to target-only decoding.
- Replay check on the same 12 prompts with `max_new_tokens = 512`: replay acceptance 27.89% vs live 27.87%; all 3,423 target tokens reproduced.
- **Evidence:** `results/pair2_comparison.json`; `results/oracle_runtime_comparison.json`; `notes/oracle-replay.md` ("Validation and runtime comparison"); `README.md` "Pair 2 M1 demo".
- **Keep separate from §4.3 and §5.1:** different precision (4-bit), prompt set, token limit and cache. Do not put these numbers in the same table. It is the only measured wall-clock comparison in the project so far, on 12 prompts.

### 5.3 Adaptive draft length results · `BLANK`

Depends on §3.2.5.

### 5.4 Live speed-up and adapter overhead · `BLANK`

Planned: stage 5 live sweep on `test/core` (`training_notes/training.md` §5, §7). Nothing measured for trained drafters. The thesis may not state a speed-up for T1 or T2 until this runs.

---

## Chapter 6. Discussion

Interprets Chapters 4 and 5. Every statement points back to a table in those chapters. A cause that was not tested is left out, or written as a hypothesis and labelled as one.

### 6.1 Answer to the research question · `PARTIAL`

- **Supported (P2, validation prompts, replay):**
  - LoRA (T2) gains 12.83 pp of acceptance. The output adapter (T1) gains 4.75 pp, about 37% of T2's gain.
  - The two adapters are nearly the same size (565,248 and 556,288 parameters), so adapter size does not explain the difference. Placement and training schedule both differ (T2: all projections, 4 epochs; T1: LM head only, 2 epochs), and the runs do not separate the two.
  - The training-free module (M1, no seed cache) adds 1.55 pp of acceptance on the untrained drafter, on 80 prompts.
- **Author's framing decision:** see §1.2. The measured numbers do not support "most of the gain without gradient-based training". Write the answer as measured.
- **Pending:** the speed-up half of the question (§5.4), confirmation on held-out test prompts (§4.5), other pairs (§4.4).
- **Evidence:** §4.3, §5.1, §3.2.4 (adapter sizes).

### 6.2 What the acceptance results show · `READY`

Patterns visible in the §4.3 tables:

- **Gain by task over B1, in pp.** T2: math +17.94, code +16.66, instruction +4.07, summarization +5.15. T1: math +7.32, code +5.97, instruction +0.85, summarization +1.15. These are differences of the per-task rates in §4.3; recompute them from the counts before printing.
- **Sweep.** For T2, agreement rises with the learning rate at both checkpoints and from epoch 2 to epoch 4 at every learning rate. For T1, it rises with the learning rate at both ranks, and rank 16 is above rank 8 at every learning rate.
- **Grid edge.** The selected learning rate is the largest one tried in both tiers, so the grid did not bracket the best value. A larger learning rate was not tried.
- **Do not explain** why math and code gain more than instruction and summarization. No experiment tested a cause.
- **Evidence:** `results/runs/20261124T052050221490Z/summary.json` → `tiers`, `trials`.

### 6.3 Accepted tokens, drafter calls and modeled cost · `PARTIAL`

- **Supported (P2, 80 validation prompts, §5.1):**
  - With the trained drafter, M1 leaves tokens per target call unchanged (1.824 without, 1.821 with) and removes about 20% of drafter steps (24,514 → 19,662). Modeled cost per token falls from 0.650 to 0.631.
  - With the untrained drafter, M1 raises tokens per target call from 1.512 to 1.564 and removes about 21% of drafter steps (29,686 → 23,397). Modeled cost falls from 0.785 to 0.737.
  - Training alone (B1 to T2, baseline bridge) moves modeled cost from 0.785 to 0.650.
- **A consequence of the cost model, not a result:** the weight of a drafter step is 0.0623 for P2 and 0.2315 for P4 (§4.1), about 3.7 times larger, so the same saving in drafter steps would count for more on P4. No P4 run exists.
- **Pending:** measured time. Modeled cost counts target calls and drafter steps only and leaves out bridge and cache time. The one timing on file (§5.2, October setup) put bridge time at 41% (baseline) and 28% (M1) of a draft step.
- **Evidence:** `results/runs/20261124T063633692969Z/summary.json` → `arms`; step ratios in `results/modal/20261123T141520817523Z/smoke-qwen_7b.json` → `pairs`.

### 6.4 Relation to prior work · `BLANK` (needs Chapter 2)

Compare with the literature only after Chapter 2 is written from the sources. No numerical comparison with OmniDraft: the setups differ and no matched run exists.

### 6.5 Threats to validity · `READY`

Each item is true of the current evidence:

1. Validation prompts were used both to select configurations and to report acceptance; held-out test numbers are pending.
2. Validation and training prompts come from the same four dataset families; transfer to unseen task types is untested.
3. One pair (P2) is trained; generality across pairs is untested.
4. Acceptance is a proxy for speed. No live timing exists for the trained drafters.
5. Replay follows the recorded greedy trajectory of an FP16 target on an A10; a live target on other hardware may diverge (§4.1 shows 1 to 6 of 40 prompts differ even for B1).
6. Greedy decoding only; sampling-based verification is not implemented.
7. Single seed (42) for training; no variance estimate.
8. No confidence intervals for the 24 Nov results yet.
9. The split-overlap test gate is not committed.

---

## Chapter 7. Conclusion and future work

### 7.1 Conclusion · `PARTIAL`

Write only from §4.3 and §5.1 until more results land. Supported statements: on P2 validation, draft-side LoRA raised acceptance from 18.63% to 31.46% and the output adapter to 23.38%; the bridge-plus-cache module mainly reduced drafter calls rather than adding accepted tokens once the drafter was trained.

### 7.2 Future work · `READY`

Everything marked `BLANK` above, in the order of `training_notes/iteration.md` and `training_notes/training.md` §5: P1 and P4, test replay, adaptive `K`, live sweep, ablations, Wave 2 targets, stochastic verification, a wider learning-rate grid.

---

## Back matter

- **References:** see [Literature](#literature).
- **Appendix A. Reproducibility:** run table (timestamp, kernel, Git SHA, job file) for every run cited above; exact commands from `training_notes/iteration.md` and `training_notes/babysitter.md`. `READY`.
- **Appendix B. Full sweep and per-task tables:** from the two `summary.json` files. `READY`.
- **Appendix C. Environment and pinned revisions:** `asset_plan.toml`, `summary.json` → `gpu_config.packages`, dataset revisions under `provenance.dataset_revisions`. `READY`.
- **Appendix D. Prompt templates and split slices:** `training_notes/training.md` §3. `READY`.

## Figures and tables plan

| Item | Chapter | Source | Status |
| --- | --- | --- | --- |
| Fig. system overview (drafter, bridge, verification) | 3 | draw from `first_principle_notes/vocab_bridge/pipeline.md` | to draw |
| Fig. one speculative round with rollback | 2 or 3 | `notes/psuedo_code.md` | to draw |
| Fig. oracle record-then-replay pipeline | 3 | `notes/oracle-replay.md`, `training_notes/training_phases.md` | to draw |
| Fig. training loss per configuration, P2 | 4 | heartbeat log in `results/submission-20261124T052050221490Z.json` → `run_log`, as plotted by `training/red_flag_figure.py` | data ready |
| Fig. top-1 agreement per checkpoint, P2 | 4 | `summary.json` → `trials` | data ready |
| Fig. acceptance per task, B1 vs T1 vs T2 | 4 | `summary.json` → `tiers.*.per_task` | data ready |
| Fig. tokens per target call and cache share, four arms | 5 | combo `summary.json` → `arms` | data ready |
| Table of pairs, datasets, splits, hyperparameters | 3 | sources in §3.3.1 to §3.3.4 | data ready |
| Tables in Chapters 4 and 5 | 4, 5 | as listed | data ready |
| Algorithm: one SpecAdapt round | 3 | `notes/psuedo_code.md` | ready to typeset |

Figures must be regenerated from the result files by a script kept under `thesis/figures/`; do not screenshot the red-flag PNG.

## Literature

The repo records one reference. Everything else must be found, read and added by the author before it is cited.

| Key | Work | Status |
| --- | --- | --- |
| OmniDraft | Ramakrishnan et al., "OmniDraft: A Cross-vocabulary, Online Adaptive Drafter", arXiv:2507.02659 | named in `notes/idea.md`; bibliographic details and content not verified here |
| Models | `JackFram/llama-68m`, `JackFram/llama-160m`, `lmsys/vicuna-7b-v1.5`, `Qwen/Qwen2.5-7B-Instruct` | cite their papers or model cards; revisions in `asset_plan.toml` |
| Datasets | GSM8K, HumanEval, MBPP, Alpaca, XSum | cite their papers; revisions in `summary.json` → `provenance.dataset_revisions` |
| To find | original speculative decoding papers, LoRA, knowledge distillation, prior cross-vocabulary and adaptive-`K` work | `[TO ADD]` |

## Run register

| Run | What | Git | Result |
| --- | --- | --- | --- |
| `20261123T141520817523Z` (Modal) | Stage 1 live smoke, P1, P2, P4 | `8c3dea7` | `results/modal/20261123T141520817523Z/` |
| `20261123T154925641802Z` (Modal) | Test and training oracles, Qwen and Vicuna | `640c25f` | `results/modal/20261123T154925641802Z/oracle-dump/` |
| `20261124T014249246739Z` | Validation replay, 40 prompts per pair | see file | `results/submission-20261124T014249246739Z.json` |
| `20261124T050647634040Z` | P2 smoke (64 fit, 40 val) | `7004468` clean | `results/runs/20261124T050647634040Z/` |
| `20261124T052050221490Z` | P2 full: B1, T2, T1 | `f5f1767` (notes-only dirty) | `results/runs/20261124T052050221490Z/` |
| `20261124T063355273658Z` | Kaggle CPU suite, 87 tests | `1812446` | `results/submission-20261124T063355273658Z.json` |
| `20261124T063633692969Z` | P2 M1 × T2, four arms, 80 prompts | `1812446` clean | `results/runs/20261124T063633692969Z/` |
| `20261013T070257913336Z` | October P2 M1 demo, 12 prompts | see archive | `results/pair2_comparison.json` |
