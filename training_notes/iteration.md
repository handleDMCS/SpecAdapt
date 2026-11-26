# Plan: adaptive K on P2, then the full method on P1 and P4

2026-11-26. Kaggle T4 only, $0 Modal. Data: the oracle dump at `results/modal/20261123T154925641802Z/oracle-dump/` (1,440 fit, 320 val per target). No new recording is needed. Pairs with a recorded target: **P2** (`llama_68m` → `qwen_7b`), **P1** (`llama_68m` → `vicuna_7b`), **P4** (`llama_160m` → `qwen_7b`).

**Cutoff: 2026-11-27 17:00 local time.** A run that is not collected by then waits for the next plan.

**Two directions, in this order:**

1. Train the adaptive-`K` policy on P2, where the LoRA drafter already exists. This comes first.
2. Carry the full method (M1 + LoRA + adaptive `K`) to P1 and P4. Their LoRA training needs no new code and no agent, so it is fired once the P2 `K` job is running.

**Where P2 stands** (validation, fixed `K = 3`; details in [Done](#done-2026-11-24))

| Arm | Prompts | Acceptance | Modeled cost / token |
| --- | --- | --- | --- |
| B1 | 320 | 18.63% | |
| T1 (rank 16, lr `1e-2`, 2 epochs) | 320 | 23.38% | |
| T2 (rank 16, lr `1e-3`, 4 epochs) | 320 | 31.46% | |
| B1 | 80 | 18.73% | 0.785 |
| B1 + M1 | 80 | 20.28% | 0.737 |
| T2 | 80 | 30.98% | 0.650 |
| T2 + M1 | 80 | 30.24% | 0.631 |

| # | What | Model | Status / needs | Agent session | Job wall time | Yields | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | Adaptive `K` on P2 | SOL | Start now | 60–90 min | about 90 min | P2 table of seven arms, fitted policy, cache statistics | 1.5 |
| 6 | P1 and P4 LoRA training | you | After the P2 `K` job is fired | none | P4 60–90 min, then P1 about 30 min | P1 and P4 tables (B1, T2, T1), weights | 2 |
| 7 | Full method on P1 and P4 | OPUS | After 5 passes and 6 is collected | 20–30 min | P1 about 90 min, P4 about 2.5 h, side by side | P1 and P4 tables of seven arms | 4 |
| 8 | Cache rule on P2 (optional) | OPUS | After 5, only if its cache table says so | 20 min | 15 min | Minimum match length and cache proposal length | 0.25 |

**Time to results, counted from the start of block 5**

| Milestone | Elapsed |
| --- | --- |
| SOL session for block 5 started | 0:00 |
| P2 `K` job fired, then P4 training fired beside it | 1:30 |
| P2 `K` result; P4 collected, P1 training fired | 3:00 |
| P1 collected | 3:40 |
| P1 and P4 `K` jobs fired | 4:15 |
| P1 `K` result | 5:45 |
| P4 `K` result | 6:45 |
| Same, if one run fails and is fixed in one session | 8:15 |

These are estimates. Measured on P2: the full training run took 30 minutes, and fixed `K = 3` replay runs at 1.0–1.2 s per prompt. P4 times are scaled guesses for the 160M drafter.

## Decisions

Settled; workers do not reopen them.

- **The drafter is frozen.** The `K` policy is a separate small module fitted on the LoRA drafter's confidence. No joint training.
- **Labels come from the drafter alone** (mode `baseline`, no M1), at fixed `K = 8`, so the policy can learn to draft longer as well as to stop early. M1 is switched on afterwards, around the fitted policy.
- **The policy is fitted on `fit/base` labels and its threshold is tuned on validation.** The drafter trained on `fit/base`, so it looks better there; the job reports the gap.
- **Features:** `entropy`, `top1_prob`, `top1_top2_margin`. `cache_hit` from [training.md §6](training.md#6-training-and-selection-kaggle-t4) is left out: it is constant in drafter-only labels.
- **Selection metric:** `modeled_cost_per_token = (rounds + c * draft_steps) / generated_tokens`, never acceptance rate (acceptance rewards always stopping after one token). `c` is the stage 1 draft-to-target step ratio: P1 `0.0575`, P2 `0.0623`, P4 `0.2315`. It is built from counts, so it is not a timing.
- **Refit per pair.** A policy or threshold from one pair is never reused on another.
- **Scope:** T2 (LoRA) only. T1 gets no `K` policy in this plan.

## GPU config

Fixed by the repo:

| Item | Value | Source |
| --- | --- | --- |
| Accelerator | Kaggle `NvidiaTeslaT4`, 16 GB per card. Use device 0 only. | `kernel-metadata.json`, `device_map={"": 0}` in the replay code |
| Image | Pinned by digest (`docker_image` in `kernel-metadata.json`, `docker_image_pinning_type = original`) | `kernel-metadata.json` |
| Packages | `transformers 4.57.1`, `accelerate 1.11.0`, `sentencepiece 0.2.1`, installed by `demo.benchmark.setup()`. Pin anything new the same way. | `src/demo/benchmark.py` |
| Replay | Drafter in FP16, `attn_implementation="sdpa"`, `batch = 1`, greedy, `max_new_tokens = 256`. No target model on Kaggle; the oracle verifies. | `training/kaggle/m1_combo.py` |
| Internet | On (model snapshots come from Hugging Face at the revisions in `asset_plan.toml`) | `kernel-metadata.json` |

Defaults for this plan. They are starting points; the worker may change them and must say so in the Log.

| Item | Value | Why |
| --- | --- | --- |
| Training | Unchanged from `training/jobs/p1.toml` and `p4.toml` | Same recipe and grid as P2. |
| Policy fit | On CPU inside the job, from the logged labels. No GPU training. | Three features; it fits in seconds. |
| Policy features | Computed from the drafter's logits cast to FP32 | FP16 softmax tails distort entropy. |
| `k_max` | 8 | Fixed `K = 3` is near the best fixed setting on P2, so the gain must come from varying `K`. |
| Seed | 42 | Matches `splits.toml`. |
| `max_minutes` | P1 120, P4 240, `p2-k` 180, `p1-k` 180, `p4-k` 300 | About twice the estimate. Kaggle ends a session at 12 hours regardless. |
| Concurrency | One job per kernel, two GPU sessions at a time. The P2 `K` job always gets a slot first. | If Kaggle refuses a second GPU session, run them one after the other. |
| Kernels | Training: `p1`, `p4`. Policy jobs: `p2-k`, `p1-k`, `p4-k`. | Policy jobs attach a training kernel's output as weights, so a collected training kernel is never rerun. |

Every job writes its actual config to `summary.json`: GPU name, torch and package versions, precision, seed, Git SHA.

## Rules for every worker

- Read "Decisions", "GPU config", this section and your own block. Nothing else in `training_notes/` unless the block names it.
- **Hands off the write-up.** Do not read, edit, move, stage or commit anything under `thesis/`, `.claude/` or `notes/`. No figures, report tables or write-up text; iterations are for code and experiments only. Stage your own files by path (`git add <paths>`), never `git add -A` or `git add .`. The untracked `thesis/` makes run cards say "dirty tree"; that is expected, so leave it.
- Before any GPU submission the CPU suite passes: `uv run --locked python -m demo.kaggle_run --metadata build/kernel-tests-metadata.json --cpu --no-cache tests/run_tests.py`
- **Fire and forget.** Submit GPU jobs with `--detach`, print the timestamp, kernel URL, expected minutes and the collect command, then end the session. Never wait on, poll or tail a GPU job.
- A job polices itself: a preflight of at most 5 minutes that asserts the cheap things and aborts on failure, a heartbeat at least every 60 s, abort on NaN or on passing `max_minutes`, results written as they are produced, and a final `summary.json` with `verdict` and the numbers the block names.
- Launch pad API (see `training/kaggle/launchpad_probe.py`): submit with `--detach --kernel <name> --job training/jobs/<job>.toml`; attach another kernel's output with `--kernel-source <owner>/specadapt-<name>`. In the job, `job_config()`, `heartbeat(phase, step, total, **metrics)` and `abort(reason)` come from `demo.kaggle_runtime`. Weights and tables go under the relative `artifacts/` folder, and `summary.json` goes in the working directory. Collect with `--resume <ts>`; watch with `uv run python -m demo.kaggle_watch <ts>`.
- Validation only. Nothing reads the test oracle.
- Replay stays lossless: every arm must reproduce the recorded token sequence, whatever the policy does.
- Never read `results/submission-*.json` (it embeds whole logs). Read the collected `results/runs/<ts>/summary.json`.
- Guardrails in [training.md §8](training.md#8-guardrails) apply. If something cannot be done as written, stop and say so.
- Finish with a local commit `iter <#>: <summary>` (no push) and at most 10 lines under `## Log` in [progress.md](progress.md): status, numbers, what is fired, what is open.

---

## 5 · Adaptive K on P2 · SOL

Fixed `K = 3` is close to the best fixed setting on P2, so the policy only pays if it can tell easy positions from hard ones. The job therefore also reports the ceiling: what a policy that stopped exactly at the first rejection would reach.

```text
Iteration 5. Read training_notes/iteration.md "Decisions", "GPU config", "Rules for every worker" and
block 5, then src/specadapt/decoding.py, src/specadapt/bridge.py and training/kaggle/m1_combo.py.
Reuse m1_combo's loading, weight check, replay loop and summary code by importing or factoring it,
not by copying it. Keep the script free of anything P2-specific: pair, cost ratio, weight hash and
kernel source come from the job TOML, because iteration 7 runs it on P1 and P4 unchanged.

1. src/specadapt/adaptive_k.py
   - EntropyKPolicy(threshold, k_max): keep drafting while the entropy of the token just drafted is
     at most the threshold.
   - LogisticKPolicy(threshold, k_max, weights): features entropy, top1_prob, top1_top2_margin of the
     drafter's next-token distribution (logits cast to FP32). Keep drafting while P(accept) is at
     least the threshold. fit() is L2 logistic regression on standardized features, on CPU, saved
     and loaded as JSON. No new dependency unless it is already on the Kaggle image.
   - Both draft at least one token and at most k_max = 8. The token that triggers the stop is still
     proposed: its draft step is already paid.
2. generate(..., policy=None, cache_k=None, trace=False). With the defaults, behaviour and every
   existing test are unchanged.
   - policy: decides after each drafted token whether to draft another.
   - cache_k: how many tokens a continuation-cache hit proposes (default k).
   - trace: return one record per round: source (draft or cache), cache match length, target tokens
     proposed and accepted, and per drafted token its features and label.
   - Label, defined on text so it also holds across tokenizers: drafted token i is accepted when the
     decoded text of drafted tokens 1..i is no longer than the decoded text of the target tokens
     accepted in that round. Record tokens up to and including the first rejected one.
   - The bridge's `[:k]` cut must never shorten a k_max-token draft proposal. Fix that in the policy
     and trace paths only.
3. training/kaggle/k_policy.py + training/jobs/p2-k.toml (c = 0.0623, t2_sha256
   fe8bf05e3b77ff9460e9ff61f73bef4d28c1992cf0d9a2d773510d360506bcfb, weights from
   --kernel-source dungg1810/specadapt-p2). Stages, results written as each one ends:
   a. Gates on all 320 validation prompts, fixed K = 3, mode baseline: B1 must give exactly
      18,316 accepted of 98,317 proposed, and T2 must be within 0.1 points of 31.46%. Abort otherwise.
   b. Labels: T2, mode baseline, fixed K = 8, trace on, over the first 180 fit prompts of each task
      (720) and over all 320 validation prompts. Fit the logistic policy on the fit labels. Report
      AUC and the label acceptance rate on fit and on validation, overall and per task.
   c. Ceiling, computed from the validation K = 8 trace without another replay: the modeled cost if
      drafting had stopped right after the first rejected token of every round.
   d. Threshold sweeps on the 80-prompt validation subset (m1_combo.select_subset, 20 per task), five
      thresholds each, taken at the 10th, 30th, 50th, 70th and 90th percentile of that policy's score
      on the fit labels: entropy on B1 in mode m1, entropy on T2, logistic on T2. Pick each by
      modeled cost.
   e. Seven arms on all 320 validation prompts:
      B1 fixed K=3 | T0 = B1 + M1 + entropy | T2 fixed K=3 | T2 fixed K=3 + M1 | T2 + entropy |
      T2 + logistic | T2 + logistic + M1 (cache_k = 8).
   One `phase=result arm=<name>` heartbeat line per arm. summary.json: per arm and per task
   {acceptance, tokens_per_round, draft_steps_per_token, mean_k, cache_round_share,
   modeled_cost_per_token, cost relative to B1 fixed K=3}; the fitted policy with its AUCs and the
   swept thresholds with their costs; the ceiling; and for each M1 arm the cache rounds by match
   length {proposed, accepted}. artifacts/: the policy JSON, the labels (compact, so a refit needs no
   GPU) and the per-prompt counts of each arm.
4. src/demo/kaggle_run.py: add "thesis" and "notes" to EXCLUDED_DIRS so a submission bundle never
   carries them, with a test.
5. CPU tests on the tiny models already in the suite: policy=None is unchanged; each policy varies K
   and stops as specified; the label rule on a same-tokenizer and a cross-tokenizer case; replay stays
   lossless under both policies; k_policy.py end to end.
6. CPU suite, commit, then submit detached: kernel `p2-k`, the training-inputs dataset attached,
   --expected-minutes 90. Print the run card and the watch and collect commands, and the same launch
   command with <pair> placeholders for iteration 7. End the session.
```

**Next** (you; bring `results/runs/<ts>/summary.json` to Fable)

| P2 result | Do |
| --- | --- |
| `verdict = pass`, and `T2 + logistic + M1` costs at least 2% less than `T2 fixed K=3 + M1` | The recipe works. Go to 7. |
| Pass, gain under 2%, ceiling also within 5% of fixed `K = 3` | Adaptive `K` has little room on P2. It is a result. Still go to 7: P4's drafter costs four times more per step, so stopping early matters more there. |
| Pass, gain under 2%, but the ceiling is far better | The features do not separate. Prompt 5.c, OPUS. |
| Logistic no better than entropy | Note it. 7 runs both anyway. |
| A gate aborts, or the run fails | The failure prompt at the end of this note, OPUS. |

```text
Iteration 5.c. The P2 policy gained little although the ceiling is far away:
results/runs/<timestamp>/ (summary and the labels artifact only). No GPU. From the saved labels:
1. AUC per task and per position in the round. Is the policy failing everywhere or on one task?
2. Refit with added features that are free at inference: position in the round, the previous drafted
   token's top1_prob, the running product of top1_prob in this round. Report the AUC of each.
3. Calibration on validation of the policy fitted on fit/base. If it is badly off, refit by 2-fold
   cross-fitting by prompt on validation and compare.
Report which change lifts validation AUC the most and the modeled cost it would give at its best
threshold, estimated from the trace. Do not submit a job; that decision goes to Fable.
```

## 6 · P1 and P4 LoRA training · you

No agent. Same trainer and grid as P2, T2 first. Fire these only after the P2 `K` job from block 5 is submitted, so adaptive `K` is never waiting on a GPU slot. P4 goes first because it is the slow one and runs beside the `K` job; P1 goes when either of the two finishes.

```powershell
uv run --locked python -m demo.kaggle_run training/kaggle/train_pair.py --detach --kernel p4 --job training/jobs/p4.toml --dataset dungg1810/specadapt-train-inputs --expected-minutes 90
uv run --locked python -m demo.kaggle_run training/kaggle/train_pair.py --detach --kernel p1 --job training/jobs/p1.toml --dataset dungg1810/specadapt-train-inputs --expected-minutes 40
```

Watch with `uv run python -m demo.kaggle_watch <ts>`. Collect each with `uv run python -m demo.kaggle_run --resume <ts>`. The numbers are in `results/runs/<ts>/summary.json` and the weights in `artifacts/`.

**Next**

| What you see | Do |
| --- | --- |
| `verdict = pass` and `t2_gain_pp ≥ 2` | That pair is ready for 7. |
| `t2_gain_pp < 2`, or T2 below B1 | It is still a result. Prompt 6.b, OPUS, before spending more GPU on that pair. |
| The run fails | The failure prompt at the end of this note, OPUS. |

```text
Iteration 6.b. Training barely moved acceptance on <pair>: results/runs/<timestamp>/ (summary only).
P2 gained 12.8 points with the same code, so look for what differs on this pair. Check in this
order, each with a CPU test or a 15-minute smoke-sized job, not a full run:
1. The trained weights are the ones being replayed (merge applied, right checkpoint loaded).
2. Training text matches inference: generate() feeds the drafter draft_prompt(prompt) + decoded
   target prefix, re-encoded every round. Compare with what the trainer tokenized. On P1 also check
   the Vicuna v1.1 prompt and same_tokenizer=True.
3. Label mask and EOS handling.
4. Teacher-forced top-1 agreement rose but replay acceptance did not: then it is the proposal path.
   Measure acceptance when the proposal is the teacher text itself re-tokenized by the drafter.
5. Learning rate too low or too high: look at the loss values in the heartbeat lines.
Report the cause and the fix, resubmit detached once in a fresh kernel (`<pair>-r2`). If none of the
five explains it, stop and write up what you ruled out; that goes to Fable.
```

## 7 · Full method on P1 and P4 · OPUS

Needs: block 5 passed, and the P1 and P4 training runs collected with `t2_gain_pp ≥ 2`. A pair that is not ready is left out; the other still runs.

```text
Iteration 7. Read training_notes/iteration.md "Decisions", "GPU config", "Rules for every worker" and
block 7, then the iteration 5 entry in the Log of training_notes/progress.md and
training/kaggle/k_policy.py. The collected training runs are results/runs/<p1 timestamp>/ and
results/runs/<p4 timestamp>/ (summary only).

1. training/jobs/p1-k.toml and p4-k.toml, copied from p2-k.toml: pair, cost ratio (P1 0.0575,
   P4 0.2315), the T2 weight hash from each run's summary, max_minutes (P1 180, P4 300). Replace the
   two P2 gate counts with each pair's own B1 and T2 numbers from its training summary.
2. Move anything pair-specific that iteration 5 left in code into the TOML. Do not change the policy
   logic, the features, the threshold grid or the arm list. The policy is refitted per pair by the job.
3. P1 shares one tokenizer between drafter and target: drafter-only arms use same_tokenizer=True as
   the trainer does. Its M1 arms go through the vocabulary bridge, which P1 has never run. Add a CPU
   test that replays two recorded P1 validation rows in mode m1 with the pinned tokenizers and
   reproduces the recorded sequence. If M1 lowers P1 acceptance on GPU, that is a finding: report
   it, do not tune the bridge.
4. CPU suite, commit, then submit both detached, P4 first: kernels `p4-k` and `p1-k`, with
   --kernel-source dungg1810/specadapt-p4 and dungg1810/specadapt-p1 and the training-inputs dataset.
   Print both run cards and the watch and collect commands. End the session.
```

**Next:** collect both and bring the three summaries (P2, P1, P4) to Fable. The numbers go into the Log.

## 8 · Cache rule on P2 (optional) · OPUS

Only when the cache table from block 5 shows that short matches are accepted clearly less often than the T2 drafter's proposals, and time remains before the cutoff.

```text
Iteration 8. Read training_notes/iteration.md "Decisions", "Rules for every worker" and block 8, then
training/kaggle/k_policy.py and the cache-rounds-by-match-length table in results/runs/<p2-k
timestamp>/summary.json.
Add `cache_min_match` to generate() (default: today's behaviour). One job, kernel `p2-cache`, on the
80-prompt validation subset with T2 + the fitted logistic policy + M1: cache_min_match in
{2, 3, 4, 5} times cache_k in {3, 8}. Report modeled cost and acceptance per setting and name the
best. CPU test, CPU suite, commit, submit detached (--expected-minutes 15), end the session.
```

---

## Failure prompt (any block) · OPUS

```text
Iteration <#>.b. A run failed: results/runs/<timestamp>/ after collecting it (summary and the printed
log tail only). Read training_notes/iteration.md "Rules for every worker" and block <#>. Find the
cause, reproduce it in a CPU test, fix, rerun the CPU suite, commit. Resubmit detached in a fresh
kernel (`<name>-r2`) so a still-running version is not disturbed. Print the run card and commands
again. If this is a second failure for the same reason, stop and write what you know.
```

---

## Done (2026-11-24)

Finished steps, kept for reference. Details are in the Log in [progress.md](progress.md).

| # | What | Model | Commit | Result |
| --- | --- | --- | --- | --- |
| 1 | Launch pad | OPUS | `1f42541` | `--detach`, `--kernel`, `--job`, `heartbeat()`, `artifacts/`, `kaggle_watch`. |
| 2 | Data and trainer | SOL | `d42d173` | Private dataset `dungg1810/specadapt-train-inputs`, `src/specadapt/alignment.py`, `training/kaggle/train_pair.py`, job TOMLs `p2-smoke`, `p2`, `p1`, `p4`. |
| 3 | LoRA first, smoke and P2 fired | OPUS | `7004468` | `tier_order` in the job TOML, one `phase=result` line per tier, short path `p2-lora`. Smoke `20261124T050647634040Z` passed its B1 gate. |
| 4 | P2 collected | you | | Run `20261124T052050221490Z`, 30 minutes. B1 18.63%, T2 31.46% (+12.83 points), T1 23.38% (+4.75 points) on 320 validation prompts. Weights in `results/runs/20261124T052050221490Z/artifacts/`. |
| | M1 × T2 combination | OPUS | `1812446` | Run `20261124T063633692969Z`, `training/kaggle/m1_combo.py`, 80 validation prompts, four arms (table at the top). With LoRA, M1 no longer raises acceptance (−0.74 points) but cuts draft steps by 20% and modeled cost by 3%. |

## Not in this plan

- Checkpoint freeze and the 400-prompt test replay.
- A `K` policy for T1, a hidden-state stopping head, and joint training of drafter and policy.
- The M1 seed cache (built from GSM8K train answers).
- Learning curve, leave-one-task-out, cache-size ablations.
- Anything on Modal. One warning for when stage 5 comes up: in the stage 1 smoke, live B1 matched B0 on only 34/40 (P2), 36/40 (P4), 39/40 (P1) prompts, and the stage 5 gate asks for 99%. Probe that before paying for the sweep.
- P3 and P5: their target is not recorded.
- Reports, figures and the write-up. Writing is its own track; iterations here are for code and experiments only.
