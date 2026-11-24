# Today's plan: train on the recorded oracles

2026-11-24. Kaggle T4 only, $0 Modal. Data: the oracle dump at `results/modal/20261123T154925641802Z/oracle-dump/` (1,440 fit, 320 val per target). Pairs with a recorded target: **P2** (`llama_68m` → `qwen_7b`), **P1** (`llama_68m` → `vicuna_7b`), **P4** (`llama_160m` → `qwen_7b`).

**Priority: one complete P2 result as early as possible, LoRA (T2) before T1.** P1, P4 and the `K` policies follow in a later session and reuse everything unchanged.

**First milestone:** a P2 validation table with B1 and T2 acceptance at fixed `K = 3`, overall and per task, and the LoRA weights saved. T1 lands in the same run, after T2.

| # | What | Model | Status / needs | Agent session | Job wall time | Yields | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Launch pad | OPUS | Done, `1f42541` | | | Detached submit, live log, weights out | 0 |
| 2 | Data + trainer | SOL | Done, `d42d173` | | | Training-inputs dataset, `train_pair.py`, job TOMLs | 0 |
| 3 | LoRA first, fire smoke and P2 together | OPUS | Next | 25–30 min | smoke 15 min; P2 about 70 min, T2 numbers at about 45 min | P2 table (B1, T2, then T1), weights | 1.5 |
| 4 | Collect P2 and branch | you | After 3 | none | 2 min | P2 `summary.json` and weights on disk | 0 |
| 6 | P1 and P4 | you | Later session | none | P1 55 min, P4 2–2.5 h, side by side | P1 and P4 tables | 3.25 |
| 5 | `K` policies on P2 | SOL | Later session | 60–90 min | 2 h | P2 T0 and logistic-policy numbers | 2 |

**Time to the first P2 result, counted from the start of iteration 3**

| Milestone | Elapsed |
| --- | --- |
| Smoke and P2 fired | 0:30 |
| Smoke verdict | 0:45 |
| T2 numbers on the live log (`phase=result tier=T2`) | 1:15 |
| P2 run complete, summary and weights on disk | 1:45 |
| Same, if the smoke fails once and is fixed in one session | 2:45 |
| Short path `p2-lora` (B1 and T2 only, one learning rate), from its own firing | 0:40 |

These are estimates. Replay is measured at about 2.2 s per prompt (`K = 3`, 68M drafter). Training time is a guess until the smoke reports minutes per phase.

## GPU config

Fixed by the repo:

| Item | Value | Source |
| --- | --- | --- |
| Accelerator | Kaggle `NvidiaTeslaT4`, 16 GB per card. Use device 0 only. | `kernel-metadata.json`, `device_map={"": 0}` in the replay code |
| Image | Pinned by digest (`docker_image` in `kernel-metadata.json`, `docker_image_pinning_type = original`) | `kernel-metadata.json` |
| Packages | `transformers 4.57.1`, `accelerate 1.11.0`, `sentencepiece 0.2.1`, installed by `demo.benchmark.setup()`. Pin `peft` the same way if it is used. | `src/demo/benchmark.py` |
| Replay | Drafter in FP16, `attn_implementation="sdpa"`, `batch = 1`, greedy, `max_new_tokens = 256`. No target model on Kaggle; the oracle verifies. | `modal/stage1_smoke.py` `kaggle_replay` |
| Internet | On (model snapshots come from Hugging Face at the revisions in `asset_plan.toml`) | `kernel-metadata.json` |

Defaults I am choosing for training. They are starting points; the worker may change them and must say so in the Log.

| Item | Value | Why |
| --- | --- | --- |
| Training precision | FP32 weights, no autocast | The drafters are 68M and 160M parameters, so FP32 fits easily and avoids FP16 loss-scaling problems. |
| After training | Merge in FP32, run the `1e-3` merge check in FP32, then cast to FP16 for replay | Replay must use the same precision the live sweep will use. |
| Batching | Length-sorted, at most 16 sequences and 16,000 tokens per batch for `llama_68m`; 8 and 8,000 for `llama_160m` | XSum prompts run long (the drafter limit is context minus 512). On out-of-memory, halve and retry once, then abort. |
| Optimizer | AdamW, no weight decay on the adapters, linear warmup over 5% of steps then cosine | Standard for small LoRA fits. |
| Seed | 42 for subset choice, shuffling and init | Matches `splits.toml`. |
| `max_minutes` | Smoke 20, P2 120, `p2-lora` 60, P1 120, P4 240, `K` policies 180 | About twice the estimate. Kaggle ends a session at 12 hours regardless. |
| Tier order | T2 (LoRA) before T1 in every job | The LoRA result is needed first; T1 follows in the same run. |
| Concurrency | One job per kernel. Smoke and P2 side by side, later P1 and P4. | If Kaggle refuses a second GPU session, run them one after the other. |

Every job writes its actual config to `summary.json`: GPU name, torch and package versions, precision, batch limits, seed, Git SHA.

## Rules for every worker

- Read this section, "GPU config" above, and your own block. Nothing else in `training_notes/` unless the block names it.
- Before any GPU submission the CPU suite passes: `uv run --locked python -m demo.kaggle_run --metadata build/kernel-tests-metadata.json --cpu --no-cache tests/run_tests.py`
- **Fire and forget.** Submit GPU jobs with `--detach`, print the timestamp, kernel URL, expected minutes and the collect command, then end the session. Never wait on, poll or tail a GPU job.
- A job polices itself: a preflight of at most 5 minutes that asserts the cheap things and aborts on failure, a heartbeat at least every 60 s, abort on NaN loss or on passing `max_minutes`, results written as they are produced, and a final `summary.json` with `verdict` and the numbers the block names.
- Launch pad API (from iteration 1, see `training/kaggle/launchpad_probe.py`): submit with `--detach --kernel <name> --job training/jobs/<job>.toml`. In the job, `job_config()`, `heartbeat(phase, step, total, **metrics)` and `abort(reason)` come from `demo.kaggle_runtime`. `heartbeat` already aborts on NaN or infinite float metrics and after `max_minutes`. Weights and tables go under the relative `artifacts/` folder, and `summary.json` goes in the working directory. Collect with `--resume <ts>`; watch with `uv run python -m demo.kaggle_watch <ts>`.
- Validation only. Nothing reads the test oracle today.
- Never read `results/submission-*.json` (it embeds whole logs). Read the collected summary.
- Guardrails in [training.md §8](training.md#8-guardrails) apply. If something cannot be done as written, stop and say so.
- Finish with a local commit `iter <#>: <summary>` (no push) and at most 10 lines under `## Log` in [progress.md](progress.md): status, numbers, what is fired, what is open.

---

## Done

- **1 · Launch pad · OPUS.** `--detach`, `--kernel`, `--job`, `heartbeat()`, `artifacts/`, `kaggle_watch`. See the Log in [progress.md](progress.md).
- **2 · Data and trainer · SOL.** Private dataset `dungg1810/specadapt-train-inputs`, `src/specadapt/alignment.py`, `training/kaggle/train_pair.py` (already heartbeats, writes `summary.json` and `artifacts/`), job TOMLs `p2-smoke`, `p2`, `p1`, `p4`. CPU suite 69/69.

## 3 · LoRA first, fire smoke and P2 together · OPUS

The trainer currently runs B1, then all of T1, then T2. T2 is the result needed first, so it moves to the front. The smoke and the full run start together: GPU hours are plentiful (about 30 unused this week) and wall time is the scarce resource, so the smoke acts as an early warning instead of a gate.

```text
Iteration 3. Read training_notes/iteration.md "GPU config", "Rules for every worker" and block 3,
then training/kaggle/train_pair.py and training/jobs/p2.toml. Iterations 1 and 2 are committed.
Keep the change set small: every edit here sits in front of a time-critical run.

1. Tier order from the job TOML: `tier_order = ["T2", "T1"]`. Default when the key is absent: the
   current order. A one-element list runs that tier only. Set T2 first in p2-smoke, p2, p1 and p4.
2. After each tier's validation replay, one heartbeat line that carries the whole result:
   phase=result tier=<T> acceptance=<..> b1=<..> gain_pp=<..> plus acceptance per task. It must be
   readable on the live log before the job ends.
3. A short-path job TOML training/jobs/p2-lora.toml: tier_order = ["T2"], one learning rate (3e-4),
   checkpoints [2, 4], max_minutes = 60. Everything else as p2.toml.
4. CPU tests for 1 to 3 on the tiny models already used in the suite. Run the CPU suite. Commit
   everything in the tree, training_notes/ included, so both runs cite a clean Git SHA.
5. Submit both detached, each in its own kernel with the training-inputs dataset attached:
   p2-smoke (kernel `p2-smoke`) and p2 (kernel `p2`). If Kaggle refuses the second GPU session,
   submit only the smoke and print the p2 command for me.
6. Print, ready to copy: both run cards, the watch and collect commands for each, and the launch
   commands for p2-lora, p1 and p4 with expected minutes.
End the session. Do not wait for either job.
```

**Next** (you; watch both with `uv run python -m demo.kaggle_watch <ts>`)

| What you see | Do |
| --- | --- |
| Smoke `verdict = pass` | Nothing. P2 is already running. Go to 4 when it ends. |
| Smoke fails, or P2 aborts in preflight | Prompt 3.b, OPUS. |
| P2 fails after the smoke passed | Prompt 3.b with the P2 timestamp. |

```text
Iteration 3.b. A run failed: results/runs/<timestamp>/ after collecting it (summary and the printed
log tail only). Find the cause, reproduce it in a CPU test, fix, rerun the CPU suite, commit.
Resubmit detached in fresh kernels (`p2-smoke-r2`, `p2-r2`) so a still-running version is not
disturbed: the smoke again, and with it either the full p2 or, when I say the short path is needed,
p2-lora. Print the run cards and commands again. If this is a second failure for the same reason,
stop and write what you know.
```

`p2-lora` is the short path: about 35–40 minutes from firing to a B1-versus-T2 table and saved LoRA weights. Use it when the full run cannot finish in the time available. The full `p2` run can follow afterwards to add the learning-rate sweep and T1.

## 4 · Collect P2 and branch · you

```powershell
uv run python -m demo.kaggle_run --resume <p2 timestamp>
```

The numbers are in `results/runs/<p2 timestamp>/summary.json` (acceptance per tier, overall and per task, gains over B1, the sweep) and the weights are in `artifacts/`. If the run is still training T1 when the T2 numbers are needed, they are on the live log in the `phase=result tier=T2` line.

| P2 result | Do |
| --- | --- |
| `t2_gain_pp ≥ 2` | The recipe works. Later session: 6, then 5. |
| `t2_gain_pp < 2`, or T2 below B1 | It is still a result. Then prompt 4.b, OPUS, before spending GPU on P1 and P4. |

```text
Iteration 4.b. P2 training barely moved acceptance: results/runs/<timestamp>/ (summary only). Check in
this order, each with a CPU test or a 15-minute smoke-sized job, not a full run:
1. The trained weights are the ones being replayed (merge applied, right checkpoint loaded).
2. Training text matches inference: generate() feeds the drafter draft_prompt(prompt) + decoded
   target prefix, re-encoded every round. Compare with what the trainer tokenized.
3. Label mask and EOS handling.
4. Teacher-forced top-1 agreement rose but replay acceptance did not: then it is the bridge. Measure
   acceptance when the proposal is the teacher text itself re-tokenized by the drafter.
5. Learning rate too low or too high: look at the loss values in the heartbeat lines.
Report the cause and the fix, resubmit p2 detached once. If none of the five explains it, stop and
write up what you ruled out; that goes to Fable.
```

---

## Later session

### 6 · P1 and P4 · you

Run the `p4` and `p1` launch commands printed in iteration 3, P4 first because it is the slow one. Each uses the same trainer and grid as P2, T2 first. Collect each exactly as in 4. If Kaggle refuses a second GPU session, run them one after the other.

### 5 · K policies on P2 · SOL

Fire it only after P2 has winners.

```text
Iteration 5. Read training_notes/iteration.md "Rules for every worker" and block 5, then
src/specadapt/decoding.py and bridge.py.

1. src/specadapt/adaptive_k.py: EntropyKPolicy(threshold, k_max) and LogisticKPolicy (L2, features
   entropy, top1_prob, top1_top2_margin, cache_hit), k_max = 8. cache_hit = 1 when the previous round
   was served by a continuation cache. `generate(..., policy=None)`: with None, behaviour and all
   existing tests are unchanged.
2. Selection metric. Do not select by acceptance rate: it rewards always stopping after one token.
   Use modeled_cost_per_token = (rounds + c * draft_steps) / generated_tokens with c = 0.0623 for P2
   (stage 1 measured draft-to-target step ratio; P1 0.0575, P4 0.2315). It is built from counts, so
   it is not a timing.
3. training/kaggle/k_policy.py + job TOML, for P2, for the base drafter and each trained winner
   loaded from the p2 run's artifacts:
   - T0: mode "m1" (bridge + continuation caches) with EntropyKPolicy. Sweep 5 thresholds on an
     80-prompt validation subset (20 per task), then run the best on all 320.
   - T1, T2: one fixed K=8 replay over validation logging the features and accepted/rejected per
     drafted token, up to and including the first rejection. Fit the logistic policy with 2-fold
     cross-fitting by prompt, sweep its threshold on the 80-prompt subset, run the best on all 320.
   summary.json: per tier {setting, acceptance, accepted tokens per round, modeled cost} next to B1.
CPU tests with stub models for both policies. Submit detached when the p2 artifacts exist; otherwise
commit and print the launch command for me.
```

**Next:** numbers go into the Log. Applying the policies to P1 and P4 follows.

---

## Not in this plan

- Checkpoint freeze and the 400-prompt test replay.
- Learning curve, leave-one-task-out, cache ablations.
- Anything on Modal. One warning for when stage 5 comes up: in the stage 1 smoke, live B1 matched B0 on only 34/40 (P2), 36/40 (P4), 39/40 (P1) prompts, and the stage 5 gate asks for 99%. Probe that before paying for the sweep.
- P3 and P5: their target is not recorded.
- Reports, figures and the write-up. Writing is its own track; iterations here are for code and experiments only.
