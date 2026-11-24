# Today's plan: train on the recorded oracles

2026-11-24. Kaggle T4 only, $0 Modal. Data: the oracle dump at `results/modal/20261123T154925641802Z/oracle-dump/` (1,440 fit, 320 val per target). Pairs with a recorded target: **P2** (`llama_68m` → `qwen_7b`, first), **P1** (`llama_68m` → `vicuna_7b`), **P4** (`llama_160m` → `qwen_7b`).

**Done today means:** one validation table per pair with B1, T1, T2 acceptance at fixed `K = 3`, and adapter weights saved. Stretch: T0 and the `K` policies on P2.

| # | What | Model | Needs | Agent session | Job wall time | Yields | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Launch pad: detach, heartbeat, weights out | OPUS | | 45–60 min | two CPU probes, 5 min each | Detached submit and a live log | 0 |
| 2 | Data + trainer | SOL | | 60–90 min | CPU suite, 5 min | Training-inputs dataset, job ready to fire | 0 |
| 3 | Smoke, then you launch P2 | OPUS | 1, 2 | 20 min | smoke 15 min, then P2 60 min | Pipeline proven; P2 table (B1, T1, T2) | 1.25 |
| 4 | Read P2, launch P1 and P4 | you | 3 | none | P1 55 min, P4 2–2.5 h, side by side | P1 and P4 tables | 3.25 |
| 5 | `K` policies (stretch) | SOL | 2; fire after P2 | 60–90 min | 2 h | P2 T0 and logistic-policy numbers | 2 |

1 and 2 run in parallel. 5 can be written while P2 trains. Total about 6.5 of 30 GPU-hours.

**Time to result, from a start now**

| Milestone | Elapsed |
| --- | --- |
| 1 and 2 done | 1.5 h |
| Smoke verdict | 2.25 h |
| P2 table, the first real answer | 3.25 h |
| P1 and P4 tables, "done today" | 5.5–6 h |
| Stretch: `K` policies on P2 | 5.5–6 h if Kaggle accepts a third concurrent GPU session, otherwise about 8 h |

All of these are estimates. The replay times come from one measurement: 340 s for 120 prompt replays on T4, about 2.2 s per prompt at `K = 3` with the 68M drafter. P4 is scaled from the stage 1 step ratio (the 160M drafter step is about 3.7 times slower). Training time is a guess until the smoke reports minutes per phase; the smoke's `summary.json` is where these numbers get corrected. Agent session times are guesses too.

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
| `max_minutes` | Smoke 20, P2 120, P1 120, P4 240, `K` policies 180 | About twice the estimate. Kaggle ends a session at 12 hours regardless. |
| Concurrency | One job per kernel. P1 and P4 side by side. | If Kaggle refuses a second GPU session, run P4 first. |

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

## 1 · Launch pad · OPUS

```text
Iteration 1. Read training_notes/iteration.md "Rules for every worker" and block 1.

Today kaggle_run blocks until the kernel ends, and kaggle_runtime redirects stdout into a file, so the
Kaggle log page is blank during a run. Fix that with the smallest change that works. CPU kernels only.

1. Commit the working tree as it is (modal/ scripts, training_notes/, the two deleted notes).
2. src/demo/kaggle_run.py: `--detach` (push, print timestamp + kernel URL + the --resume command,
   exit 0) and `--kernel <name>` (run in <owner>/specadapt-<name> so parallel jobs do not collide).
   Record the git SHA in the submission record.
3. A bundled `heartbeat(phase, step, total, **metrics)` that prints one line to the real kernel
   stdout, bypassing capture_output, so I can watch on the Kaggle page.
4. Files a script writes under artifacts/ stay as kernel output files (not embedded in the result
   JSON, not deleted). `--resume` downloads them to results/runs/<timestamp>/ and prints only the
   job's summary.json, plus the last 20 log lines if it failed.

Done when: one CPU kernel probe (2 minutes of heartbeats, a 1 MB artifact) shows heartbeats live on
the Kaggle page and the artifact comes back with a matching sha256. Tell me exactly where to look.
```

**Next:** works → 3. Heartbeats not visible live → 3 anyway, and you watch kernel status instead. Artifact does not come back → stay in 1; trained weights depend on it.

## 2 · Data and trainer · SOL

```text
Iteration 2. Read training_notes/iteration.md "Rules for every worker" and block 2, then
training.md section 6 (rows "Training text", T1, T2) and src/specadapt/decoding.py.
Do not edit src/demo/kaggle_run.py or kaggle_runtime.py; another session owns them.

Goal: one Kaggle job that takes a pair and returns "did training raise acceptance on validation".

1. training/build_training_inputs.py: for qwen_7b and vicuna_7b, join training/dataset/train/
   {fit/base,val}.jsonl to the oracle rows in results/modal/20261123T154925641802Z/oracle-dump/
   <target>/train/ by prompt_id. Check the prompt hash per row and exact coverage (1,440 and 320).
   Keep prompt_id, task, prompt, teacher text, target token_ids, stop_reason. Upload as one private
   Kaggle dataset `specadapt-train-inputs` (approved). No test rows.
2. src/specadapt/alignment.py:
   - T1 OutputLogitAdapter: logit bias + low-rank lm_head delta, rank 8 or 16.
   - T2 LoRAAdapter: rank 16 on q,k,v,o,gate,up,down, merge_and_unload().
   - Trainer: text = draft_prompt(prompt) + teacher text, + draft EOS only when stop_reason is "eos",
     tokenized with the draft tokenizer, loss on answer tokens only. Only the drafter trains.
   - Merged and unmerged logits differ by less than 1e-3.
3. training/kaggle/train_pair.py, configured by training/jobs/<job>.toml (the runner bundles only
   .py and .toml and passes no arguments). For one pair:
   a. preflight (inputs present, one training step per tier, merge check);
   b. B1: fixed K=3 replay of the base drafter on validation through generate(oracle=...), the way
      modal/stage1_smoke.py kaggle_replay does it (same_tokenizer for P1);
   c. train at most 6 configurations per tier. Suggested: T1 rank {8,16} x lr {1e-3,3e-3,1e-2};
      T2 lr {1e-4,3e-4,1e-3} checked at 2 and 4 epochs. Rank them by teacher-forced top-1 agreement
      on validation answer tokens, which takes seconds;
   d. merge the winner of each tier and run the same fixed K=3 replay on all 320 validation prompts;
   e. save the two winners' unmerged weights under artifacts/.
   summary.json: verdict, per tier {config, top-1 agreement, acceptance overall and per task},
   B1 acceptance, t1_gain_pp, t2_gain_pp (tier minus B1, percentage points), minutes per phase,
   seconds per replayed prompt, and the actual GPU config (see "GPU config" in iteration.md; its
   training defaults and max_minutes go in the job TOML).
4. A smoke mode in the TOML: 64 fit prompts, one configuration per tier, and the first 10 validation
   prompts per task in file order. On those 40 prompts P2's B1 must give exactly accepted=2610,
   proposed=13938 (the recorded run). Abort otherwise.
5. Job TOMLs for p2-smoke, p2, p1, p4, all with the same grid.

CPU tests on tiny random models: merge equivalence for both adapters, label masking, EOS for both
stop reasons, the join rejecting a wrong hash and a missing row. Do not submit any GPU job.
```

**Next:** 3.

## 3 · Smoke, then P2 · OPUS

```text
Iteration 3. Read training_notes/iteration.md "Rules for every worker" and block 3. Iterations 1 and
2 are committed; check the Log in progress.md.

1. Wire training/kaggle/train_pair.py to heartbeat() and artifacts/ from iteration 1: a heartbeat in
   the training loop and the replay loop with loss or running acceptance and an ETA.
2. Run the CPU suite. Submit p2-smoke detached in kernel `p2-smoke` with the training-inputs dataset
   attached. It should finish in under 15 minutes.
3. Print, ready to copy: the collect command for the smoke, and one launch command each for p2, p1
   and p4 (detached, own kernel, dataset attached) with expected minutes.
End the session.
```

**Next** (you, after collecting the smoke)

| Smoke | Do |
| --- | --- |
| `verdict = pass` | Run the printed `p2` command yourself. No agent needed. |
| Fail | Prompt 3.b below, OPUS. |

```text
Iteration 3.b. The p2-smoke run failed: results/runs/<timestamp>/. Find the cause, reproduce it in a
CPU test, fix, rerun the CPU suite, resubmit p2-smoke detached, print the commands again. If it is a
second failure for the same reason, stop and write what you know.
```

## 4 · Read P2, branch · you

Collect the `p2` run and read `summary.json`.

| P2 result | Do |
| --- | --- |
| `t2_gain_pp ≥ 2` | Run the printed `p1` and `p4` commands. P4 first, it is the slow one. Start 5 if there is time. |
| `t2_gain_pp < 2`, or T1 and T2 both below B1 | Prompt 4.b, OPUS. Do not spend GPU on P1 and P4 yet. |
| Job failed | Prompt 3.b with the `p2` timestamp. |

```text
Iteration 4.b. P2 training barely moved acceptance: results/runs/<timestamp>/ (summary only). Check in
this order, each with a CPU test or a 15-minute smoke-sized job, not a full run:
1. The trained weights are the ones being replayed (merge applied, right checkpoint loaded).
2. Training text matches inference: generate() feeds the drafter draft_prompt(prompt) + decoded
   target prefix, re-encoded every round. Compare with what the trainer tokenized.
3. Label mask and EOS handling.
4. Teacher-forced top-1 agreement rose but replay acceptance did not: then it is the bridge. Measure
   acceptance when the proposal is the teacher text itself re-tokenized by the drafter.
5. Learning rate too low or too high: look at the loss curves in the heartbeats.
Report the cause and the fix, resubmit p2 detached once. If none of the five explains it, stop and
write up what you ruled out; that goes to Fable.
```

## 5 · K policies on P2 · SOL · stretch

Write it while P2 trains. Fire it only after P2 has winners.

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

**Next:** numbers go into the Log. Applying the policies to P1 and P4 is tomorrow.

---

## Not today

- Checkpoint freeze and the 400-prompt test replay.
- Learning curve, leave-one-task-out, cache ablations.
- Anything on Modal. One warning for when stage 5 comes up: in the stage 1 smoke, live B1 matched B0 on only 34/40 (P2), 36/40 (P4), 39/40 (P1) prompts, and the stage 5 gate asks for 99%. Probe that before paying for the sweep.
- P3 and P5: their target is not recorded.
- The write-up. It is its own track and no iteration here produces it.
