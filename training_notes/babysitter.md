# P2 babysitter sheet

Iteration 3, Tue 24 Nov 2026. Times are local (UTC+7); heartbeat clocks are UTC, so add 7 hours. Two Kaggle T4 jobs train the `llama_68m` drafter to match `qwen_7b`, LoRA (T2) first. They run on Kaggle, so closing the laptop or the watcher does not stop them.

- Smoke: **pass**, `20261124T050647634040Z`. B1 gate exact (2,610 / 13,938), 4.9 min total, replay 1.29 s per prompt. Full P2 should take 35 to 45 min once it leaves the queue, T2 line about 25 min in.
- Full P2: fired 12:20, `20261124T052050221490Z`, kernel `specadapt-p2` (step 1 done; do not fire it again)
- Code: `7004468`, Kaggle CPU suite 72/72

**Quick check, any time:** `uv run python -m training.training_red_flag <timestamp>` (add `--visual` for a chart) prints one line per check (`green-flag`, `neutral`, `red-flag`) and an overall verdict. It reads the live Kaggle log while the job runs and the collected result afterwards. `neutral (wait)` means too early to judge and does not count toward the verdict. On `red-flag`, use the failure table below. The expectations it checks against are in `EXPECT` at the top of `training/training_red_flag.py`.

## Timeline

P2 was fired at 12:20, so the P2 rows below may come about 5 minutes early. Training minutes are estimates until the smoke reports real ones.

| Time | Event |
| --- | --- |
| 12:06 | Smoke fired |
| 12:25 | Fire full P2 (step 1). Smoke verdict around now, at the latest 12:30 (20-minute cap) |
| 12:30 | P2 finishes B1 replay on 320 prompts (untrained baseline, should be near 18.7% acceptance) |
| **13:10** | **First P2 result: the `phase=result tier=T2` line.** LoRA weights are already saved at this point |
| 13:35 | P2 complete with T1 too. Collect (step 4) |
| 14:25 | P2 hard stop (120-minute cap); a job still running aborts itself |
| 14:30 | Last sensible time to fire a full P2 rerun. After this, fire `p2-lora` instead (about 40 min) |
| 15:05 | Last time to fire `p2-lora` and still read its result before leaving |
| 15:45 | Before-you-leave checklist (step 5) |
| 16:00 | Meeting. Jobs keep running on Kaggle |

## Steps

### 1. Fire the full P2 run · now · 1 min

```powershell
uv run --locked python -m demo.kaggle_run training/kaggle/train_pair.py --detach --kernel p2 --job training/jobs/p2.toml --dataset dungg1810/specadapt-train-inputs --expected-minutes 70
```

- Fired at 12:20 as `20261124T052050221490Z`. Do not run it again: a new push to the same notebook replaces the running version.
- If Kaggle refuses a second GPU session, wait for the smoke to end and run the same command again.

### 2. Check the smoke verdict · 12:25 to 12:30

Watch live (beeps when the job ends or goes silent for 10 minutes; Ctrl+C stops only the watcher):

```powershell
uv run python -m demo.kaggle_watch 20261124T050647634040Z
```

When it says `Kaggle: complete` or `error`, collect:

```powershell
uv run python -m demo.kaggle_run --resume 20261124T050647634040Z
```

- **Pass:** `"verdict": "pass"`. Nothing to do; P2 is already running. Glance at `minutes_per_phase`: if `T2_train` is far above 1 minute, the P2 times above will slip.
- **Fail:** see the failure table and use prompt 3.b.

### 3. Watch P2 and read the T2 line · from about 12:30, glance every 15 min

```powershell
uv run python -m demo.kaggle_watch 20261124T052050221490Z
```

The line to wait for, around 13:10 (numbers made up):

```text
[heartbeat 06:10:02Z] phase=result tier=T2 acceptance=0.2512 b1=0.1873 gain_pp=6.39 math=0.27 code=0.31 instruction=0.22 summarization=0.19
```

- `gain_pp` is percentage points over the untrained drafter. **2 or more means the recipe works.**
- The Kaggle kernel page shows the same log if the watcher misbehaves: `kaggle.com/code/dungg1810/specadapt-p2`.

### 4. Collect P2 and decide · when the watcher says complete, about 13:35

```powershell
uv run python -m demo.kaggle_run --resume 20261124T052050221490Z
```

- Numbers: `results/runs/20261124T052050221490Z/summary.json`. Look at `t2_gain_pp`, `t1_gain_pp` and `tiers.T2.per_task`.
- Weights: `results/runs/20261124T052050221490Z/artifacts/t2.safetensors` (LoRA) and `t1.safetensors`.
- **`t2_gain_pp` ≥ 2:** done for today. The next session runs P4 and P1 (block 6), then K policies (block 5).
- **Below 2, or T2 below B1:** still a result. Paste prompt 4.b (in `iteration.md`) into Opus before spending GPU on P1 and P4. After 15:00, leave that for after the meeting.

### 5. Before you leave · 15:45

- Something is either finished and collected, or running. Never leave with nothing fired.
- A job still running is fine: it caps itself (P2 120 min, `p2-lora` 60 min) and keeps its output. Collect after the meeting with the same `--resume` command.
- Note the timestamps you fired and the T2 line, so the next session can start from them.
- Do not start an Opus fix session you cannot finish before 16:00.

## If something goes wrong

| You see | Meaning | Do |
| --- | --- | --- |
| Smoke `verdict: fail` | The trainer broke; P2 likely breaks the same way | Prompt 3.b with the smoke timestamp. After 14:30, add "the short path is needed" |
| `phase=abort reason=…` | The job stopped itself on a failed check | Collect it, then prompt 3.b with that timestamp |
| `P2 smoke B1 counted (…), expected (2610, 13938)` | Untrained replay no longer matches stage 1 (the smoke gate) | Prompt 3.b. P2 is probably affected too |
| `STALLED` from the watcher | No heartbeat for 10 minutes; usually queuing or a dropped log stream | Open the kernel page. Log moving: ignore. Kaggle error: collect and use 3.b |
| `Kaggle: queued` for a long time | No GPU assigned yet | Wait. Still queued at 14:30: cancel on the kernel page and fire `p2-lora` |
| `…-oom-retry` | Out of GPU memory; batch halved and retried. A second OOM aborts | Nothing, unless it aborts |
| Second submit refused | Only one GPU session allowed right now | Fire P2 again after the smoke ends |
| `Kaggle: complete` | Finished and wrote `summary.json` | Collect with `--resume` |

**Prompt 3.b (paste into Opus):**

```text
Iteration 3.b. A run failed: results/runs/<timestamp>/ after collecting it (summary and the printed
log tail only). Find the cause, reproduce it in a CPU test, fix, rerun the CPU suite, commit.
Resubmit detached in fresh kernels (`p2-smoke-r2`, `p2-r2`) so a still-running version is not
disturbed: the smoke again, and with it either the full p2 or, when I say the short path is needed,
p2-lora. Print the run cards and commands again. If this is a second failure for the same reason,
stop and write what you know.
```

**Short path, if time runs out:**

```powershell
uv run --locked python -m demo.kaggle_run training/kaggle/train_pair.py --detach --kernel p2-lora --job training/jobs/p2-lora.toml --dataset dungg1810/specadapt-train-inputs --expected-minutes 40
```

## What is being trained

| Setting | Smoke (fired) | Full P2 | `p2-lora` (short path) |
| --- | --- | --- | --- |
| Pair | P2: drafter `llama_68m` learns to propose what target `qwen_7b` would say | same | same |
| Data | 64 fit prompts, 40 val (10 per task) | 1,440 fit, 320 val | 1,440 fit, 320 val |
| Tasks | math, code, instruction, summarization | same | same |
| Order | B1, T2, T1 | B1, T2, T1 | B1, T2 |
| T2 LoRA | rank 16, lr 1e-4, 2 epochs | rank 16 on all 7 attention and MLP projections; lr 1e-4, 3e-4, 1e-3; 4-epoch schedule scored at epochs 2 and 4 (6 tries) | rank 16, lr 3e-4, scored at epochs 2 and 4 (2 tries) |
| T1 output adapter | rank 8, lr 1e-3, 2 epochs | ranks 8, 16 × lr 1e-3, 3e-3, 1e-2; 2 epochs (6 tries) | not run |
| Time cap | 20 min | 120 min | 60 min |
| Expected | about 15 min | about 70 min, T2 line at about 45 min | about 40 min |

Shared by all three:

- **Winner picked by** the highest top-1 agreement with the recorded target text on validation answers, then merged and replayed.
- **Scored by** replay acceptance at fixed K = 3, greedy, up to 256 new tokens, checked against the recorded Qwen oracle.
- **Precision:** train FP32 (no autocast), AdamW, no weight decay, 5% warmup then cosine; merge check below 1e-3; replay FP16.
- **Batch limit:** 16 sequences or 16,000 tokens; halves once on out-of-memory.
- **Hardware:** Kaggle Tesla T4, device 0, pinned image, seed 42.
- **Outputs:** `summary.json`, `artifacts/t2.safetensors`, `artifacts/t1.safetensors`, `artifacts/validation.json`.

## Reading the live log

Phases in the order they appear. Each line has `step=n/total` and an `eta` once a phase is under way.

| Phase | What is happening | Metric to watch |
| --- | --- | --- |
| `preflight` | Checks inputs, installs packages, loads tokenizers, downloads the drafter. Capped at 5 minutes | `action=` |
| `preflight-T2`, `preflight-T1` | One-example training and merge check per tier | |
| `B1-replay` | Untrained drafter replayed on validation | `acceptance`, near 0.187 |
| `T2-fit-1` … `T2-fit-3` | LoRA training, one per learning rate | `loss` should fall; NaN aborts |
| `T2-score-i-epoch-e` | Teacher-forced agreement on validation at a checkpoint | `top1_agreement` |
| `T2-replay` | Best LoRA, merged, replayed | `acceptance` |
| **`result tier=T2`** | **The LoRA result** | `gain_pp` |
| `T1-fit-1` … `T1-fit-6`, `T1-score-…`, `T1-replay` | Same for the output adapter | `loss`, `acceptance` |
| **`result tier=T1`** | The T1 result, then the job ends | `gain_pp` |
