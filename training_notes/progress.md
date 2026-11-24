# Progress against the training plan

Snapshot from 2026-11-24 at base commit `640c25f`, with uncommitted Modal scripts. [Training plan](training.md) · [Kaggle handoff](kaggle_training_handoff.md)

Wave 1 test and training oracle recordings are complete for P1, P2, and P4. Kaggle passed both the same-prompt smoke replay and the validation replay. Stage 4 fitting and selection is pending, and the stage 5 live sweep remains outstanding.

## Work items (§2)

| # | Status | Evidence |
|---|---|---|
| W1 Vicuna prompt fallback | ✅ Verified | The [template test](../tests/test_bridge.py) covers Vicuna with `chat_template = None`. |
| W2 Bridge exact-piece / `direct` | ⚠️ Implemented, P1 check pending | The [bridge](../src/specadapt/bridge.py) has exact-piece matching and a `direct_hits` counter. The P1 smoke used the same-tokenizer baseline, so it did not exercise that path. |
| W3 Oracle FP16, splits, recording/replay GPU | ✅ Verified | Both targets have FP16 test and train manifests. Kaggle T4 replayed the A10G validation oracle with provenance checks. [Replay result](../results/submission-20261124T014249246739Z.json) |
| W4 Remove T4 / `/kaggle/input` hard-codes | ❌ **Still present** | `src/demo/benchmark.py:18` and `:99` |
| W5 `build_splits.py` + `splits.toml` | ⚠️ Partial | The builder and committed split manifest exist. The manifest matches the oracle dump checksum. The `test_splits.py` gate is missing. |
| W6 Modal harness + ledger | ✅ Operational | The Modal runs used the Volume, spend guard, and spend accounting in the [harness](../modal/modal_app.py). |
| W7 Recorders with resume | ⚠️ Implemented, resume test pending | Both [overnight](../modal/overnight_oracles.py) and [stage 2](../modal/stage2_test_oracle.py) recorders skip existing prompt IDs and stop before timeout. A passing resume test is still unverified. |
| W8 `adaptive_k.py` (Entropy/Logistic K policies) | ❌ | File does not exist |
| W9 `alignment.py` (adapters, `M2Trainer`) | ❌ | File does not exist |
| W10 Ratio bootstrap and breakdowns | ⚠️ Early version only | A paired bootstrap exists in `src/demo/compare_demo.py:49`. There is no ratio bootstrap or breakdown report for the new splits. |
| W11 Stochastic verification | ❌ | Not started |

## Stages (§5), Wave 1

| Stage | Status |
|---|---|
| 0. Asset sync | ✅ Complete. Model and dataset revisions are pinned in the [run budget](../results/modal/20261123T154925641802Z/oracle-dump/budget.json). |
| 1. Smoke | ✅ Live smoke and same-prompt replay complete. Draft-plus-bridge to target time ratios were 0.057 for P1, 0.062 for P2, and 0.231 for P4, all below 0.5. Batched throughput was 199.5 tok/s for Qwen and 79.0 for Vicuna, both below 250. The [original Kaggle smoke replay](https://www.kaggle.com/code/dungg1810/specadapt), submission `20261123T144002663579Z`, passed the `≤ 0.005` acceptance-delta gate for all three pairs (P1 `0.00112`, P2 `0.00183`, P4 `0.00053`). |
| 2. Test oracle | ✅ Complete. Both targets have 400 test rows with verified counts and checksums. [Oracle dump](../results/modal/20261123T154925641802Z/oracle-dump) |
| 3. Train recording | ✅ Complete at the reduced recording size. Each target has 1,440 fit-base and 320 validation rows with verified counts and checksums. `fit/extra` is deferred. |
| ↳ Kaggle validation replay | ✅ Passed on 40 `train/val` prompts per pair. Untrained replay acceptance was 28.2% for P1, 18.7% for P2, and 24.3% for P4. [Replay result](../results/submission-20261124T014249246739Z.json) |
| 4. Fit and select (Kaggle) | ❌ Pending. The [Kaggle handoff](kaggle_training_handoff.md) starts with preparation of the training inputs. |
| 5. Live sweep | ❌ Pending. |
| Wave 2 (P3/P5) | ❌ Pending. Eligibility depends on approved model access and spend of at most $15.00 after Wave 1 stage 5. |
| 6. Stochastic check | ❌ Pending. Requires W11 and the stage 6 budget gate. |

## Spend and open checks

- **Modal spend.** Recorded spend for asset sync, smoke, and overnight oracle recording totals **$3.52**. The Volume ledger (`/vol/specadapt/ledger.jsonl`) records $0.47 for asset sync and smoke, including $0.45 for smoke. Four `spend-*.json` receipts under `/vol/specadapt/oracles/20261123T154925641802Z/` total $3.04. The overnight run reserved $7.65. [Run budget](../results/modal/20261123T154925641802Z/oracle-dump/budget.json) · [Spend accounting](../modal/modal_app.py)
- **Stage 4 prerequisites.** Prepare the Kaggle training inputs and implement W8 (`adaptive_k.py`) and W9 (`alignment.py`). W4, the split test gate, and the W2 and W7 verification checks are still open. [Kaggle handoff](kaggle_training_handoff.md)
- **Audit trail.** The live smoke reports record `8c3dea7`. The overnight budget and Kaggle validation result record `640c25f`. The overnight recorder checksum matches the current script. `modal/dump_oracles.py`, `modal/kaggle_oracle_smoke.py`, `modal/overnight_oracles.py`, and `kaggle_training_handoff.md` are still uncommitted.
- **CPU tests.** Three Kaggle CPU runs on 2026-11-23 passed 45 tests each. A test run for the current working tree is pending. [Latest passing result](../results/submission-20261123T141758931693Z.json)

## Log

**Iteration 1 · launch pad · 2026-11-24 · DONE** (`9bd7e58` baseline commit of the Modal scripts and notes, then `f2ac2f0`, `c011e7b`)
- `kaggle_run`: `--detach` (run card, exit 0), `--kernel <name>` → `<owner>/specadapt-<name>`, `--job training/jobs/<job>.toml` (validated locally, read in the job with `demo.kaggle_runtime.job_config()`), git SHA and dirty flag in every submission. `--resume <ts>` downloads `artifacts/` into `results/runs/<ts>/` with size and SHA-256 checks, and prints only `summary.json` (+ last 20 log lines on failure) when the job wrote one.
- `kaggle_runtime`: `heartbeat(phase, step, total, **metrics)` reaches the live Kaggle log, and auto-aborts on a NaN or infinite float metric and on any heartbeat after the job's `max_minutes`. `abort(reason)` fails the job. Files under relative `artifacts/` stay as kernel output files, listed and not embedded. A failed job always gets `verdict: fail` in `summary.json`.
- Watch from a terminal: `uv run python -m demo.kaggle_watch <ts>` (heartbeats, STALLED after 10 minutes of silence, end notice). It uses Kaggle's session-log stream through SDK internals, so a `kaggle` upgrade could break it; the kernel page is the fallback.
- Verified on Kaggle CPU: CPU suite 59/59 passed (`20261124T035728993070Z`). Probe `20261124T035729226906Z`: 12/12 heartbeats live, 1 MB artifact SHA-256 matched. Abort probe `20261124T035735874445Z`: NaN at step 3 aborted the job, `verdict: fail`, artifact still returned and verified. Watcher run `20261124T040051760583Z`: all heartbeats shown live, exit 0.
- Local Windows runs of `tests/test_workflow.py` fail 4 runner tests both before and after this change (8.3 temp paths, CRLF writes). Kaggle Linux is the reference.
- GPU-hours used: 0. Open: none for iteration 2.

**Iteration 2 · data and trainer · 2026-11-24 · DONE**
- Published [private training inputs](https://www.kaggle.com/datasets/dungg1810/specadapt-train-inputs). Qwen and Vicuna each have 1,440 fit and 320 validation rows. All five published files matched the verified local build byte for byte.
- P2 smoke, P2, P1 and P4 bundle successfully. T1 fits for two epochs. T2 scores at epochs two and four on a four-epoch schedule, with a two-epoch schedule in smoke. Both tiers select by validation answer-token top-1 agreement and save the unmerged winners as safetensors.
- Kaggle CPU suite passed 69/69 (`20261124T045042312564Z`). The pinned draft tokenizers fit every joined example in context, and both selected adapter checkpoints saved, reloaded and merged on tiny Llama models.
- GPU-hours used 0. Iteration 3 starts with the P2 T4 smoke and its B1 gate of 2,610 accepted out of 13,938 proposed tokens.
