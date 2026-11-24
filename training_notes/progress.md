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
