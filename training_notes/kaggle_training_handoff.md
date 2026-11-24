# Kaggle training handoff

The Wave 1 oracles are complete. Start the next session with dataset preparation and adapter experiments on Kaggle T4. Use **P2** (`llama_68m` → `qwen_7b`) to establish the training recipe, then apply the selected recipe to P1 and P4. [Training plan](training.md#6-training-and-selection-kaggle-t4) · [Phase summary](training_phases.md)

## Starting inputs

- Frozen prompts live in `training/dataset/`. The `specadapt` Modal Volume holds the same prompt files under `dataset/`. [Dataset layout](data_layout.md)
- The verified local oracle dump is `results/modal/20261123T154925641802Z/oracle-dump/`. Vicuna and Qwen each have **1,440 fit-base, 320 validation, and 400 held-out test rows**, with a manifest and checksums. P1 uses Vicuna and P2/P4 share Qwen.
- Kaggle replay passed on **40 validation prompts per pair** for P1, P2, and P4. The smoke verified prompt and tokenizer provenance and reproduced the saved token sequences on T4. [Smoke result](../results/submission-20261124T014249246739Z.json)

## Next iterations

### 1. Build and validate the Kaggle training inputs

Build derived `fit-base.jsonl` and `val.jsonl` for each target. Join the canonical prompt rows to the oracle rows by `prompt_id` and verify the prompt hash before emitting a record. Require exact split coverage, unique IDs, and the pinned target revision and tokenizer fingerprint. Carry the prompt, task, source dataset, teacher text, target token IDs, and stop reason into the derived input. Package the joined fit and validation files as versioned private Kaggle inputs and record their checksums. The raw sources serve as the audit trail for the join. [Kaggle input attachment](../src/demo/kaggle_run.py)

### 2. Establish the P2 baseline and train the adapters

Implement and test the pending adaptive-K policies and adapter trainer. Start by measuring P2 B1 at fixed `K = 3` and T0 on all 320 validation prompts. T0 uses the vocabulary bridge, continuation caches, and an entropy-based K policy. Then train T1 and T2 on fit base. T1 fits the output bias and a rank-8 or rank-16 `lm_head` delta. T2 fits rank-16 LoRA. Re-tokenize `draft_prompt(prompt) + answer_text` with the draft tokenizer, mask the prompt with `-100`, and add draft EOS only for teacher records that ended at EOS. Verify merged and unmerged logits agree within `1e-3`. [Work items](training.md#2-work-items-in-order) · [Training details](training.md#6-training-and-selection-kaggle-t4)

### 3. Fit the stopping rule and select on validation

For each trained tier, replay fit base to collect accepted/rejected labels and fit the L2-regularized `LogisticKPolicy` using entropy, top-1 probability, the top-1/top-2 margin, and cache hit. Tune the threshold on validation. On P2, compare **360, 720, and 1,440** fit prompts against the same 320 validation prompts. Consider recording `fit/extra` only if validation acceptance improves by **at least 1 percentage point** from 720 to 1,440. Track validation acceptance by task and apply the plan’s adapter-overhead limit before freezing the chosen checkpoints. [Selection rules](training.md#6-training-and-selection-kaggle-t4)

## Run and evaluation gates

- Run the repo’s Kaggle CPU suite after code changes and before requesting a GPU job. Use the dedicated test metadata when available in the workspace.

  ```powershell
  uv run --locked python -m demo.kaggle_run --metadata build/kernel-tests-metadata.json --cpu --no-cache tests/run_tests.py
  ```

- Keep the held-out oracle as a separate evaluation input. Commit `checkpoints_frozen.toml` with the chosen checkpoint hashes before test replay. Report replay acceptance on the 400 test prompts and measure live speed in the separate 200-prompt Modal core run. [Evaluation plan](training.md#7-evaluation)
