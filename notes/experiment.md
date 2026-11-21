**tl;dr:** Same 5 pairs and 4 datasets as before. All tests and model runs happen on Kaggle, with 4-bit targets on a single T4. The laptop uses PowerShell for editing, job submission and result downloads. Acceptance rate is the headline metric. Speedup is secondary and labelled "T4, 4-bit target". Changes from v1:
- a naive cross-tokenizer baseline
- fixed prompt format and data splits
- calibration data built in one cheap pass
- a 4-bit sanity check
- Tier 0 ablations
- Kaggle settings pinned

**Model pairs (5)**

| # | Drafter → Target | Role |
|---|---|---|
| P1 | Llama-68M → Vicuna-7B | same-tokenizer control |
| P2 | Llama-68M → Qwen2.5-7B-Instruct | cross-tokenizer, paper-matched |
| P3 | Llama-68M → Llama-3.1-8B-Instruct | cross-tokenizer, paper-matched |
| P4 | Llama-160M → Qwen2.5-7B-Instruct | drafter size vs latency |
| P5 | Llama-160M → Llama-3.1-8B-Instruct | cross-tokenizer |

- **Pin versions:** record each model's exact Hugging Face ID and revision hash in the config.
- **Asset configuration:** `asset_plan.toml` pins the five model repositories and their exact revisions. Vicuna uses `lmsys/vicuna-7b-v1.5`; the drafters use `JackFram/llama-68m` and `JackFram/llama-160m`.
- **Cut:** P6 (Llama-160M → Vicuna-7B) and Qwen2.5-14B are stretch goals.
- **Dropped drafters:** SmolLM2-135M and Qwen2.5-0.5B. Optionally time their c in the smoke test to confirm the cut.
- **Advisor flag:** you have 5 pairs instead of 6. P2–P5 meet the 4 cross-tokenizer quota.

**Prompt format (frozen across all pairs and tiers)**
- **Target:** its own chat template, rendered to text once.
- **Drafter:** a plain-text template (`{prompt}\n\n`) followed by the shared generated continuation. It never sees the target's special markers.
- **Code:** store both templates in code and test them on Kaggle.

**Datasets (4)**

| Task | Test (100 fixed prompts) | Calibration | Answer text used for calibration |
|---|---|---|---|
| GSM8K | `test` split | 2,000 from `train` | reference solution |
| Code | HumanEval (100 of 164) | all of MBPP (974) | reference code |
| Alpaca | 100 from fixed-seed shuffle | 2,000 from same shuffle, disjoint | reference output |
| XSum | `test` split | 2,000 from `train` | reference summary |

- **Fixed IDs:** commit all test and calibration IDs to the repo. A test on Kaggle asserts that the two sets have zero overlap.
- **If MBPP must be in the test set** (paper-matching): test on 50 HumanEval + 50 MBPP `test`, and calibrate on MBPP `train`, `validation` and `prompt` (474).
- **Source cache:** `src/demo/prepare_data.py` caches GSM8K (`main`), HumanEval, full MBPP, Alpaca and XSum as pinned Parquet exports. It logs actual per-split and total row counts. Test/calibration selection remains a separate experiment step.

**Settings**
- **Decoding:** temperature 0.01 if it matches OmniDraft, otherwise greedy (T=0), with `max_new_tokens=128` and 1 seed.
- **Extra run:** GSM8K, P2, T=1, with 3 seeds.
- **Statistics:** use a paired bootstrap over prompt IDs for every tier-vs-baseline comparison. That means resampling the per-prompt difference, not comparing two separate confidence intervals.

**Metrics logged per prompt**
- Acceptance rate.
- Accepted tokens per round.
- Wall time per token.
- Adapter overhead in ms/token, against the ≤10% of one draft step budget.
- Exact match with target-only output. This is a bug alarm, not a quality score: expect it near 100%, with small drift from 4-bit numerics.

**Run ladder**

| Run | What | Pairs |
|---|---|---|
| B0 | Target only, no drafter | 3 targets |
| B1 | Plain speculative decoding (P1); naive text bridge (P2–P5): decode the draft to text and re-tokenize it with the target tokenizer, k=3 | all |
| T0 | Tier 0: vocabulary bridge + n-gram cache + entropy-based K | all |
| T0-ablate | T0 minus the n-gram cache; T0 with fixed k=3 | P2, P3 × GSM8K, Alpaca |
| T1 | Tier 1: logit bias + low-rank head + logistic-regression K | all |
| T2 | Tier 2: LoRA + distillation | all |

**What runs where**
- **Laptop (PowerShell):** edit code and settings, manage Git and Kaggle credentials, submit jobs, and download results. No tests or model execution run locally.
- **Kaggle (code tests):** `pytest` with fake token IDs and made-up probabilities. It covers:
  - the n-gram cache
  - the token mapping
  - the naive bridge
  - the K-stopping rule
  - both prompt templates
  - split overlap
  - the JSONL resume logic

  No real model, tokenizer download or forward pass runs here.
- **Kaggle:** everything else.

**Smoke test (first Kaggle run)**
- **Scope:** 50 GSM8K prompts per pair, running B0 and B1, k=3.
- **c:** the time for one drafter forward step divided by the time for one target forward step, measured on the T4. For P2–P5, add the bridge cost.
- **Kill rule:** drop any pair with c > 0.5.
- **Bug checks:** results far above the paper's baselines likely mean a bug, and so does exact match well below 100%.

**4-bit check (once)**
- Run P1, B1, on 50 GSM8K prompts, with Vicuna-7B in fp16 across 2×T4 vs in 4-bit on 1×T4. Compare acceptance rates.
- If they are within about 2 points, state that in the thesis. If not, report the gap next to every OmniDraft comparison.

**Calibration data (for T1/T2)**
- **One pass, no generation:** feed each prompt plus its reference answer to the target in a single batched forward pass.
- **What to save:** the top-20 token IDs and their log-probabilities at every answer position. That is about 120 MB per target, so it stays under the 20 GB working dir.
- **Output:** save it as a Kaggle dataset version and attach it to the T1/T2 kernels.
- **Limitation (state in thesis):** the reference answers come from the dataset, not from the target itself.

**Kaggle run plan**
- **One kernel per target:** 3 kernels (Vicuna, Qwen, Llama-3.1). Each loads its target once, then loops over drafters, runs and datasets.
- **`kernel-metadata.json`:**
  - `"enable_gpu": true` and `"enable_internet": true`.
  - `"machine_shape": "NvidiaTeslaT4"`. Without this, the default may be a P100, which current PyTorch doesn't support.
  - For the 4-bit check, use a 2×T4 shape.
- **Submission CLI:** `uv run python -m demo.kaggle_run script1.py script2.py` bundles root-relative Python scripts and runs them in the supplied order, sharing a notebook-like namespace. The generated submission records the exact submitted script hashes; it uploads the snapshot directly rather than cloning GitHub.
- **Persistent preparation:** `uv run python -m demo.kaggle_run --cpu src/demo/prepare_data.py` downloads missing assets on Kaggle and saves them as private Kaggle Datasets. Later submissions automatically attach the matching caches; reruns validate files and log model paths and dataset row counts.
- **Secrets:** Kaggle CLI pushes do not carry Add-ons secret attachments. Initialize once from the browser with the secrets enabled, or attach a separately authorized private credential input for CLI-only initialization. `KAGGLE_API_TOKEN` publishes private caches; `HF_TOKEN` or `HUGGING_FACE_TOKEN` downloads gated Llama-3.1. Missing or denied Hugging Face access leaves Llama-3.1 pending while the other assets are cached. The laptop `.env` is never bundled.
- **Storage:** download staging and `HF_HOME` stay under `/tmp`, outside `/kaggle/working`. Persistent assets live in attached read-only Kaggle inputs, not in notebook outputs.
- **One output file:** each submission yields `results/submission-<UTC-timestamp>.zip`, containing the combined log, submission metadata and the scripts' generated files. Use `--resume <timestamp>` to recover the same submission after a laptop disconnect.
- **Memory:** 4-bit targets take about 4–6 GB each, so the target, drafter and KV cache fit in one T4's 16 GB.
- **Results:** JSONL, one row per prompt. Each row records:
  - pair, run, task, prompt ID and seed
  - the metrics above
  - GPU name, commit SHA, config hash
  - torch, transformers and bitsandbytes versions
- **Resume:** the script skips prompt IDs already in the JSONL. To resume after a crash, attach the previous run's result archive via `kernel_sources` and extract the prior JSONL before continuing. `--resume <timestamp>` in the submission CLI reconnects the laptop to a run; it does not rerun experiment code.

**Budget (assumes ~15 tok/s; measure it in the smoke test)**

| Item | GPU-hours |
|---|---|
| Smoke test (5 pairs, B0 + B1, 50 prompts) | ~1 |
| 4-bit check | ~0.5 |
| Calibration pass (3 targets) | ~3 |
| Full passes: B0+B1, T0, T1, T2 (~7 h each) | ~28 |
| T0 ablations | ~1 |
| T=1 seed runs | ~1 |
| T1 fitting + T2 LoRA training (small drafters, cached logits) | ~5 |
| **Total** | **~40** |

- **Quota:** that is about 1.5 weeks of the roughly 30 h/week quota. Keep about 1 week spare for reruns, and check your account page, since the quota floats.
