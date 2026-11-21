**Scope and current state**

SpecAdapt uses **Windows PowerShell + uv for orchestration; Kaggle for all tests, downloads and model execution**. The submission runner and persistent asset preparation work. Pair 2 now has a greedy speculative decoder, naive text baseline, and M1 vocabulary/continuation bridge, exposed through `src/demo/demo_baseline.py` and `src/demo/demo_m1.py`. Calibration, M2/M3, stochastic decoding and the full benchmark remain planned in [experiment.md](experiment.md).

Verified on 2026-10-13: 33 workflow, bridge and decoder tests passed on Kaggle CPU ([test archive](../results/submission-20261013T070543756123Z.zip)). A prior fresh session reused all available caches without downloading assets from Hugging Face. The GPU smoke run detected **two Tesla T4s**, PyTorch `2.10.0+cu128`. Four models are cached; Llama 3.1 remains pending gated access.

**File map**

Layout verified on 2026-10-27: root Python scripts and demo infrastructure moved
to `src/demo/`; imports and commands updated. Dry-run staging, source syntax,
and wheel contents checked locally without running tests. Live CLI submission
`20261027T040702280467Z` (notebook version 21) authenticated, attached five ready
caches, and reached `running`. Local polling stopped at the intentional
15-second timeout; remote test results were not collected or verified.

| File / directory | Purpose |
|---|---|
| `src/demo/kaggle_run.py` | Local CLI: validate, bundle, attach inputs, submit, wait, fetch one ZIP. |
| `src/demo/kaggle_runtime.py` | Remote execution: shared namespace, output capture, artifact packaging. |
| `src/demo/prepare_data.py` | Remote asset download/publication or cache validation; exports paths and row counts. |
| `src/demo/assets.py` | Asset-plan validation, cache fingerprints, readiness polling and file validation. |
| `asset_plan.toml` | Exact Hugging Face repository IDs, commit hashes and selected files. |
| `kernel-metadata.json` | Local notebook ID and compute/input settings. Currently private `dungg1810/specAdapt`, GPU + internet enabled. |
| `src/demo/check_cuda.py` | GPU/environment smoke check; writes `cuda_spec.txt`. |
| `tests/run_tests.py`, `tests/test_workflow.py` | Remote stdlib `unittest` suite using synthetic assets. |
| `src/specadapt/bridge.py`, `src/specadapt/decoding.py` | M1 overlap/trie/continuation cache and greedy verification with KV rollback. |
| `src/demo/benchmark.py`, `pair2_demo.toml` | Fixed P2/GSM8K demo, package pins, target-only references and per-prompt metrics. |
| `src/demo/demo_baseline.py`, `src/demo/demo_m1.py` | Separate Kaggle entry scripts; prepend `src/demo/prepare_data.py`. |
| `src/demo/compare_demo.py` | Local analysis of downloaded result ZIPs, including paired bootstrap confidence intervals. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Local Python 3.12 environment; development dependency is `kaggle>=2.2.4`. |
| `build/submissions/<timestamp>/` | Ignored local staging: generated source, submission record and effective metadata. Keep for `--resume`. |
| `results/` | Ignored downloaded submission ZIPs. |

SpecAdapt is a standalone library: the wheel contains only `src/specadapt/`.
Kaggle and benchmark code lives in `src/demo/`, and Kaggle is a development
dependency. The placeholder `specadapt` CLI has been removed. From the repository,
use `uv run python -m demo.kaggle_run` for submissions.

**Commands: run from the project root**

```powershell
uv sync --locked

# GPU/environment smoke run.
uv run python -m demo.kaggle_run src/demo/check_cuda.py

# Scripts execute in argument order, within one remote Python session.
uv run python -m demo.kaggle_run first.py second.py third.py

# Validate caches, then run an experiment using the exported paths.
uv run python -m demo.kaggle_run src/demo/prepare_data.py your_experiment.py

# Cache check on CPU; available caches are attached automatically.
uv run python -m demo.kaggle_run --cpu src/demo/prepare_data.py

# Workflow tests on Kaggle CPU, without attaching real asset caches.
uv run python -m demo.kaggle_run --cpu --no-cache tests/run_tests.py

# Stage and syntax-check selected scripts; no Kaggle calls or script execution.
uv run python -m demo.kaggle_run --dry-run --cpu src/demo/prepare_data.py

# Reconnect after interruption; copy the UTC timestamp printed at submission.
uv run python -m demo.kaggle_run --resume SUBMISSION_TIMESTAMP
```

| Option | Effect |
|---|---|
| `--cpu` | Disable GPU/TPU and remove `machine_shape` in staged metadata. |
| `--metadata relative/path.json` | Select another notebook/config; default `kernel-metadata.json`. |
| `--dataset owner/slug` | Attach an extra Kaggle Dataset; repeatable. |
| `--no-cache` | Disable automatic cache attachment; retain configured/explicit inputs. |
| `--poll-interval SECONDS` | Status polling interval; default 10. |
| `--wait-timeout SECONDS` | Maximum status wait; default 50,400 (14 hours). |

Script paths must be **relative to the project root**. PowerShell backslashes work; quote spaces. Positional arguments are script paths only: there is no per-script argument forwarding.

**Execution contract**

1. The laptop bundles the current `.py` and `.toml` files, authenticates using `.env`, discovers matching private caches, and pushes a generated `submission.py` to Kaggle. Git commit/push is optional; submission does not clone GitHub.
2. Kaggle extracts the bundle into temporary project storage and runs scripts sequentially with shared globals/import state. Every script runs as `__main__`; main guards execute, and `__file__` identifies the current script. Relative file access starts at the bundled project root. Root, `src/` and the current script's directory are importable.
3. Exceptions or nonzero `SystemExit` stop later scripts. `SystemExit(0)` continues. Reusing global names can overwrite earlier values.
4. The runtime captures stdout, stderr, subprocess output and tracebacks, then packages generated files. The CLI waits, downloads the ZIP, prints its log, and exits nonzero on failure.

**One output file:** `results/submission-<UTC-timestamp>.zip`, where the timestamp is assigned on the laptop at submission, including microseconds. Inside:

- `run.log`: combined script output; displayed after completion, without live script-log streaming.
- `submission.json`: ordered scripts, selected-script SHA-256 hashes, cache owner, status, error and elapsed time.
- `outputs/`: new/modified project files; explicit new `/kaggle/working` files appear under `outputs/kaggle_working/`.

Excluded from upload: hidden files/directories, symlinks, `.venv`, `.git`, `build`, `results`, `logs`, Python caches, and file types other than `.py`/`.toml`. **JSON, CSV and arbitrary data files are not bundled**; attach them as Kaggle inputs or use supported configuration files. `.env` is never bundled. The manifest hashes selected scripts, not the complete repository snapshot.

**Persistent assets**

Caches are private Kaggle Datasets: **one per model, one shared dataset-source bundle**. Slugs follow `specadapt-<key>-<fingerprint-prefix>`, derived from the pinned asset definition. Existing matching caches are automatically mounted under read-only `/kaggle/input`. Fresh downloads and Hugging Face staging use `/tmp`, outside notebook outputs.

| Model key | Hugging Face repository | Last verified state |
|---|---|---|
| `llama_68m` | `JackFram/llama-68m` | Cached |
| `llama_160m` | `JackFram/llama-160m` | Cached |
| `vicuna_7b` | `lmsys/vicuna-7b-v1.5` | Cached |
| `qwen_7b` | `Qwen/Qwen2.5-7B-Instruct` | Cached |
| `llama_31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | Pending access approval |

| Dataset key | Total cached rows | Splits |
|---|---:|---|
| `gsm8k` | 8,792 | train 7,473; test 1,319 |
| `humaneval` | 164 | test 164 |
| `mbpp` | 974 | train 374; validation 90; prompt 10; test 500 |
| `alpaca` | 52,002 | train 52,002 |
| `xsum` | 226,711 | train 204,045; validation 11,332; test 11,334 |

`src/demo/prepare_data.py` exports these globals for subsequent scripts:

- `MODEL_PATHS[key]`: local directory containing weights/tokenizer/config files.
- `DATASET_PATHS[key]`: local dataset directory, retaining config/split subdirectories and Parquet files.
- `DATASET_ROWS[key]`: `total` and `splits` counts.
- `PENDING_MODELS`: unavailable gated model keys.
- `CACHE_COMPLETE`: false while any gated model is pending; the submission can still succeed.

Cache hits validate fingerprint, exact expected file list and file sizes; dataset counts come from Parquet metadata. This is not a content-checksum verification. A corrupt cache fails rather than silently downloading again. Changed revisions/file selections create new cache identities. Preparation caches raw sources only; it does not load models, quantize weights, choose benchmark subsets or generate calibration data.

**Credentials and missing Llama cache**

The local `.env` supplies `KAGGLE_API_TOKEN` for orchestration. Publishing new remote caches also needs that token **inside Kaggle**; gated downloads need `HF_TOKEN` or its supported alias `HUGGING_FACE_TOKEN` with approved repository access.

CLI pushes do not carry browser Add-ons secret bindings. The user already authorized the private input `dungg1810/specadapt-run-credentials`, containing both tokens. It is attached only explicitly; ordinary cache checks and experiments do not need it. Tokens remain in that private dataset until removed and do not automatically track changes to `.env`.

After Llama access is approved, initialize the missing cache with:

```powershell
uv run python -m demo.kaggle_run --cpu --dataset dungg1810/specadapt-run-credentials src/demo/prepare_data.py
```

This reuses available caches and downloads only the missing asset. Missing/denied gated access leaves Llama pending while other assets remain usable. Keep credentials out of printed output and generated artifacts.

**Constraints for other agents**

- Execute tests and experiments on Kaggle. Local dry runs validate/stage without running supplied scripts.
- The runner does not install the project's dependencies remotely. Preparation lazily installs its own missing dependencies; experiment scripts must arrange their required Kaggle packages.
- Submit serially to each notebook ID. Kaggle status/output APIs follow the current run; use separate metadata/notebook IDs for independent jobs.
- `--resume` waits/fetches the original submission using local staging. It does not rerun scripts or implement experiment checkpoints. A timeout/interrupted CLI does not cancel the remote job.
- The local demo metadata now pins `machine_shape = NvidiaTeslaT4` and the baseline's Kaggle image digest (PyTorch `2.10.0+cu128`). The decoder uses device 0 only. Separate notebooks initially selected different PyTorch images, so use the pinned metadata or `src/demo/demo_compare.py` for comparable measurements.
- Keep weights/downloads outside generated-output directories. The ZIP collects generated files; persistent assets belong in private Kaggle Datasets.
- Treat [experiment.md](experiment.md) as the intended research plan, not a list of implemented experiment features.

Evidence: [fresh-session tests and cache reuse](../results/submission-20261013T051435880249Z.zip), [GPU smoke run](../results/submission-20261013T052916251728Z.zip). Both archives are local ignored artifacts.
