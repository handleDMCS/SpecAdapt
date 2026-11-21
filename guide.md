# PowerShell setup and Kaggle workflow

Use PowerShell 5.1 or PowerShell 7. The laptop manages files, credentials, submissions and result downloads. Scripts, model downloads and tests execute on Kaggle.

## Local setup

Git and uv are installed. The Windows environment is already configured:

```powershell
Set-Location -LiteralPath "F:\SpecAdapt"
$env:PYTHONUTF8 = "1"
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw "Dependency setup failed." }
```

Python 3.12 and the Kaggle dependency are pinned by the project. `uv run` uses the environment without activation or an execution-policy change.

## Credentials

Keep `KAGGLE_API_TOKEN` in the ignored local `.env`. The runner loads that file from the project root, even when launched from another working directory. Local authentication can be checked with:

```powershell
uv run --locked --env-file .env kaggle kernels list -m
if ($LASTEXITCODE -ne 0) { throw "Kaggle authentication failed." }
```

The remote preparation script also needs `KAGGLE_API_TOKEN` to publish private cache datasets. Kaggle CLI pushes do not carry Add-ons secret attachments. For a browser-started initialization, open the configured notebook, enable the secrets under **Add-ons → Secrets**, and use **Save Version → Save & Run All** on the already-uploaded preparation code. Local `.env` files and tokens are not bundled.

For Llama 3.1, use an `HF_TOKEN` or `HUGGING_FACE_TOKEN` secret with approved access to `meta-llama/Llama-3.1-8B-Instruct`. Missing credentials or denied access leave this model pending while other assets are prepared. Existing cached models do not need Hugging Face credentials to be read.

For CLI-only initialization, a separately authorized private credential dataset can supply the tokens: attach it with `--dataset owner/specadapt-run-credentials`. This is optional and stores actual token values in that private input until removed; it is never created automatically by the submission runner. The preparation script checks publishing credentials before large downloads begin.

## Submit Python scripts

```powershell
uv run python -m demo.kaggle_run src/demo/check_cuda.py
if ($LASTEXITCODE -ne 0) { throw "Kaggle job failed; inspect its result archive." }

# Replace these names with your root-relative script paths.
uv run python -m demo.kaggle_run first.py second.py
if ($LASTEXITCODE -ne 0) { throw "Kaggle job failed; inspect its result archive." }
```

The argument order determines execution order. All scripts run in the same Python process and share globals, imports, variables and model instances. Main blocks run normally. Errors stop subsequent scripts and are recorded in the combined log.

The runner snapshots Python files and TOML configuration into an ignored `build\submissions\<timestamp>` directory. It uploads one generated script with that snapshot embedded, polls Kaggle, and downloads one result archive.

The staged metadata uses the ID and compute settings from `kernel-metadata.json`; its `code_file` is replaced by the generated submission script. `--cpu` disables GPUs for the staged run without editing your metadata file.

Relative paths in submitted code refer to the bundled project root. New or changed files are collected as output. Explicit writes under `/kaggle/working` are collected into the same archive. Persistent model storage is kept outside these output folders.

## Initialize and reuse the cache

```powershell
uv run python -m demo.kaggle_run --cpu src/demo/prepare_data.py
if ($LASTEXITCODE -ne 0) { throw "Preparation failed; inspect the returned log." }
```

The preparation script follows `asset_plan.toml`: five models and five dataset sources representing the four tasks in the experiment plan. Source revisions are exact commit hashes. Dataset downloads use Parquet exports, including full MBPP rather than its sanitized subset.

The first run downloads missing assets into Kaggle's temporary storage and publishes private Kaggle Datasets. These persist independently of notebook sessions. The CLI finds the matching caches and attaches them on later submissions. It never transfers weights or datasets through the laptop.

Run the same command again to check mounted cache manifests, file sizes, model paths and actual per-split/total row counts. A complete cache hit does not require Hugging Face or publishing credentials. Interrupted initialization can reuse assets that were already published.

Prepare the cache and then run an experiment in one session:

```powershell
uv run python -m demo.kaggle_run src/demo/prepare_data.py your_experiment.py
```

The experiment receives the shared `MODEL_PATHS`, `DATASET_PATHS`, `DATASET_ROWS`, `PENDING_MODELS` and `CACHE_COMPLETE` values. The first-run paths point to temporary downloads; subsequent runs point to attached cache directories. Do not modify attached files.

This prepares source data only. Test/calibration subsets and model execution belong in later experiment scripts.

## One result file per submission

Each submission returns `results\submission-<UTC-submission-timestamp>.zip`:

- `run.log`: stdout, stderr, subprocess output and tracebacks in execution order.
- `submission.json`: script order, content hashes, timing and status.
- `outputs/`: files produced by the scripts.

For example, the GPU check's `cuda_spec.txt` is inside `outputs/`. The CLI prints the combined log. If a script fails, the result archive is still saved and the CLI returns a nonzero exit code.

Read the archive directly or extract it with PowerShell when needed:

```powershell
$latestResult = Get-ChildItem -LiteralPath .\results -Filter "submission-*.zip" |
    Sort-Object Name -Descending | Select-Object -First 1
if ($null -eq $latestResult) { throw "No submission result found." }
$extractedDirectory = Join-Path "results" $latestResult.BaseName
Expand-Archive -LiteralPath $latestResult.FullName -DestinationPath $extractedDirectory
Get-Content -LiteralPath (Join-Path $extractedDirectory "run.log") -Encoding UTF8
```

Extraction is optional and creates local copies; the runner itself returns only the one ZIP file.

## Validation and interrupted runs

```powershell
# Stage and check syntax only; no scripts run on the laptop.
uv run python -m demo.kaggle_run --dry-run --cpu src/demo/prepare_data.py

# Check submission setup, then stop waiting after 15 seconds.
# A timeout exits nonzero and leaves the remote job running.
uv run --locked python -m demo.kaggle_run --cpu --wait-timeout 15 --poll-interval 3 tests/run_tests.py

# Run meaningful workflow tests on Kaggle CPU using synthetic data.
uv run python -m demo.kaggle_run --cpu --no-cache tests/run_tests.py

# Reconnect to the same submission after a laptop disconnect.
uv run python -m demo.kaggle_run --resume SUBMISSION_TIMESTAMP
```

Use the UTC timestamp printed when the job was submitted. Resuming does not resubmit the scripts. `--wait-timeout` controls how long the laptop waits; reaching it does not cancel Kaggle execution.

Use `--metadata relative/path.json` to select another notebook and `--dataset owner/dataset-slug` to attach additional inputs. Avoid concurrent jobs targeting the same notebook because Kaggle's high-level status/output methods use the current run.

## Git tracking

Commit and push using separate PowerShell commands:

```powershell
git add -A
if ($LASTEXITCODE -ne 0) { throw "Git staging failed." }
git commit -m "run"
if ($LASTEXITCODE -ne 0) { throw "Git commit failed." }
git push
if ($LASTEXITCODE -ne 0) { throw "Git push failed." }
```

Review staged files before committing. The submission runner uploads the current file snapshot, including uncommitted edits, and records script hashes. It does not clone GitHub.

References: [uv projects](https://docs.astral.sh/uv/guides/projects/), [Kaggle kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [Kaggle datasets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
