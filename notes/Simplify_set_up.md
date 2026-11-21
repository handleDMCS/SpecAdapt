**Overview**
tl;dr: You edit the tool in PowerShell on your laptop, commit it, then submit Python scripts to one Kaggle session and pull back one timestamped result archive. The CLI uploads the current file snapshot and records script hashes. Model and dataset caches persist as private Kaggle Datasets.

**Laptop: Edit & Submit**
**tl;dr**: You write the code and prepare jobs in Windows PowerShell. Tests and experiments execute on Kaggle.

**Why**: The laptop manages files, dependencies, credentials and submissions. Kaggle provides the execution environment.

**Input**:
```
- The plan for what the tool should do
- Kaggle credentials in the ignored .env file
- A settings file (config)
```

**Output**:
```
- Working code
- A configured Kaggle submission
- Downloaded results from the remote run
```

**Technical keywords**: PowerShell, uv, pyproject.toml, kaggle_run.py

**Commit & Push: Pin the Version**
**tl;dr**: You save your code as one labeled snapshot and upload it to GitHub.

**Why**: Gives the GPU machine one exact, fetchable version of the code, so every result traces back to it.

**Input**:
```
- Edited code and settings ready for remote testing
```

**Output**:
```
- A timestamped snapshot with a unique ID
- A copy of that snapshot online
```

**Technical keywords**: git commit, git push, commit SHA

**Kaggle: Run on GPU**
**tl;dr**: One command tells a cloud GPU machine to download your snapshot and run the real experiment.

**Why**: Testing, model downloads and heavy computing happen on Kaggle, using the submitted snapshot.

**Input**:
```
- Root-relative Python script paths, in execution order
- The settings file for this run
- A secret access key (only for restricted models)
```

**Output**:
```
- Raw results from the experiment
- A record of which code and settings made them
```

**Technical keywords**: kaggle kernels push, kernel-metadata.json, HF_TOKEN, T4 GPU

**Fetch: Bring Results Home**
**tl;dr**: You download the finished results file from the cloud machine to your laptop.

**Why**: Closes the loop so you can inspect and compare results locally.

**Input**:
```
- A completed cloud run
```

**Output**:
```
- A local results file, labeled by time and tied to its snapshot ID
```

**Technical keywords**: kaggle kernels output

**PowerShell commands** (current scaffold):
```powershell
Set-Location -LiteralPath "F:\SpecAdapt"
$env:PYTHONUTF8 = "1"
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw "Dependency setup failed." }
uv run python -m demo.kaggle_run src/demo/check_cuda.py
if ($LASTEXITCODE -ne 0) { throw "Kaggle job failed." }
```

The CLI loads `.env`, runs the requested scripts in argument order with shared variables, waits for Kaggle, and downloads `results\submission-<UTC-timestamp>.zip`. This one file contains the combined log, submission metadata and any generated files. Only orchestration runs locally. See [the PowerShell guide](../guide.md) for authentication, cache preparation and result handling.

**Persistent cache:**
- Run `uv run python -m demo.kaggle_run --cpu src/demo/prepare_data.py` to prepare the pinned model and dataset sources from `asset_plan.toml` on Kaggle. Later submissions attach the private caches automatically.
- CLI pushes do not carry Add-ons secret attachments. Use a browser-started initialization with those secrets enabled, or attach the separately approved private credential input using `--dataset dungg1810/specadapt-run-credentials`. Llama-3.1 additionally needs approved `HF_TOKEN` or `HUGGING_FACE_TOKEN` access; until then it is pending.
- Run the same preparation command again to validate cached files and log model paths and dataset row counts. Use `--resume <timestamp>` after a laptop disconnect.
