**Overview**
tl;dr: You build and test the tool on your laptop, commit it, then a single Kaggle command runs that exact commit on a GPU and you pull the results back.

**Laptop: Build & Test**
**tl;dr**: You write the code and run quick, free checks on your own computer.

**Why**: All logic gets built and debugged here, before any GPU time is spent.

**Input**:
```
- The plan for what the tool should do
- Small free stand-in models for testing
- A settings file (config)
```

**Output**:
```
- Working code
- A passing local test run
- A tiny results file from a quick test run
```

**Technical keywords**: uv, pytest, CPU-only torch, pyproject.toml

**Commit & Push: Pin the Version**
**tl;dr**: You save your code as one labeled snapshot and upload it to GitHub.

**Why**: Gives the GPU machine one exact, fetchable version of the code, so every result traces back to it.

**Input**:
```
- Code that passed your local tests
```

**Output**:
```
- A timestamped snapshot with a unique ID
- A copy of that snapshot online
```

**Technical keywords**: git commit, git push, commit SHA

**Kaggle: Run on GPU**
**tl;dr**: One command tells a cloud GPU machine to download your snapshot and run the real experiment.

**Why**: The slow, heavy computing happens here, on a snapshot you already tested locally.

**Input**:
```
- The snapshot ID
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

**Hackathon commands** (copy-paste loop):
```bash
uv run pytest -q                                # 1. test locally
git add -A && git commit -m "run" && git push   # 2. pin version
kaggle kernels push -p .                        # 3. run on GPU
kaggle kernels status <user>/<slug>             # 4. poll until "complete"
kaggle kernels output <user>/<slug> -p results/$(git rev-parse --short HEAD)   # 5. fetch
```

**Two things to watch:**
- Have the Kaggle kernel script run `git clone <repo> && git checkout <SHA>`. Without that, it won't run the pinned commit.
- Pass the SHA in via an env var or by editing the script before `kaggle kernels push`.