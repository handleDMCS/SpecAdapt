**Overview**
tl;dr: You build and test the tool on your own laptop, push finished code to GitHub where it's auto-checked, then send one verified version to a Kaggle GPU machine to run the real experiment and bring the results back home.

**Laptop — Build & Test**
**tl;dr**: You write the code and run quick, free checks on your own computer before sending anything anywhere.

**Why**: This is where all the logic gets built and debugged cheaply, before any paid GPU time is spent.

**Input**:
```
- The plan for what the tool should do
- Small free stand-in models for testing
- A settings file (config)
```

**Output**:
```
- Working code
- A record that tests passed
- A tiny results file from a quick test run
```

**Technical keywords**: uv, pytest, CPU-only torch, pyproject.toml

**GitHub — Save & Verify**
**tl;dr**: You send your finished code to a shared online storage spot, which automatically double-checks it still works.

**Why**: This is the one trusted, official copy of the code — only a version that passes here is allowed to move on to the GPU stage.

**Input**:
```
- Code sent over from the laptop
- The automatic checking scripts
```

**Output**:
```
- A saved, timestamped snapshot of the code
- A pass or fail check mark
```

**Technical keywords**: GitHub Actions, CI, commit SHA, git push

**Kaggle — Run on GPU**
**tl;dr**: A single command tells a powerful cloud computer to grab one exact saved version of your code and run the real, heavier experiment on it.

**Why**: This is where the slow, expensive computing actually happens — and only ever on code that's already been saved and verified.

**Input**:
```
- The exact saved code version
- The settings file for this run
- A secret access key (only for restricted models)
```

**Output**:
```
- Raw results from the experiment
- A record of exactly which code and settings made them
```

**Technical keywords**: Kaggle API, kernel-metadata.json, HF_TOKEN, T4 GPU

**Fetch — Bring Results Home**
**tl;dr**: You pull the finished results file down from the cloud machine back onto your own laptop.

**Why**: Closes the loop so you can look at, compare, or write about your findings right from your own computer.

**Input**:
```
- A completed cloud run
```

**Output**:
```
- A local results file, labeled by time and traceable to its code version
```

**Technical keywords**: kaggle kernels output