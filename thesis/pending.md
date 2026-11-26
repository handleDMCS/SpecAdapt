# Thesis: what is pending

Everything left out of `build/thesis.pdf` because the evidence or the source does not exist yet. The PDF itself carries no gap markers. Section numbers follow `skeleton.md`. Update this file whenever the PDF is rebuilt.

Snapshot: 2026-11-24, PDF built from repo `1812446`.

## Front matter

All fields are filled; no `[TO FILL]` marker is left in the PDF.

- From `personal_info/info.md`: title, author (Trương Tuấn Dũng), supervisor, logo (`assets/logo.png`).
- Supervisor printed as "Dr. Nguyễn Văn Vinh": the UET NLP Lab people page lists him as "TS. Nguyễn Văn Vinh".
- Major "Information Technology": UET K66 class list puts student 21020058 in class K66C-CLC, "Ngành: Công nghệ thông tin".
- Year "Ha Noi - 2026": defense expected around December 2026 (author).
- Acknowledgement: drafted from the author's brief (intern at Dr. Vinh's UET NLP Lab, learned speculative decoding there). Author to read and adjust.

To confirm:

- **Degree name** in the approval text is "Bachelor of Information Technology", inferred from the major. Check the exact English degree name for the K66 C-CLC programme.
- Whether the cover should name the programme as "Information Technology" or with its high-quality-programme label.
- University block "Vietnam National University, Hanoi / University of Engineering and Technology" (matches the logo).

## Literature (needs sources; nothing is cited yet except the GitHub repo)

No reference has a checked BibTeX entry. Each needs to be found, read and added to `src/refs.bib` before it can be cited.

| Skeleton section | Left out of the PDF | Needs |
| --- | --- | --- |
| 1.1 Motivation | Speed-ups reported for SpD; why heterogeneous pairs matter in deployment | SpD papers |
| 2.1 Autoregressive decoding and its cost | Whole section | transformer decoding, KV cache sources |
| 2.2 Speculative decoding | Original formulation, stochastic verification, standard speed-up model (the greedy mechanism is written, from the code) | original SpD papers |
| 2.3 Heterogeneous pairs | Prior cross-vocabulary SpD work (the mechanism is written, from the code) | cross-vocabulary SpD papers |
| 2.4 OmniDraft | Whole section | Ramakrishnan et al., arXiv:2507.02659. No numerical comparison: setups differ |
| 2.5 Parameter-efficient adaptation and distillation | Whole section | LoRA, low-rank heads, knowledge distillation |
| 2.6 Adaptive draft length | Whole section | adaptive-K papers |
| 6.4 Relation to prior work | Whole section | Chapter 2 first |
| Models and datasets | No citations for Llama-68M/160M, Vicuna, Qwen2.5, GSM8K, HumanEval, MBPP, Alpaca, XSum | papers or model cards |

## Results not yet run

| Skeleton section | Left out of the PDF | Plan / evidence needed |
| --- | --- | --- |
| 3.2.5 M3 adaptive draft length | Whole method section (mentioned only as future work) | `src/specadapt/adaptive_k.py`; `training_notes/iteration.md` block 5 |
| 3.2.6 Run-time cost | Measured bridge and cache overhead in the current setup vs the 10% budget | stage 5 live sweep |
| 4.2 Untrained baseline | B1 for P1 and P4 on all 320 validation prompts | comes with P1/P4 training runs |
| 4.3 Logit alignment | Figure: training loss per configuration | heartbeat log in `results/submission-20261124T052050221490Z.json` → `run_log`; plot with a script in `thesis/figures/` |
| 4.4 Other pairs (P1, P4) | Whole section | `training/jobs/p1.toml`, `training/jobs/p4.toml`; `training_notes/iteration.md` block 6 |
| 4.5 Held-out test results | Whole section | freeze checkpoints in `checkpoints_frozen.toml`, replay 400 test prompts per pair |
| 4.6 Ablations | Whole section | learning curve, leave-one-task-out, frozen-cache size, T0 without caches |
| 5.3 Adaptive draft length results | Whole section | depends on 3.2.5 |
| 5.4 Live speed-up and adapter overhead | Whole section | stage 5 live sweep on `test/core` |
| 3.3.5 / 6.5 Confidence intervals | No CI for the 24 Nov results | paired bootstrap over prompt IDs; per-prompt counts exist for the 80-prompt M1 run (`results/runs/20261124T063633692969Z/artifacts/*.json`); the P2 full run saved totals only |

## Evidence files and open checks

- `results/submission-20261123T144002663579Z.json` (replay-vs-live acceptance deltas, P1 0.00112, P2 0.00183, P4 0.00053) is **not on disk**. §4.1 states the agreement as "recorded in the project log". Re-collect with `uv run python -m demo.kaggle_run --resume 20261123T144002663579Z` to cite the file.
- The exact submission command of the M1 × T2 run `20261124T063633692969Z` is not recorded in the notes.
- Split-overlap test gate `test_splits.py` is not committed (W5).
- W2 (P1 proposals resolve as `direct`) is implemented but unchecked.
- Live B1 output differed from target-only output on 1 to 6 of 40 smoke prompts per pair; the stage 5 gate asks for 99%.
- Skeleton says "Modal A10" for oracle recording; the manifests say A10 for test prompts and **A10G** for training/validation prompts. The PDF follows the manifests. Update `skeleton.md` §3.3.3.
- `results/` and `build/` are git-ignored: back them up.

## Length

Body before the references is below the 50 to 75 page rule. The gap is mostly Chapter 2 and the unrun results above. Do not pad.

## Format points to confirm with the supervisor

From `.claude/skills/thesis-pdf/format-rules.md` "Open points": references before appendices (EN) vs after (VN); IEEE by first citation; no thesis-type line on the cover; Times New Roman 13 pt; `Figure 2.1 –` label form; Roman page numbers counted from the cover; degree name in the approval text; abstract length (about 360 words).
