#import "../uet-thesis.typ": *

= Reproducibility <app:repro>

Every result in this thesis comes from one of the runs in @tab:runs. Each run records the code version it ran, so any number can be traced back to the exact code, data and models that produced it. The code, the experiment definitions and the scripts that draw every figure are in the project repository @specadapt-repo.

#figure(
  {
    set text(size: 10.5pt)
    table(columns: (auto, 1fr, auto), align: left,
      table.header[*Run*][*What it produced*][*Code version*],
      [20261123T141520817523Z], [Live check of P1, P2 and P4 on the smoke prompts], [8c3dea7],
      [20261123T154925641802Z], [Recorded answers of both targets: test, training and validation prompts], [640c25f],
      [20261124T014249246739Z], [Untrained replay on a 40-prompt validation sample, all three pairs], [640c25f],
      [20261124T050647634040Z], [Small trial run of the P2 training pipeline], [7004468],
      [20261124T052050221490Z], [Main P2 experiment: untrained drafter, LoRA and output adapter], [f5f1767#super[a]],
      [20261124T063355273658Z], [Automated test suite, all 87 tests passed], [1812446],
      [20261124T063633692969Z], [Bridge combined with the LoRA drafter, four combinations], [1812446],
      [20261013T070257913336Z], [Early demonstration of the bridge, 12 prompts], [—],
    )
  },
  caption: [Run register. Runs are named by their start time (UTC). #super[a] The only uncommitted change in this run was to a notes file; the training code was unchanged.],
) <tab:runs>

= Full sweep and per-task tables <app:tables>

@tab:app-sweep lists every candidate setting tried for pair P2, with the number of the target's 62,972 validation answer tokens that the drafter ranked first.

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right),
    table.header[*Tier*][*Rank*][*Learning rate*][*Epoch*][*Matched tokens*][*Top-1 agreement*],
    [T2], [16], [$10^(-4)$], [2], [31,792], [0.5049],
    [T2], [16], [$10^(-4)$], [4], [32,655], [0.5186],
    [T2], [16], [$3 times 10^(-4)$], [2], [34,655], [0.5503],
    [T2], [16], [$3 times 10^(-4)$], [4], [35,504], [0.5638],
    [T2], [16], [$10^(-3)$], [2], [37,086], [0.5889],
    [T2], [16], [$10^(-3)$], [4], [*37,753*], [*0.5995*],
    [T1], [8], [$10^(-3)$], [2], [28,097], [0.4462],
    [T1], [8], [$3 times 10^(-3)$], [2], [29,567], [0.4695],
    [T1], [8], [$10^(-2)$], [2], [30,691], [0.4874],
    [T1], [16], [$10^(-3)$], [2], [28,624], [0.4546],
    [T1], [16], [$3 times 10^(-3)$], [2], [30,239], [0.4802],
    [T1], [16], [$10^(-2)$], [2], [*31,456*], [*0.4995*],
  ),
  caption: [All scored configurations, P2, 320 validation prompts. Selected configurations in bold.],
) <tab:app-sweep>

@tab:app-m1-task breaks the four combinations of @tab:m1-arms down by task.

#figure(
  {
    set text(size: 10.5pt)
    table(columns: (auto, auto, 1.2fr, auto, auto, auto, auto), align: (left, left, right, right, right, right, right),
      table.header[*Task*][*Arm*][*Accepted / proposed*][*Accept.*][*Tokens per target call*][*Drafter steps*][*Cache rounds*],
      [math], [B1], [1,899 / 8,338], [22.78%], [1.636], [8,921], [0],
      [], [B1 + M1], [2,116 / 7,920], [26.72%], [1.764], [5,756], [841],
      [], [T2], [2,575 / 6,333], [40.66%], [2.115], [6,882], [0],
      [], [T2 + M1], [2,562 / 6,562], [39.04%], [2.103], [4,606], [774],
      [code], [B1], [1,553 / 9,310], [16.68%], [1.454], [10,175], [0],
      [], [B1 + M1], [1,624 / 9,250], [17.56%], [1.485], [8,396], [523],
      [], [T2], [2,341 / 7,010], [33.40%], [1.894], [7,814], [0],
      [], [T2 + M1], [2,342 / 7,121], [32.89%], [1.895], [6,381], [476],
      [instruction], [B1], [1,313 / 7,704], [17.04%], [1.462], [8,494], [0],
      [], [B1 + M1], [1,343 / 7,706], [17.43%], [1.477], [7,486], [307],
      [], [T2], [1,489 / 6,935], [21.47%], [1.557], [7,940], [0],
      [], [T2 + M1], [1,486 / 7,043], [21.10%], [1.556], [7,066], [295],
      [summarization], [B1], [327 / 1,834], [17.83%], [1.467], [2,096], [0],
      [], [B1 + M1], [338 / 1,853], [18.24%], [1.491], [1,759], [102],
      [], [T2], [388 / 1,651], [23.50%], [1.607], [1,878], [0],
      [], [T2 + M1], [386 / 1,685], [22.91%], [1.602], [1,609], [91],
    )
  },
  caption: [Four M1 arms per task, P2, 20 validation prompts per task, oracle replay, $K = 3$, no seed cache.],
) <tab:app-m1-task>

= Environment and pinned revisions <app:env>

@tab:app-models and @tab:app-data list the exact versions of the models and datasets, as Hugging Face repository names and revision hashes. @tab:app-packages gives the software environment of the main P2 experiment.

#figure(
  {
    set text(size: 10pt)
    show raw: set text(size: 8.5pt)
    table(columns: (1fr, auto), align: left,
      table.header[*Model*][*Revision*],
      [`JackFram/llama-68m`], [`9de84537b6aa98de634ad0fbb1608e9d6a019355`],
      [`JackFram/llama-160m`], [`aca9b687d1425f863dcf5de9a4c96e3fe36266dd`],
      [`lmsys/vicuna-7b-v1.5`], [`3321f76e3f527bd14065daf69dad9344000a201d`],
      [`Qwen/Qwen2.5-7B-Instruct`], [`a09a35458c702b33eeacc393d103063234e8bc28`],
      [`meta-llama/Llama-3.1-8B-Instruct` (planned)], [`0e9e39f249a16976918f6564b8830bc894c89659`],
    )
  },
  caption: [Pinned model revisions.],
) <tab:app-models>

#figure(
  {
    set text(size: 10pt)
    show raw: set text(size: 8.5pt)
    table(columns: (1fr, auto), align: left,
      table.header[*Dataset*][*Revision*],
      [`openai/gsm8k`], [`a05f38c23a0e9ab0b71de8a2b4947e20f74f68f7`],
      [`openai/openai_humaneval`], [`c4dedfa884e8add4c679789be33fe7147d318ca2`],
      [`google-research-datasets/mbpp`], [`f7f18be5a72d664e3884261303eb5b7f01dd10c2`],
      [`tatsu-lab/alpaca`], [`3c2d40e7db7d37f75b9cfd5b2685762a1c3c9ae8`],
      [`EdinburghNLP/xsum`], [`03339c27cd655c3369346bacb533293f222a9d42`],
    )
  },
  caption: [Pinned dataset revisions (Parquet exports).],
) <tab:app-data>

#figure(
  table(columns: (auto, 1fr), align: left,
    table.header[*Item*][*Value*],
    [Replay and training GPU], [Tesla T4 (Kaggle), pinned Docker image digest],
    [Oracle recording GPU], [NVIDIA A10 (test), NVIDIA A10G (fit and validation), Modal],
    [`torch` / CUDA], [2.10.0+cu128 / 12.8],
    [`transformers`], [4.57.1],
    [`accelerate`], [1.11.0],
    [`sentencepiece`], [0.2.1],
    [`safetensors`], [0.7.0],
    [`tokenizers`], [0.22.2],
    [`huggingface-hub`], [0.36.2],
    [`bitsandbytes`], [0.48.1],
    [Precision], [FP32 training, FP16 replay, SDPA attention, no autocast],
  ),
  caption: [Environment of the P2 full run.],
) <tab:app-packages>

= Prompt formats and data splits <app:prompts>

Each dataset row is turned into a prompt in a fixed way (@tab:app-formats), and the prompt sets of @tab:splits are drawn from the datasets as shown in @tab:app-slices.

#figure(
  table(columns: (auto, 1fr), align: left,
    table.header[*Source*][*How the prompt is formed*],
    [GSM8K], [the question],
    [HumanEval], [the given code stub with its docstring],
    [MBPP], [the task description, a blank line, "Example test:" and the first test case],
    [Alpaca], [the instruction; if the row has an input, a blank line, "Input:" and the input],
    [XSum], [the line "Summarize the following article in one sentence.", a blank line and the article],
    [Drafter's view], [the prompt, a blank line, and the text committed so far],
  ),
  caption: [Prompt formats. The target additionally wraps the prompt in its own chat template.],
) <tab:app-formats>

#figure(
  {
    set text(size: 10.5pt)
    table(columns: (auto, 1fr, 1fr, 1fr, 1fr), align: (left, left, left, left, left),
      table.header[*Set*][*Maths*][*Code*][*Instruction*][*Summarization*],
      [Test, core], [50 from GSM8K test], [25 from HumanEval, 25 from MBPP test], [50 from Alpaca], [50 from XSum test],
      [Test, expansion], [50 from GSM8K test], [25 from HumanEval, 25 from MBPP test], [50 from Alpaca], [50 from XSum test],
      [Smoke], [10 from GSM8K train], [10 from MBPP prompt set], [10 from Alpaca], [10 from XSum validation],
      [Training], [360 from GSM8K train], [360 from MBPP train], [360 from Alpaca], [360 from XSum train],
      [Validation], [80 from GSM8K train], [80 from MBPP validation], [80 from Alpaca], [80 from XSum train],
      [Development], [12 from GSM8K test], [—], [—], [—],
    )
  },
  caption: [Where each prompt set comes from. Within a source, the sets take disjoint slices of the deduplicated, shuffled rows; Alpaca has only a training split, so all its sets are disjoint slices of it.],
) <tab:app-slices>
