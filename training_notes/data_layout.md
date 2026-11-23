# Hybrid dataset layout

The builder merges the source slices for each role into one JSONL file. A hybrid file contains prompts from multiple source datasets. The dev file contains the fixed GSM8K demo rows.

```text
dataset/
├── splits.toml
├── dev.jsonl
├── smoke.jsonl
├── train/
│   ├── fit/
│   │   ├── base.jsonl
│   │   └── extra.jsonl
│   └── val.jsonl
└── test/
    ├── core.jsonl
    └── expansion.jsonl
```

Each JSONL row carries `source_dataset` (`gsm8k`, `humaneval`, `mbpp`, `alpaca`, or `xsum`), `source_split`, and `row_id` to identify the original dataset row. It also carries `task`, `prompt_id`, `prompt`, and `prompt_sha256`. The manifest lists the source revisions and records row origins and hashes in JSONL order for each file.

Slice bounds are zero-based and end-exclusive. The builder concatenates the slices in the order shown, with the listed code sources in order for each test file.

| Output | Math | Code | Instruction | Summarization |
| --- | --- | --- | --- | --- |
| `dev.jsonl` | `gsm8k_test` IDs from `pair2_demo.toml` | | | |
| `smoke.jsonl` | `gsm8k_train[0:10]` | `mbpp_prompt[0:10]` | `alpaca_train[100:110]` | `xsum_validation[0:10]` |
| `train/fit/base.jsonl` | `gsm8k_train[10:370]` | `mbpp_train[0:360]` | `alpaca_train[110:470]` | `xsum_train[0:360]` |
| `train/fit/extra.jsonl` | `gsm8k_train[450:1530]` | `mbpp_train[360:]` | `alpaca_train[550:1630]` | `xsum_train[440:1520]` |
| `train/val.jsonl` | `gsm8k_train[370:450]` | `mbpp_validation[0:80]` | `alpaca_train[470:550]` | `xsum_train[360:440]` |
| `test/core.jsonl` | `gsm8k_test[0:50]` | `humaneval_test[0:25]` + `mbpp_test[0:25]` | `alpaca_train[0:50]` | `xsum_test[0:50]` |
| `test/expansion.jsonl` | `gsm8k_test[50:100]` | `humaneval_test[25:50]` + `mbpp_test[25:50]` | `alpaca_train[50:100]` | `xsum_test[50:100]` |
