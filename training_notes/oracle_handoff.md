# Oracle storage and Kaggle replay

The oracle is a saved record of what the target model generated for each prompt. Modal runs the target once per prompt and target revision. Kaggle can then use that record to score guesses from different draft models and adapter versions.

## What gets stored

The current P2 demo uses a `target_references.toml` file with 12 Qwen references. Each reference has a prompt ID and hash, the exact input and output token IDs, the generation limit, and the stop reason. The file also carries the target revision, tokenizer fingerprint, and recording settings. [Current oracle format](../src/demo/oracle.py)

For the training pipeline, Modal will write one JSONL row per prompt for each target and split, plus a manifest. Each row includes both the decoded answer text and the exact target token IDs. The text is for adapter training. The IDs are for replay. The recorder appends rows as it finishes them and uses `prompt_id` to skip completed rows on restart. [Recording contract](training.md#4-recording-contract)

## Modal run order

1. **Smoke benchmark.** Run the target-only baseline and plain speculative decoding on 40 smoke prompts at batch size 1, plus a 32-prompt batched throughput probe.
2. **Test oracle recording.** Record 400 held-out prompts per target at batch size 1, with 200 core and 200 expansion prompts. Kaggle uses these recordings to measure acceptance after the checkpoints are frozen.
3. **Teacher-answer recording.** Record up to 5,008 training and validation prompts per target with the current dataset. If batched throughput is below 250 tokens per second, record the 1,760 fit base and validation prompts.

[Training plan](training.md#5-modal-workflow-and-budget)

## How Kaggle will reuse it

The proposed handoff is to publish the completed recordings and manifest as a private, versioned Kaggle Dataset and attach it as a read-only input using the runner’s existing `--dataset` option. Kaggle joins the recordings to the hybrid prompts by `prompt_id` and verifies the prompt hashes, target revision, and tokenizer fingerprint. The loader needs an update because the current P2 contract includes the recording GPU. The new contract will track the Modal A10 recording GPU and the Kaggle T4 replay GPU independently. [Kaggle runner](../src/demo/kaggle_run.py) · [Current oracle loader](../src/demo/oracle.py)

- **Adapter training.** Kaggle tokenizes the saved answer text with the draft tokenizer and trains the draft to predict the answer. [Training plan](training.md#6-training-and-selection-kaggle-t4)
- **Replay.** The draft proposes tokens, the vocabulary bridge maps them to target token IDs, and the oracle checks them against the saved sequence. If the target recorded `[41, 57, 92]` and the draft proposes `[41, 99]`, replay accepts `41` and advances with the recorded `57`. This provides acceptance labels for training the stopping rule and acceptance metrics for evaluation. [Replay code](../src/specadapt/decoding.py)

Kaggle replays the 400 test prompts after the checkpoints are frozen. The live A10 run on 200 core prompts supplies the actual speed measurements. [Evaluation plan](training.md#7-evaluation)
