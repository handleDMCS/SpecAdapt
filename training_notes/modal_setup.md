# Modal setup for SpecAdapt

**Modal runs Phase 1 and the live-speed portion of Phase 5.** [Five-phase training note](training_phases.md)

| Phase | Where it runs | What happens |
| --- | --- | --- |
| **1. Collect teacher answers** | **Modal A10** | The target model answers about 5,000 training prompts. Save the text and exact tokens for draft training. |
| **2–4. Train and tune** | **Kaggle T4** | Train the two draft adapters and the stopping rule, then tune on validation. |
| **5. Freeze, then test** | **Kaggle T4 → Modal A10** | After freezing the chosen models, measure acceptance by replaying all 400 test prompts on Kaggle. Then measure live speed on the 200 core prompts on Modal. |

The detailed runbook also uses Modal for asset sync on CPU, a smoke run, and recording the target outputs that Kaggle replays during Phase 5. [Detailed runbook](training.md#5-modal-workflow-and-budget)

## Manual Modal setup

1. **Authenticate and create the Volume.** Run these commands from the SpecAdapt repository root. Use the `main` environment for both the Volume and the secret. The harness will mount `specadapt` at `/vol/specadapt`. [Modal setup](https://modal.com/docs/guide/getting-started) · [Volume CLI](https://modal.com/docs/cli/latest/volume)

   ```powershell
   uvx modal setup
   uvx modal token info
   uvx modal volume create --env main --version 2 specadapt
   uvx modal volume put --env main specadapt .\training\dataset /dataset
   uvx modal volume ls --env main specadapt /dataset
   ```

2. **Add the Hugging Face secret.** In Modal’s **Secrets** panel for `main`, create `huggingface` with a key named `HF_TOKEN` and a Hugging Face read token as its value. For Wave 2, the token’s account needs approved access to Llama 3.1 8B Instruct. [Modal secrets](https://modal.com/docs/guide/secrets) · [Llama access](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. **Set the usage budget.** If the workspace is dedicated to SpecAdapt, set its **usage budget** to **$25** in **Usage & Billing**. This caps usage before credits. [Modal budgets](https://modal.com/docs/guide/budgets)

`modal_app.py` is still pending. Once setup is ready, wire the harness to these resources and verify it with a **Modal CPU dry run** before launching a GPU phase. [Harness plan](training.md#2-work-items-in-order)
