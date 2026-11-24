"""Does the trained T2 drafter combine with M1? Replay B1 and T2 in baseline and m1 mode on a small subset.

Four arms on the same validation prompts, oracle-verified, K = 3, no training:
B1-baseline, B1-m1, T2-baseline, T2-m1. The T2 weights come from a finished pair run, attached
with --kernel-source and checked against the SHA-256 in the job TOML. M1 runs without the
training-derived seed cache (it is built from GSM8K train answers, the same source as the math
validation prompts), so only each prompt's own continuation cache is used.
"""

from pathlib import Path
import json

from demo import kaggle_runtime as runtime
from demo.assets import load_plan
from demo.benchmark import setup
from demo.oracle import tokenizer_digest
from specadapt.bridge import decode
from specadapt.decoding import generate, target_prompt
from specadapt.training_inputs import MANIFEST_NAME, MAX_NEW_TOKENS, TASKS, load_target, row_oracle, sha256_file
from training.kaggle.train_pair import (
    PAIRS, Progress, attach, checked_merge, deadline, submission_provenance, write_json,
)

ARMS = (("B1", "baseline"), ("B1", "m1"), ("T2", "baseline"), ("T2", "m1"))
COUNTS = ("accepted", "proposed", "rounds", "generated_tokens", "draft_steps", "cache_rounds")


def select_subset(validation, per_task):
    """The first per_task prompts of each task, in file order (the file is sorted by task)."""
    chosen, seen = [], {task: 0 for task in TASKS}
    for row in validation:
        if seen[row["task"]] < per_task:
            chosen.append(row)
            seen[row["task"]] += 1
    if any(count != per_task for count in seen.values()):
        raise ValueError(f"Validation has fewer than {per_task} prompts for some task: {seen}")
    return chosen


def rates(counts, cost_ratio):
    """Acceptance, tokens per target call and modeled cost per token from summed counts."""
    accepted, proposed, rounds, generated, draft_steps, cache_rounds = (counts[key] for key in COUNTS)
    return dict(acceptance=accepted / proposed if proposed else 0.0,
                tokens_per_round=generated / rounds if rounds else 0.0,
                modeled_cost_per_token=(rounds + cost_ratio * draft_steps) / generated if generated else 0.0,
                cache_round_share=cache_rounds / rounds if rounds else 0.0)


def arm_summary(per_prompt, cost_ratio):
    total = {key: sum(data[key] for data in per_prompt) for key in COUNTS}
    by_task = {}
    for task in TASKS:
        rows = [data for data in per_prompt if data["task"] == task]
        counts = {key: sum(data[key] for data in rows) for key in COUNTS}
        by_task[task] = dict(**counts, **rates(counts, cost_ratio))
    return dict(**total, **rates(total, cost_ratio), per_task=by_task, prompts=len(per_prompt))


def find_weights(expected_sha256):
    candidates = [path for path in Path("/kaggle/input").rglob("t2.safetensors")]
    matching = [path for path in candidates if sha256_file(path) == expected_sha256]
    if not matching:
        runtime.abort(f"No attached t2.safetensors has SHA-256 {expected_sha256}; found {len(candidates)} file(s).")
    return matching[0]


def load_models(job, summary, progress):
    progress.update("preflight", action="check_inputs")
    draft_key, target_key = PAIRS[job["pair"]]
    manifests = list(Path("/kaggle/input").rglob(MANIFEST_NAME))
    if len(manifests) != 1:
        raise ValueError("Attach exactly one private specadapt-train-inputs dataset.")
    record, _, validation = load_target(manifests[0].parent, target_key)
    rows = select_subset(validation, job["prompts_per_task"])
    weights = find_weights(job["t2_sha256"])
    plan = {entry["key"]: entry for entry in load_plan(Path("asset_plan.toml"))["models"]}
    draft_spec, target_spec = plan[draft_key], plan[target_key]
    if record["target"] != target_spec:
        raise ValueError("The recorded target differs from the revision in asset_plan.toml.")
    progress.update("preflight", action="install_dependencies")
    setup()
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specadapt.alignment import adapter_parameters, encode_example

    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        runtime.abort(f"Expected a Kaggle T4 on device 0, got {gpu}.")
    torch.cuda.set_device(0)
    summary["provenance"].update(replay_gpu=gpu, torch=torch.__version__, draft_revision=draft_spec["revision"],
                                 target_revision=target_spec["revision"], t2_sha256=job["t2_sha256"],
                                 inputs_sha256=sha256_file(manifests[0]))
    progress.update("preflight", action="load_tokenizers")
    tokenizer_files = [name for name in target_spec["files"] if name.startswith("tokenizer") or name in
                       {"config.json", "merges.txt", "vocab.json", "special_tokens_map.json"}]
    target_dir = snapshot_download(target_spec["repo"], revision=target_spec["revision"],
                                   allow_patterns=tokenizer_files)
    target_tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
    if tokenizer_digest(target_tokenizer) != record["tokenizer_sha256"]:
        runtime.abort("Target tokenizer fingerprint differs from the recorded oracle.")
    draft_dir = snapshot_download(draft_spec["repo"], revision=draft_spec["revision"],
                                  allow_patterns=draft_spec["files"])
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_dir, local_files_only=True)
    for row in rows:
        ids = target_tokenizer.encode(target_prompt(target_tokenizer, row["prompt"]), add_special_tokens=False)
        if ids != row["input_ids"] or decode(target_tokenizer, row["token_ids"]) != row["text"]:
            runtime.abort(f"Recorded target input or text differs for {row['prompt_id']}.")

    def load_draft(precision):
        return AutoModelForCausalLM.from_pretrained(draft_dir, local_files_only=True, dtype=precision,
                                                   device_map={"": 0}, attn_implementation="sdpa").eval()

    progress.update("preflight", action="merge_t2")
    with safe_open(str(weights), framework="pt") as stream:
        config = json.loads(stream.metadata()["config"])
    trained = load_draft(torch.float32)
    adapter = attach(trained, "T2", config)
    state = load_file(weights)
    if state.keys() != adapter_parameters(trained).keys():
        raise ValueError("Unexpected parameter names in the T2 adapter checkpoint.")
    trained.load_state_dict(state, strict=False)
    max_length = json.loads((Path(draft_dir) / "config.json").read_text())["max_position_embeddings"]
    example = encode_example(draft_tokenizer, rows[0], max_length)
    summary["t2"] = dict(config=config, merge_error=checked_merge(trained, adapter, "T2", example))
    models = {"B1": load_draft(torch.float16), "T2": trained.to(dtype=torch.float16)}
    oracles = [row_oracle(row, record["eos_token_ids"], MAX_NEW_TOKENS) for row in rows]
    return models, draft_tokenizer, target_tokenizer, list(zip(rows, oracles))


def replay_arm(model, draft_tokenizer, target_tokenizer, prompts, job, name, mode, progress):
    per_prompt = []
    progress.update(f"{name}-replay", 0, len(prompts))
    for step, (row, oracle) in enumerate(prompts, 1):
        data = generate(None, model, target_tokenizer, draft_tokenizer, row["prompt"], mode=mode, k=job["k"],
                        same_tokenizer=job["pair"] == "P1", max_new_tokens=MAX_NEW_TOKENS, oracle=oracle)
        if tuple(data["token_ids"]) != oracle.token_ids:
            runtime.abort(f"{name} replay departed from the recorded trajectory for {row['prompt_id']}.")
        per_prompt.append(dict(task=row["task"], prompt_id=row["prompt_id"],
                               **{key: data[key] for key in COUNTS}, replay_seconds=data["replay_seconds"]))
        accepted = sum(item["accepted"] for item in per_prompt)
        proposed = sum(item["proposed"] for item in per_prompt)
        progress.update(f"{name}-replay", step, len(prompts), acceptance=accepted / proposed if proposed else 0.0)
    return per_prompt


def run(job, summary, progress):
    with deadline(job["preflight_minutes"] * 60, "Preflight exceeded its budget."):
        models, draft_tokenizer, target_tokenizer, prompts = load_models(job, summary, progress)
    summary["prompts"] = len(prompts)
    write_json(Path("summary.json"), summary)
    for tier, mode in ARMS:
        name = f"{tier}-{mode}"
        per_prompt = replay_arm(models[tier], draft_tokenizer, target_tokenizer, prompts, job, name, mode, progress)
        summary["arms"][name] = arm_summary(per_prompt, job["cost_ratio"])
        Path("artifacts").mkdir(exist_ok=True)
        write_json(Path("artifacts") / f"{name}.json", per_prompt)
        arm = summary["arms"][name]
        progress.once("result", arm=name, acceptance=arm["acceptance"], tokens_per_round=arm["tokens_per_round"],
                      cost=arm["modeled_cost_per_token"], cache_share=arm["cache_round_share"])
        write_json(Path("summary.json"), summary)
    base = summary["arms"]["B1-baseline"]
    summary["vs_b1_baseline"] = {
        name: dict(acceptance_pp=100 * (arm["acceptance"] - base["acceptance"]),
                   tokens_per_round=arm["tokens_per_round"] / base["tokens_per_round"],
                   modeled_cost=arm["modeled_cost_per_token"] / base["modeled_cost_per_token"])
        for name, arm in summary["arms"].items()}


def main():
    job = runtime.job_config()
    summary = dict(verdict="running", pair=job["pair"], k=job["k"], max_new_tokens=MAX_NEW_TOKENS,
                   seed_cache=False, provenance=submission_provenance(), job=job, arms={})
    try:
        with deadline(job["max_minutes"] * 60, "Job exceeded max_minutes."), Progress() as progress:
            run(job, summary, progress)
    except Exception as error:
        summary.update(verdict="fail", error=f"{type(error).__name__}: {error}")
        write_json(Path("summary.json"), summary)
        raise
    summary["verdict"] = "pass"
    write_json(Path("summary.json"), summary)


if __name__ == "__main__":
    main()
