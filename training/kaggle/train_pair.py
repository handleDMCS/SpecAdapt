"""Fit a drafter to recorded teacher text and compare validation acceptance on Kaggle T4."""

from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version
import json
from pathlib import Path
from random import Random
import signal
import sys
from threading import Event, Lock, Thread
from time import monotonic

from demo import kaggle_runtime as runtime
from demo.assets import load_plan
from demo.benchmark import setup
from demo.oracle import tokenizer_digest
from specadapt.bridge import decode
from specadapt.decoding import generate, target_prompt
from specadapt.training_inputs import (
    MANIFEST_NAME, MAX_NEW_TOKENS, RECORDING_RUN, TASKS, load_target, row_oracle, sha256_file,
)


PAIRS = {"P1": ("llama_68m", "vicuna_7b"), "P2": ("llama_68m", "qwen_7b"),
         "P4": ("llama_160m", "qwen_7b")}
K = 3
# Default tier order when the job TOML has no tier_order.
TIERS = ("T1", "T2")
MERGE_TOLERANCE = 1e-3
SMOKE_COUNTS = (2610, 13938)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def submission_provenance():
    # Iteration 1 gives the SHA to run_submission, but does not export it to the script environment.
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is runtime.run_submission.__code__:
            submission = frame.f_locals["submission"]
            return {key: submission[key] for key in ("git_sha", "git_dirty", "timestamp")}
        frame = frame.f_back
    raise RuntimeError("Submit train_pair.py through demo.kaggle_run to record the bundled Git SHA.")


class Progress:
    """Keep reporting the last step while a download or model call is busy."""

    def __init__(self):
        self.stop = Event()
        self.lock = Lock()
        self.state = None
        self.last = 0.0
        self.worker = Thread(target=self._keepalive, daemon=True)

    def _keepalive(self):
        while not self.stop.wait(15):
            with self.lock:
                if self.state is not None and monotonic() - self.last >= 30:
                    try:
                        self._emit()
                    except runtime.JobAborted:
                        return

    def _emit(self):
        phase, step, total, metrics = self.state
        runtime.heartbeat(phase, step, total, **metrics)
        self.last = monotonic()

    def update(self, phase, step=None, total=None, **metrics):
        with self.lock:
            starts = self.state is None or self.state[0] != phase or step == 0
            self.state = phase, step, total, metrics
            if starts or (total is not None and step == total) or monotonic() - self.last >= 30:
                self._emit()

    def once(self, phase, **metrics):
        """One line that the keepalive does not repeat, such as a tier's result."""
        with self.lock:
            line = runtime.heartbeat(phase, **metrics)
            self.last = monotonic()
            return line

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *error):
        self.stop.set()
        self.worker.join()


@contextmanager
def deadline(seconds, reason):
    # Kaggle runs Linux. The alarm covers blocking setup and replay calls as well as Python loops.
    def expired(*_):
        raise TimeoutError(reason)

    handler = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    started = monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        if timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, timer[0] - (monotonic() - started)), timer[1])


@contextmanager
def timed(summary, phase):
    started = monotonic()
    summary["phase"] = phase
    write_json(Path("summary.json"), summary)
    try:
        yield
    finally:
        summary["minutes_per_phase"][phase] = (monotonic() - started) / 60
        write_json(Path("summary.json"), summary)


def tier_order(job):
    order = list(job.get("tier_order", TIERS))
    if not order or len(set(order)) != len(order) or not set(order) <= set(TIERS):
        raise ValueError(f"tier_order must list distinct tiers from {TIERS}, got {order}.")
    return order


def tier_grid(job, tier):
    if tier == "T1":
        t1 = job["t1"]
        configs = [dict(rank=rank, lr=lr, epochs=t1["epochs"])
                   for rank in t1["ranks"] for lr in t1["learning_rates"]]
        checkpoints = [t1["epochs"]]
    else:
        t2 = job["t2"]
        configs = [dict(rank=t2["rank"], lr=lr, epochs=max(t2["checkpoints"])) for lr in t2["learning_rates"]]
        checkpoints = t2["checkpoints"]
    if not 1 <= len(configs) * len(checkpoints) <= 6:
        raise ValueError("Each tier needs between one and six configurations.")
    if job["smoke"]:
        return [dict(configs[0], epochs=min(checkpoints))], [min(checkpoints)]
    return configs, checkpoints


def grid(job):
    """Configurations per tier, in the job's tier_order."""
    return {tier: tier_grid(job, tier) for tier in tier_order(job)}


def select_rows(job, fit, validation):
    if not job["smoke"]:
        return fit, validation
    fit_sample = Random(job["seed"]).sample(fit, job["smoke_fit_prompts"])
    tasks, chosen = Counter(), []
    for row in validation:
        task = row["task"]
        if tasks[task] < job["smoke_val_per_task"]:
            chosen.append(row)
            tasks[task] += 1
    if tasks != {task: job["smoke_val_per_task"] for task in TASKS}:
        raise ValueError("The validation inputs do not cover all four smoke tasks.")
    return fit_sample, chosen


def attach(model, tier, config):
    from specadapt.alignment import LoRAAdapter, OutputLogitAdapter

    if tier == "T1":
        return OutputLogitAdapter.attach(model, config["rank"])
    return LoRAAdapter(model, config["rank"])


def checked_merge(model, adapter, tier, example):
    import torch

    ids = torch.tensor([example.input_ids[-64:]], device=model.device)
    model.eval()
    with torch.inference_mode():
        before = model(input_ids=ids, use_cache=False).logits
    # Run merge outside inference mode so the result remains a regular module.
    if tier == "T1":
        model.set_output_embeddings(adapter.merged_linear())
    else:
        adapter.merge_and_unload()
    with torch.inference_mode():
        after = model(input_ids=ids, use_cache=False).logits
        error = (before - after).abs().max().item()
    if not error < MERGE_TOLERANCE:
        runtime.abort(f"{tier} max FP32 merge difference {error} must be below {MERGE_TOLERANCE}.")
    return error


def new_trainer(model, tokenizer, job, progress, phase):
    from specadapt.alignment import M2Trainer

    return M2Trainer(model, tokenizer.eos_token_id, job["batch_sequences"], job["batch_tokens"], job["seed"],
                     on_oom=lambda sequences, tokens: progress.update(
                         f"{phase}-oom-retry", max_sequences=sequences, max_tokens=tokens),
                     weight_decay=job["weight_decay"], warmup_fraction=job["warmup_fraction"])


def preflight(job, summary, progress):
    progress.update("preflight", action="check_inputs")
    draft_key, target_key = PAIRS[job["pair"]]
    if job["training_precision"] != "float32" or job["replay_precision"] != "float16" or job["seed"] != 42:
        raise ValueError("These jobs require FP32 training, FP16 replay and seed 42.")
    if (job["optimizer"] != "adamw" or job["weight_decay"] != 0
            or not 0 < job["warmup_fraction"] <= 1):
        raise ValueError("Use AdamW with zero weight decay and a positive warmup fraction.")
    candidates = list(Path("/kaggle/input").rglob(MANIFEST_NAME))
    if len(candidates) != 1:
        raise ValueError("Attach exactly one private specadapt-train-inputs dataset.")
    record, fit, val = load_target(candidates[0].parent, target_key)
    fit, val = select_rows(job, fit, val)
    plan = load_plan(Path("asset_plan.toml"))
    models = {entry["key"]: entry for entry in plan["models"]}
    draft_spec, target_spec = models[draft_key], models[target_key]
    if record["target"] != target_spec:
        raise ValueError("The recorded target differs from the revision in asset_plan.toml.")
    summary["provenance"].update(
        recording_run_id=RECORDING_RUN, recording_gpu=record["recording_gpu"], oracle_git_sha=record["git_sha"],
        draft_revision=draft_spec["revision"], target_revision=target_spec["revision"],
        dataset_revisions={entry["key"]: entry["revision"] for entry in record["datasets"]},
        oracle_sha256={name: data["oracle_sha256"] for name, data in record["files"].items()},
        inputs_sha256=sha256_file(candidates[0]),
    )
    summary.update(fit_prompts=len(fit), validation_prompts=len(val))
    progress.update("preflight", action="install_dependencies")
    setup()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specadapt.alignment import encode_example

    gpu = torch.cuda.get_device_name(0)
    summary["provenance"]["replay_gpu"] = gpu
    summary["gpu_config"].update(
        gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda, device=0, attention="sdpa",
        training_autocast=False, learning_rate_schedule="cosine",
        packages={name: version(name) for name in ("transformers", "accelerate", "sentencepiece", "safetensors",
                                                   "tokenizers", "huggingface-hub", "bitsandbytes")},
    )
    if "T4" not in gpu:
        runtime.abort(f"Expected a Kaggle T4 on device 0, got {gpu}.")
    torch.cuda.set_device(0)
    tokenizer_files = [name for name in target_spec["files"]
                       if name.startswith("tokenizer") or name in
                       {"config.json", "merges.txt", "vocab.json", "special_tokens_map.json"}]
    progress.update("preflight", action="load_tokenizers")
    target_dir = snapshot_download(target_spec["repo"], revision=target_spec["revision"],
                                   allow_patterns=tokenizer_files)
    target_tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
    if tokenizer_digest(target_tokenizer) != record["tokenizer_sha256"]:
        runtime.abort("Target tokenizer fingerprint differs from the recorded oracle.")
    progress.update("preflight", action="download_drafter")
    draft_dir = snapshot_download(draft_spec["repo"], revision=draft_spec["revision"],
                                  allow_patterns=draft_spec["files"])
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_dir, local_files_only=True)
    if not draft_tokenizer.is_fast or draft_tokenizer.eos_token_id is None:
        raise ValueError("The draft tokenizer needs offset mappings and an EOS token.")

    def load_draft(precision):
        return AutoModelForCausalLM.from_pretrained(draft_dir, local_files_only=True, dtype=precision,
                                                   device_map={"": 0}, attn_implementation="sdpa").eval()

    max_length = json.loads((Path(draft_dir) / "config.json").read_text())["max_position_embeddings"]
    fit_examples = [encode_example(draft_tokenizer, row, max_length) for row in fit]
    val_examples = [encode_example(draft_tokenizer, row, max_length) for row in val]
    for row in val:
        ids = target_tokenizer.encode(target_prompt(target_tokenizer, row["prompt"]), add_special_tokens=False)
        if ids != row["input_ids"] or decode(target_tokenizer, row["token_ids"]) != row["text"]:
            runtime.abort(f"Recorded target input or text differs for {row['prompt_id']}.")
    val_oracles = [row_oracle(row, record["eos_token_ids"], MAX_NEW_TOKENS) for row in val]
    example = min(fit_examples, key=lambda row: len(row.input_ids))
    summary["preflight_merge_error"] = {}
    for tier, (configs, _) in grid(job).items():
        torch.manual_seed(job["seed"])
        model = load_draft(torch.float32)
        cfg = dict(configs[0], epochs=1)
        adapter = attach(model, tier, cfg)
        trainer = new_trainer(model, draft_tokenizer, job, progress, f"preflight-{tier}")
        trainer.fit([example], cfg, on_epoch=lambda epoch: None,
                    on_step=lambda *args, **kw: progress.update(f"preflight-{tier}", *args, **kw))
        summary["preflight_merge_error"][tier] = checked_merge(model, adapter, tier, example)
        del trainer, adapter, model
        torch.cuda.empty_cache()
    return load_draft, draft_tokenizer, target_tokenizer, fit_examples, val_examples, list(zip(val, val_oracles))


def acceptance(counts):
    return dict(accepted=counts[0], proposed=counts[1], acceptance=counts[0] / counts[1] if counts[1] else 0.0)


def replay(model, draft_tokenizer, target_tokenizer, validation, job, tier, summary, progress):
    phase = f"{tier}-replay"
    task_counts = {task: [0, 0] for task in TASKS}
    counts = [0, 0]
    replay_seconds = 0.0
    progress.update(phase, 0, len(validation))
    for step, (row, oracle) in enumerate(validation, 1):
        data = generate(None, model, target_tokenizer, draft_tokenizer, row["prompt"],
                        mode="baseline", k=K, same_tokenizer=job["pair"] == "P1",
                        max_new_tokens=MAX_NEW_TOKENS, oracle=oracle)
        if tuple(data["token_ids"]) != oracle.token_ids:
            runtime.abort(f"{tier} replay departed from the recorded trajectory for {row['prompt_id']}.")
        for index, key in enumerate(("accepted", "proposed")):
            counts[index] += data[key]
            task_counts[row["task"]][index] += data[key]
        replay_seconds += data["replay_seconds"]
        current = dict(**acceptance(counts), per_task={task: acceptance(count) for task, count in task_counts.items()},
                       replayed_prompts=step, seconds_per_replayed_prompt=replay_seconds / step)
        summary["tiers"].setdefault(tier, {}).update(current)
        progress.update(phase, step, len(validation), acceptance=current["acceptance"])
        if step % 10 == 0 or step == len(validation):
            write_json(Path("summary.json"), summary)
    if job["smoke"] and job["pair"] == "P2" and tier == "B1" and tuple(counts) != SMOKE_COUNTS:
        runtime.abort(f"P2 smoke B1 counted {tuple(counts)}, expected {SMOKE_COUNTS} (accepted, proposed).")


def result_metrics(summary, tier):
    """The phase=result heartbeat for one tier: overall, B1, gain and acceptance per task."""
    data, base = summary["tiers"][tier], summary["tiers"]["B1"]
    return dict(tier=tier, acceptance=data["acceptance"], b1=base["acceptance"],
                gain_pp=summary[f"{tier.lower()}_gain_pp"],
                **{task: data["per_task"][task]["acceptance"] for task in TASKS})


def run_tiers(job, summary, progress, train_and_replay):
    for tier, (configs, checkpoints) in grid(job).items():
        train_and_replay(tier, configs, checkpoints)
        summary[f"{tier.lower()}_gain_pp"] = 100 * (
            summary["tiers"][tier]["acceptance"] - summary["tiers"]["B1"]["acceptance"])
        write_json(Path("summary.json"), summary)
        write_json(Path("artifacts/validation.json"), summary)
        progress.once("result", **result_metrics(summary, tier))


def fit_tier(tier, configs, checkpoints, load_draft, tokenizer, fit, val, job, summary, progress):
    import torch
    from safetensors.torch import load_file, save_file
    from specadapt.alignment import adapter_parameters

    path = Path("artifacts") / f"{tier.lower()}.safetensors"
    best = None
    trials = summary["trials"][tier] = []
    for index, config in enumerate(configs, 1):
        phase = f"{tier}-fit-{index}"
        progress.update(phase, action="load_drafter")
        torch.manual_seed(job["seed"])
        model = load_draft(torch.float32)
        attach(model, tier, config)
        trainer = new_trainer(model, tokenizer, job, progress, phase)

        def score(epoch):
            nonlocal best
            if epoch not in checkpoints:
                return
            metrics = trainer.evaluate(val, lambda *args, **kw: progress.update(
                f"{tier}-score-{index}-epoch-{epoch}", *args, **kw))
            trial = dict(config=dict(config, epochs=epoch, schedule_epochs=config["epochs"]), **metrics,
                         batch_sequences=trainer.max_sequences, batch_tokens=trainer.max_tokens)
            trials.append(trial)
            if best is None or metrics["matched_answer_tokens"] > best["matched_answer_tokens"]:
                best = trial
                state = {key: parameter.detach().cpu().contiguous()
                         for key, parameter in adapter_parameters(model).items()}
                temporary = path.with_suffix(".tmp")
                save_file(state, temporary, metadata={
                    "tier": tier, "pair": job["pair"], "config": json.dumps(trial["config"]),
                    "provenance": json.dumps(summary["provenance"]),
                })
                temporary.replace(path)
                summary["tiers"][tier] = dict(**best, weights=path.as_posix(), weights_sha256=sha256_file(path))
            write_json(Path("summary.json"), summary)

        trainer.fit(fit, config, on_epoch=score, on_step=lambda *args, **kw: progress.update(phase, *args, **kw))
        del trainer, model
        torch.cuda.empty_cache()
    model = load_draft(torch.float32)
    adapter = attach(model, tier, best["config"])
    state = load_file(path)
    if state.keys() != adapter_parameters(model).keys():
        raise ValueError(f"Unexpected parameter names in the {tier} adapter checkpoint.")
    model.load_state_dict(state, strict=False)
    summary["tiers"][tier]["merge_error"] = checked_merge(model, adapter, tier, val[0])
    return model.to(dtype=torch.float16)


def run(job, summary, progress):
    with timed(summary, "preflight"), deadline(job["preflight_minutes"] * 60, "Preflight exceeded five minutes."):
        load_draft, draft, target, fit, val, validation = preflight(job, summary, progress)
    import torch

    with timed(summary, "B1_replay"):
        progress.update("B1-replay", action="load_drafter")
        model = load_draft(torch.float16)
        replay(model, draft, target, validation, job, "B1", summary, progress)
        del model
        torch.cuda.empty_cache()
    write_json(Path("artifacts/validation.json"), summary)

    def train_and_replay(tier, configs, checkpoints):
        with timed(summary, f"{tier}_train"):
            model = fit_tier(tier, configs, checkpoints, load_draft, draft, fit, val, job, summary, progress)
        with timed(summary, f"{tier}_replay"):
            replay(model, draft, target, validation, job, tier, summary, progress)
            del model
            torch.cuda.empty_cache()

    run_tiers(job, summary, progress, train_and_replay)
    summary["seconds_per_replayed_prompt"] = {tier: data["seconds_per_replayed_prompt"]
                                               for tier, data in summary["tiers"].items()}


def main():
    job = runtime.job_config()
    if not 0 < job["preflight_minutes"] <= min(5, job["max_minutes"]):
        raise ValueError("Set a positive preflight budget of at most five minutes.")
    summary = dict(verdict="running", pair=job["pair"], mode="smoke" if job["smoke"] else "full",
                   validation_split="train/val", tier_order=tier_order(job), k=K,
                   max_new_tokens=MAX_NEW_TOKENS, replay_batch=1,
                   provenance=submission_provenance(), gpu_config=job.copy(),
                   tiers={}, trials={}, minutes_per_phase={})
    Path("artifacts").mkdir(exist_ok=True)
    try:
        with deadline(job["max_minutes"] * 60, "Job exceeded max_minutes."), Progress() as progress:
            run(job, summary, progress)
    except Exception as error:
        summary.update(verdict="fail", error=f"{type(error).__name__}: {error}")
        write_json(Path("summary.json"), summary)
        raise
    summary.update(verdict="pass", phase="complete")
    write_json(Path("summary.json"), summary)
    write_json(Path("artifacts/validation.json"), summary)


if __name__ == "__main__":
    main()
