"""Submit both raw oracles for both Wave 1 targets as detached Modal jobs."""

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import monotonic, sleep, time

import modal

from modal_app import (GPU_RATE, MAX_TOKENS, MOUNT, ROOT, gpu_image, load_model, pinned_asset_dir,
                       read_prompts, read_spend, run_context, validate_asset, verify_dataset, within_budget, write_volume)


TARGETS = ("vicuna_7b", "qwen_7b")
SPLITS = {"test": {"test/core.jsonl": ("core.jsonl", 200),
                   "test/expansion.jsonl": ("expansion.jsonl", 200)},
          "train": {"train/fit/base.jsonl": ("fit-base.jsonl", 1440), "train/val.jsonl": ("val.jsonl", 320)}}
TEST_TIMEOUT = 4400
VICUNA_TIMEOUT = 6900
QWEN_TIMEOUT = 3500
TIMEOUTS = {("test", "vicuna_7b"): TEST_TIMEOUT, ("test", "qwen_7b"): TEST_TIMEOUT,
            ("train", "vicuna_7b"): VICUNA_TIMEOUT, ("train", "qwen_7b"): QWEN_TIMEOUT}
STARTUP_RESERVE = Decimal("0.25")
OVERNIGHT_CAP = Decimal("8.00")

app = modal.App("specadapt-overnight-oracles")
gpu_options = dict(image=gpu_image, volumes={str(MOUNT): write_volume}, gpu="A10", cpu=2, memory=24576,
                   scaledown_window=2)


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_once(path, data):
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError(f"Existing file has different provenance: {path.name}.") from None


@app.function(image=gpu_image, volumes={str(MOUNT): write_volume}, cpu=0.25, memory=512, timeout=90,
              scaledown_window=2)
def preflight(context, run_id, max_usd):
    root = Path(MOUNT)
    cap = Decimal(max_usd)
    jobs = {f"{target}/{kind}": str(seconds * GPU_RATE) for (kind, target), seconds in TIMEOUTS.items()}
    reserved = sum(map(Decimal, jobs.values()), STARTUP_RESERVE)
    spent = read_spend(root / "ledger.jsonl")
    if not within_budget(spent, reserved) or not cap.is_finite() or not reserved <= cap <= OVERNIGHT_CAP:
        raise ValueError(f"Cannot reserve ${reserved:.2f} at the ${cap:.2f} run cap and ${spent:.2f} recorded spend.")
    counts = verify_dataset(root / "dataset", context["plan"], context["dev_ids"])
    for files in SPLITS.values():
        for name, (_, count) in files.items():
            if counts[name] != count:
                raise ValueError(f"Expected {count} prompts in {name}.")
    models = {entry["key"]: entry for entry in context["plan"]["models"] if entry["key"] in TARGETS}
    for target in TARGETS:
        validate_asset(pinned_asset_dir(root, "models", models[target]), models[target])
    out = root / "oracles" / run_id
    out.mkdir(parents=True)
    run = dict(run_id=run_id, reserved_usd=str(reserved), startup_reserve_usd=str(STARTUP_RESERVE),
               max_usd=max_usd, job_reserves_usd=jobs, deadline_unix=time() + 8 * 60 * 60,
               splits_sha256=file_hash(root / "dataset/splits.toml"), git_sha=context["git_sha"],
               recorder_sha256=context["recorder_sha256"], models=models, datasets=context["plan"]["datasets"])
    write_once(out / "budget.json", run)
    write_volume.commit()
    return run


def left_padded_answers(model, input_ids, eos, pad):
    import torch

    width = max(map(len, input_ids))
    tokens = torch.tensor([[pad] * (width - len(ids)) + ids for ids in input_ids], device=model.device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in input_ids], device=model.device)
    output = model.generate(input_ids=tokens, attention_mask=mask, do_sample=False, max_new_tokens=MAX_TOKENS,
                            eos_token_id=eos, pad_token_id=pad)
    answers = output[:, width:].tolist()
    for index, ids in enumerate(answers):
        end = next((i + 1 for i, token in enumerate(ids) if token in eos), len(ids))
        answers[index] = ids[:end]
    return answers


def record(job, target_key, kind):
    import torch
    from demo.oracle import make_reference, tokenizer_digest
    from specadapt.bridge import decode
    from specadapt.decoding import generate, target_prompt

    root = Path(MOUNT)
    timeout = TIMEOUTS[kind, target_key]
    out = root / "oracles" / job["run_id"] / target_key / kind
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text(encoding="utf-8"))
    job_path = out.parent.parent / "budget.json"
    if json.loads(job_path.read_text(encoding="utf-8")) != job or time() >= job["deadline_unix"]:
        raise RuntimeError("Recording reservation is missing, changed, or expired.")
    started = monotonic()

    def remaining_time():
        return min(timeout - (monotonic() - started), job["deadline_unix"] - time())

    try:
        if file_hash(root / "dataset/splits.toml") != job["splits_sha256"]:
            raise ValueError("Frozen dataset manifest changed after preflight.")
        model, tokenizer = load_model(root, job["models"][target_key])
        raw_eos = model.generation_config.eos_token_id
        eos = [raw_eos] if isinstance(raw_eos, int) else raw_eos
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
        batch_size = 1 if kind == "test" else 16
        provenance = dict(schema_version=1, target=job["models"][target_key], split=kind,
                          max_new_tokens=MAX_TOKENS, batch_size=batch_size, dtype="float16", decoding="greedy",
                          attention="sdpa", eos_token_ids=eos, tokenizer_sha256=tokenizer_digest(tokenizer),
                          splits_sha256=job["splits_sha256"], recording_gpu=torch.cuda.get_device_name(0),
                          replay_gpu=None, git_sha=job["git_sha"], recorder_sha256=job["recorder_sha256"],
                          versions={name: version(name) for name in ("torch", "transformers", "accelerate", "sentencepiece")},
                          datasets=job["datasets"])
        write_once(out / "manifest.json", provenance)
        files = {}
        total = 0
        previous_commit = monotonic()
        for source, (filename, count) in SPLITS[kind].items():
            inputs = read_prompts(root, source)
            if len(inputs) != count:
                raise ValueError(f"Expected {count} inputs in {source}.")
            by_id = {row["prompt_id"]: row for row in inputs}

            def reference(row, token_ids):
                data = make_reference(row["prompt_id"], row["prompt"], tokenizer, token_ids, MAX_TOKENS, eos)
                return dict(data, task=row["task"], source_dataset=row["source_dataset"],
                            text=decode(tokenizer, token_ids))

            destination = out / filename
            seen = set()
            if destination.exists():
                with destination.open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        key = row["prompt_id"]
                        if (not line.endswith("\n") or key in seen or key not in by_id
                                or row != reference(by_id[key], row["token_ids"])):
                            raise ValueError(f"Invalid existing row {key!r} in {filename}.")
                        seen.add(key)
            remaining = [row for row in inputs if row["prompt_id"] not in seen]
            encoded = [(row, tokenizer.encode(target_prompt(tokenizer, row["prompt"]), add_special_tokens=False))
                       for row in remaining]
            if kind == "train":
                encoded.sort(key=lambda item: len(item[1]))
            pending = [encoded[i:i + batch_size] for i in range(0, len(encoded), batch_size)][::-1]
            with torch.inference_mode():
                while pending:
                    if remaining_time() <= 45:
                        raise TimeoutError("Recording deadline reached. Verified rows can be resumed.")
                    batch = pending.pop()
                    try:
                        if kind == "test":
                            output = generate(model, None, tokenizer, None, batch[0][0]["prompt"], mode="target",
                                              max_new_tokens=MAX_TOKENS)
                            answers = [output["token_ids"]]
                        else:
                            answers = left_padded_answers(model, [ids for _, ids in batch], eos, pad)
                    except torch.OutOfMemoryError:
                        if len(batch) == 1:
                            raise
                        middle = len(batch) // 2
                        pending.extend([batch[middle:], batch[:middle]])
                    else:
                        records = [reference(row, ids) for (row, _), ids in zip(batch, answers, strict=True)]
                        with destination.open("a", encoding="utf-8", newline="\n") as stream:
                            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
                        seen.update(row["prompt_id"] for row in records)
                        if not (out / "started.json").exists():
                            write_once(out / "started.json", dict(target=target_key, oracle=kind,
                                       gpu=provenance["recording_gpu"], first_batch_rows=len(records)))
                            write_volume.commit()
                            print(f"{target_key} {kind}: first {len(records)} records saved", flush=True)
                        elif monotonic() - previous_commit >= 45:
                            write_volume.commit()
                            previous_commit = monotonic()
                            print(f"{target_key} {kind}/{filename}: {len(seen)}/{count}", flush=True)
                        continue
                    torch.cuda.empty_cache()
            if len(seen) != count:
                raise ValueError(f"Expected {count} recorded rows in {filename}, found {len(seen)}.")
            files[filename] = dict(rows=count, sha256=file_hash(destination))
            total += count
            write_volume.commit()
            print(f"{target_key} {kind}/{filename}: {count}/{count}", flush=True)
        summary = dict(target=target_key, oracle=kind, rows=total, files=files,
                       manifest_sha256=file_hash(out / "manifest.json"), recording_gpu=provenance["recording_gpu"])
        write_once(out / "complete.json", summary)
        return summary
    except Exception as exc:
        path = out / f"stopped-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
        write_once(path, dict(error=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        elapsed = Decimal(str(monotonic() - started))
        path = out / f"spend-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
        write_once(path, dict(stage=f"{kind}_oracle", target=target_key, gpu_seconds=float(elapsed),
                              usd=str((elapsed * GPU_RATE).quantize(Decimal("0.000001")))))
        write_volume.commit()


@app.function(**gpu_options, timeout=TEST_TIMEOUT, max_containers=2)
def test_oracle(job, target):
    return record(job, target, "test")


@app.function(**gpu_options, timeout=VICUNA_TIMEOUT, max_containers=1)
def vicuna_train_oracle(job):
    return record(job, "vicuna_7b", "train")


@app.function(**gpu_options, timeout=QWEN_TIMEOUT, max_containers=1)
def qwen_train_oracle(job):
    return record(job, "qwen_7b", "train")


@app.local_entrypoint()
def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    context = run_context()
    context["recorder_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    job = preflight.remote(context, run_id, str(OVERNIGHT_CAP))
    print(f"Volume: specadapt/oracles/{run_id}. Reservation: ${job['reserved_usd']}", flush=True)
    calls = {}
    receipt = ROOT / "results/modal" / run_id / "overnight-submission.jsonl"
    receipt.parent.mkdir(parents=True)
    with receipt.open("x", encoding="utf-8") as stream:
        for kind, target in TIMEOUTS:
            if kind == "test":
                call = test_oracle.spawn(job, target)
            else:
                function = vicuna_train_oracle if target == "vicuna_7b" else qwen_train_oracle
                call = function.spawn(job)
            calls[target, kind] = call
            stream.write(json.dumps(dict(target=target, oracle=kind, function_call=call.object_id)) + "\n")
            stream.flush()
            print(f"Submitted {target}/{kind}: {call.object_id}", flush=True)
    deadline = monotonic() + 600
    while calls and monotonic() < deadline:
        for (target, kind), call in list(calls.items()):
            path = f"oracles/{run_id}/{target}/{kind}/started.json"
            try:
                status = json.loads(b"".join(write_volume.read_file(path)))
            except (FileNotFoundError, modal.exception.NotFoundError):
                try:
                    call.get(timeout=0)
                except TimeoutError:
                    continue
            else:
                print(f"Running {target}/{kind} on {status['gpu']}; first {status['first_batch_rows']} rows saved.",
                      flush=True)
            del calls[target, kind]
        if calls:
            sleep(8)
    if calls:
        raise TimeoutError(f"Submitted on Modal; still waiting for startup confirmation: {list(calls)}")
