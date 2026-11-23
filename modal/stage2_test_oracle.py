"""Record the Wave 1 test oracle after the Modal and Kaggle smoke gates pass."""

from decimal import Decimal
from hashlib import file_digest
import importlib.metadata
import json
import math
from pathlib import Path
import re
from time import monotonic

import modal

from modal_app import (GPU_RATE, MAX_TOKENS, MOUNT, PAIRS, ROOT, SPEND_CAP, append_spend, gpu_image,
                       load_model, read_prompts, read_spend, require_budget, run_context, within_budget, write_volume)


TEST_TIMEOUT = 4800
EXPECTED_SPLITS = {"core": 200, "expansion": 200}
app = modal.App("specadapt-stage2-test-oracle")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_or_verify(path, document):
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
    except FileExistsError:
        if read_json(path) != document:
            raise ValueError(f"Existing {path.name} has different provenance.") from None


def verify_gates(context, run_id, replay_result):
    if not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
        raise ValueError("Expected the smoke run ID printed by stage1_smoke.py.")
    submission = read_json(replay_result)
    if submission["status"] != "complete":
        raise ValueError("Kaggle smoke replay must finish successfully.")
    replay = next((value for path, value in submission["outputs"].items()
                   if Path(path).name == "smoke_replay.json"), {})
    models = {entry["key"]: entry for entry in context["plan"]["models"]}
    eligible = {}
    current_digests = set()
    for target in dict.fromkeys(target for _, target in PAIRS.values()):
        report = read_json(ROOT / "results/modal" / run_id / f"smoke-{target}.json")
        if (report["git_sha"] != context["git_sha"] or report["target"] != target
                or report["target_revision"] != models[target]["revision"] or "A10" not in report["recording_gpu"]):
            raise ValueError(f"Smoke provenance differs for {target}.")
        current_digests.add(report["splits_sha256"])
        expected_pairs = {pair for pair, (_, target_key) in PAIRS.items() if target_key == target}
        if report["pairs"].keys() != expected_pairs:
            raise ValueError(f"Smoke report is missing a planned pair for {target}.")
        for pair, live in report["pairs"].items():
            if not live["eligible"]:
                continue
            candidate = replay.get(pair, {})
            if (not candidate.get("passed") or candidate["git_sha"] != context["git_sha"]
                    or candidate["target_revision"] != report["target_revision"]
                    or candidate["draft_revision"] != models[PAIRS[pair][0]]["revision"]
                    or candidate["recording_gpu"] != report["recording_gpu"] or "T4" not in candidate["replay_gpu"]
                    or not math.isfinite(candidate["acceptance"])
                    or abs(candidate["acceptance"] - live["acceptance"]) > 0.005):
                raise ValueError(f"Kaggle replay did not pass the 0.005 acceptance gate for {pair}.")
            eligible.setdefault(target, []).append(pair)
        if target in eligible:
            single = report["single_tokens_per_second"]
            batched = report["batch"]["tokens_per_second"]
            if not all(math.isfinite(rate) and rate > 0 for rate in (single, batched)):
                raise ValueError(f"Smoke throughput is invalid for {target}.")
    if not eligible or set(replay) != {pair for pairs in eligible.values() for pair in pairs} or len(current_digests) != 1:
        raise ValueError("Smoke and replay artifacts do not authorize a consistent set of Wave 1 targets.")
    return eligible, current_digests.pop()


@app.function(image=gpu_image, volumes={str(MOUNT): write_volume}, gpu="A10", cpu=2, memory=24576,
              timeout=TEST_TIMEOUT)
def record_target(context, run_id, target_key, splits_sha, projection):
    import torch
    from demo.oracle import make_reference, tokenizer_digest
    from specadapt.bridge import decode
    from specadapt.decoding import generate

    root = Path(MOUNT)
    require_budget(root, TEST_TIMEOUT, GPU_RATE)
    if not within_budget(read_spend(root / "ledger.jsonl"), Decimal(projection)):
        raise RuntimeError(f"Projected remaining Wave 1 cost exceeds the ${SPEND_CAP} cap.")
    if hash_file(root / "dataset/splits.toml") != splits_sha:
        raise ValueError("The dataset manifest changed after the smoke run.")
    started = monotonic()
    try:
        configuration = next(entry for entry in context["plan"]["models"] if entry["key"] == target_key)
        target, tokenizer = load_model(root, configuration)
        raw_eos = target.generation_config.eos_token_id
        eos = [raw_eos] if isinstance(raw_eos, int) else raw_eos
        out = root / "oracles" / run_id / target_key / "test"
        out.mkdir(parents=True, exist_ok=True)
        provenance = dict(schema_version=1, target=configuration, split="test", max_new_tokens=MAX_TOKENS,
                          batch_size=1, dtype="float16", decoding="greedy", attention="sdpa", eos_token_ids=eos,
                          tokenizer_sha256=tokenizer_digest(tokenizer), splits_sha256=splits_sha,
                          recording_gpu=torch.cuda.get_device_name(0), replay_gpu=None, git_sha=context["git_sha"],
                          versions={name: importlib.metadata.version(name)
                                    for name in ("torch", "transformers", "accelerate", "sentencepiece")},
                          datasets=context["plan"]["datasets"])
        write_or_verify(out / "manifest.json", provenance)
        files = {}
        for split, expected_count in EXPECTED_SPLITS.items():
            inputs = read_prompts(root, f"test/{split}.jsonl")
            by_id = {row["prompt_id"]: row for row in inputs}
            if len(by_id) != expected_count:
                raise ValueError(f"Expected {expected_count} test {split} rows, found {len(by_id)}.")

            def reference(row, token_ids):
                return dict(make_reference(row["prompt_id"], row["prompt"], tokenizer, token_ids, MAX_TOKENS, eos),
                            task=row["task"], source_dataset=row["source_dataset"], text=decode(tokenizer, token_ids))

            finished = set()
            destination = out / f"{split}.jsonl"
            if destination.exists():
                with destination.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        prompt_id = record["prompt_id"]
                        if (not line.endswith("\n") or prompt_id in finished or prompt_id not in by_id
                                or record != reference(by_id[prompt_id], record["token_ids"])):
                            raise ValueError(f"Invalid or duplicated recorded test row {prompt_id}.")
                        finished.add(prompt_id)
            for row in inputs:
                if row["prompt_id"] in finished:
                    continue
                if monotonic() - started > TEST_TIMEOUT - 30:
                    raise TimeoutError("Stopping before the Modal timeout so the recording can resume.")
                result = generate(target, None, tokenizer, None, row["prompt"], mode="target", max_new_tokens=MAX_TOKENS)
                record = reference(row, result["token_ids"])
                with destination.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                finished.add(record["prompt_id"])
                if len(finished) % 20 == 0:
                    write_volume.commit()
                    print(f"{target_key} test/{split}: {len(finished)}/{expected_count}", flush=True)
            files[destination.name] = dict(rows=len(finished), sha256=hash_file(destination))
        summary = dict(target=target_key, rows=sum(item["rows"] for item in files.values()), files=files,
                       manifest_sha256=hash_file(out / "manifest.json"), volume_path=str(out),
                       git_sha=context["git_sha"], recording_gpu=provenance["recording_gpu"], replay_gpu=None)
        write_or_verify(out / "complete.json", summary)
    finally:
        cost = append_spend(root, "test_oracle", target_key, started, GPU_RATE, gpu=True)
    return dict(summary, cost=cost)


@app.local_entrypoint()
def main(smoke_run_id: str, replay_result: str):
    context = run_context()
    eligible, splits_sha = verify_gates(context, smoke_run_id, Path(replay_result))
    projected = Decimal(0)
    pending = {}
    for target, pairs in eligible.items():
        report = read_json(ROOT / "results/modal" / smoke_run_id / f"smoke-{target}.json")
        single = Decimal(str(report["single_tokens_per_second"]))
        batch = Decimal(str(report["batch"]["tokens_per_second"]))
        train_count = 5008 if batch >= 250 else 1760
        projected += 175 * (Decimal(200 * 5 * len(pairs)) / single + train_count / batch)
        output = ROOT / "results/modal" / smoke_run_id / f"test-oracle-{target}.json"
        if output.exists():
            summary = read_json(output)
            if summary["target"] != target or summary["rows"] != 400 or summary["git_sha"] != context["git_sha"]:
                raise ValueError(f"Existing test oracle summary has different provenance for {target}.")
            print(json.dumps(summary, indent=2), flush=True)
        else:
            test_seconds = Decimal(175 * 400) / single
            projected += test_seconds
            pending[target] = (output, test_seconds)
    # Include cold starts and leave headroom for the live sweep's draft and bridge work.
    rate = Decimal("1.10") * GPU_RATE
    projected = projected * rate + Decimal("1.00")
    print(f"Projected remaining Wave 1 cost: ${projected:.2f}", flush=True)
    for target, (output, test_seconds) in pending.items():
        summary = record_target.remote(context, smoke_run_id, target, splits_sha, str(projected))
        with output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        projected -= test_seconds * rate
