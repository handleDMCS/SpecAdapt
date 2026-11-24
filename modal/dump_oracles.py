"""Clone a completed Modal oracle run and optionally submit a Kaggle validation smoke replay."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
import tomllib

from demo.assets import load_plan
from demo.oracle import digest
from specadapt.oracle import GreedyOracle


ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"P1": ("llama_68m", "vicuna_7b"), "P2": ("llama_68m", "qwen_7b"),
         "P4": ("llama_160m", "qwen_7b")}
SPLITS = {"test": {"core.jsonl": ("test/core.jsonl", 200), "expansion.jsonl": ("test/expansion.jsonl", 200)},
          "train": {"fit-base.jsonl": ("train/fit/base.jsonl", 1440), "val.jsonl": ("train/val.jsonl", 320)}}
TASKS = {"gsm8k": "math", "humaneval": "code", "mbpp": "code", "alpaca": "instruction", "xsum": "summarization"}
SMOKE_RESULT = "oracle_validation_smoke.json"
SMOKE_INPUT = "oracle_validation_input.toml"


def sha256_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def copy_file(volume, remote, destination, expected_sha=None):
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as stream:
            pending = Path(stream.name)
        try:
            with pending.open("wb") as stream:
                for chunk in volume.read_file(remote):
                    stream.write(chunk)
            if expected_sha is not None and sha256_file(pending) != expected_sha:
                raise ValueError(f"SHA-256 mismatch downloading {remote}.")
            pending.rename(destination)
        finally:
            pending.unlink(missing_ok=True)
    if expected_sha is not None and sha256_file(destination) != expected_sha:
        raise ValueError(f"Existing file has a different SHA-256: {destination}.")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_prompts(rows):
    return {f"{row['source_dataset']}/{row['source_split']}/{row['row_id']}": row for row in rows}


def verify_rows(path, origins, manifest):
    seen = set()
    eos = manifest["eos_token_ids"]
    limit = manifest["max_new_tokens"]
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = row["prompt_id"]
            origin = origins.get(key)
            inputs, tokens = row["input_ids"], row["token_ids"]
            if (not line.endswith("\n") or key in seen or not origin or row["prompt_sha256"] != origin["prompt_sha256"]
                    or row["source_dataset"] != origin["source_dataset"] or row["task"] != TASKS[row["source_dataset"]]
                    or not inputs or any(type(i) is not int or i < 0 for i in inputs) or not isinstance(row["text"], str)
                    or row["max_new_tokens"] != limit):
                raise ValueError(f"Invalid oracle row {number} in {path}.")
            GreedyOracle(tokens, eos).validate_limit(limit)
            if len(tokens) > limit or row["stop_reason"] != ("eos" if tokens[-1] in eos else "length"):
                raise ValueError(f"Invalid completion for {key} in {path}.")
            seen.add(key)
    if seen != origins.keys():
        raise ValueError(f"Oracle prompt coverage differs from splits.toml in {path}.")


def clone_run(volume, run_id, output):
    remote = f"oracles/{run_id}"
    copy_file(volume, f"{remote}/budget.json", output / "budget.json")
    run = read_json(output / "budget.json")
    if run["run_id"] != run_id or set(run["models"]) != {target for _, target in PAIRS.values()}:
        raise ValueError("Unexpected run ID or targets in the recording manifest.")
    copy_file(volume, "dataset/splits.toml", output / "splits.toml", run["splits_sha256"])
    with (output / "splits.toml").open("rb") as stream:
        split_manifest = tomllib.load(stream)
    if split_manifest["schema_version"] != 2:
        raise ValueError("Expected the hybrid dataset manifest, version 2.")
    origins_by_split = {entry["path"]: expected_prompts(entry["rows"]) for entry in split_manifest["files"]}
    row_count = 0
    for target, model in sorted(run["models"].items()):
        for role, splits in SPLITS.items():
            prefix = f"{remote}/{target}/{role}"
            directory = output / target / role
            copy_file(volume, f"{prefix}/complete.json", directory / "complete.json")
            complete = read_json(directory / "complete.json")
            expected_count = sum(count for _, count in splits.values())
            if (complete["target"] != target or complete["oracle"] != role or complete["rows"] != expected_count
                    or complete["files"].keys() != splits.keys()):
                raise ValueError(f"Unexpected completion summary for {target}/{role}.")
            copy_file(volume, f"{prefix}/manifest.json", directory / "manifest.json", complete["manifest_sha256"])
            manifest = read_json(directory / "manifest.json")
            if (manifest["target"] != model or manifest["split"] != role or manifest["splits_sha256"] != run["splits_sha256"]
                    or manifest["recorder_sha256"] != run["recorder_sha256"] or manifest["git_sha"] != run["git_sha"]
                    or manifest["datasets"] != run["datasets"] or manifest["max_new_tokens"] != 256
                    or manifest["dtype"] != "float16" or manifest["decoding"] != "greedy"
                    or manifest["recording_gpu"] != complete["recording_gpu"] or "A10" not in manifest["recording_gpu"]):
                raise ValueError(f"Oracle provenance differs for {target}/{role}.")
            for name, (split, count) in splits.items():
                origins = origins_by_split[split]
                if len(origins) != count or complete["files"][name]["rows"] != count:
                    raise ValueError(f"Unexpected prompt count for {target}/{role}/{name}.")
                local = directory / name
                copy_file(volume, f"{prefix}/{name}", local, complete["files"][name]["sha256"])
                verify_rows(local, origins, manifest)
            print(f"Verified {target}/{role}: {expected_count} rows", flush=True)
            row_count += expected_count
    print(f"Verified {row_count} oracle rows at {output}", flush=True)
    return run, origins_by_split["train/val.jsonl"]


def validation_payload(volume, output, run, origins):
    validation = output / "validation-prompts.jsonl"
    copy_file(volume, "dataset/train/val.jsonl", validation)
    chosen = {}
    checked = set()
    tasks = Counter()
    with validation.open(encoding="utf-8") as stream:
        for line in stream:
            prompt = json.loads(line)
            key = prompt["prompt_id"]
            origin = origins.get(key)
            if (key in checked or not origin or digest(prompt["prompt"]) != origin["prompt_sha256"]
                    or prompt["prompt_sha256"] != origin["prompt_sha256"]
                    or prompt["task"] != TASKS[origin["source_dataset"]]):
                raise ValueError(f"Validation prompt disagrees with the frozen manifest: {key}.")
            checked.add(key)
            if tasks[prompt["task"]] < 10:
                chosen[key] = prompt["prompt"]
                tasks[prompt["task"]] += 1
    if checked != origins.keys() or tasks != {task: 10 for task in set(TASKS.values())}:
        raise ValueError("Expected 320 validation prompts and 10 selected per task.")

    plan = load_plan(ROOT / "asset_plan.toml")
    models = {entry["key"]: entry for entry in plan["models"]}
    smoke = []
    for target, model in sorted(run["models"].items()):
        if models.get(target) != model:
            raise ValueError(f"The recording pin for {target} differs from asset_plan.toml.")
        directory = output / target / "train"
        manifest = read_json(directory / "manifest.json")
        with (directory / "val.jsonl").open(encoding="utf-8") as stream:
            indexed = {row["prompt_id"]: row for row in map(json.loads, stream) if row["prompt_id"] in chosen}
        references = [dict(indexed[key], prompt=prompt) for key, prompt in chosen.items()]
        smoke.append(dict(target=target, tokenizer_sha256=manifest["tokenizer_sha256"],
                          eos_token_ids=manifest["eos_token_ids"], recording_gpu=manifest["recording_gpu"],
                          oracle_sha256=sha256_file(directory / "val.jsonl"),
                          pairs={pair: {"draft": draft, "eligible": True}
                                 for pair, (draft, verifier) in PAIRS.items() if verifier == target},
                          references=references))
    return dict(plan=plan, git_sha=run["git_sha"], run_id=run["run_id"], smoke=smoke, split="train/val")


def submit_smoke(payload):
    metadata = read_json(ROOT / "kernel-metadata.json")
    owner = metadata["id"].split("/")[0]
    metadata.update(id=f"{owner}/specadapt-oracle-validation-smoke", title="SpecAdapt Oracle Validation Smoke")
    metadata_path = ROOT / "build/kaggle-oracle-smoke.json"
    metadata_path.parent.mkdir(exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    scratch = ROOT / "modal" / SMOKE_INPUT
    # The Kaggle runner includes this TOML in the private notebook bundle.
    with scratch.open("x", encoding="utf-8") as stream:
        stream.write("payload = " + json.dumps(json.dumps(payload, ensure_ascii=True)) + "\n")
    try:
        subprocess.run(["uv", "run", "--locked", "python", "-m", "demo.kaggle_run", "--metadata",
                        str(metadata_path.relative_to(ROOT)), "--no-cache", "modal/kaggle_oracle_smoke.py",
                        "--wait-timeout", "1800"], cwd=ROOT, check=True)
    finally:
        scratch.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="UTC run ID on the specadapt Volume")
    parser.add_argument("--output", type=Path, help="local clone directory")
    parser.add_argument("--kaggle-smoke", action="store_true", help="submit a replay using only 40 validation prompts per target")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}T\d{12}Z", args.run_id):
        parser.error("Expected the UTC run ID shown by the Modal recording launcher.")

    import modal

    output = args.output or ROOT / "results/modal" / args.run_id / "oracle-dump"
    volume = modal.Volume.from_name("specadapt", environment_name="main")
    run, origins = clone_run(volume, args.run_id, output)
    if args.kaggle_smoke:
        submit_smoke(validation_payload(volume, output, run, origins))


if __name__ == "__main__":
    main()
