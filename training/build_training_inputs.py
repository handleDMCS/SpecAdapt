"""Join the recorded fit and validation inputs and optionally publish a private Kaggle dataset."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

from demo.assets import load_plan, wait_cache_ready
from demo.kaggle_run import get_api
from specadapt.training_inputs import (
    DATASET_SLUG, MANIFEST_NAME, MAX_NEW_TOKENS, RECORDING_RUN, SPLITS, TARGETS,
    check_prompt, indexed_rows, read_json, row_oracle, sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def join_rows(prompts, oracle_rows, count, eos_ids, limit):
    missing = prompts.keys() - oracle_rows.keys()
    extra = oracle_rows.keys() - prompts.keys()
    if missing or extra or len(prompts) != count:
        raise ValueError(f"Expected {count} joined rows, got {len(prompts)} prompts, "
                         f"{len(missing)} missing oracle rows and {len(extra)} extra oracle rows.")
    joined = []
    for key, prompt in prompts.items():
        record = oracle_rows[key]
        check_prompt(prompt)
        if prompt["prompt_sha256"] != record["prompt_sha256"]:
            raise ValueError(f"Oracle prompt hash mismatch for {key}.")
        if prompt["task"] != record["task"] or record["max_new_tokens"] != limit:
            raise ValueError(f"Oracle task or token limit mismatch for {key}.")
        row_oracle(record, eos_ids, limit)
        joined.append({**{field: prompt[field] for field in ("prompt_id", "task", "prompt", "prompt_sha256")},
                       **{field: record[field] for field in ("text", "input_ids", "token_ids", "stop_reason")}})
    return joined


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def build(prompt_root, oracle_root, output, plan, dataset_ref):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    budget = read_json(oracle_root / "budget.json")
    split_path = oracle_root / "splits.toml"
    if budget["run_id"] != RECORDING_RUN or sha256_file(split_path) != budget["splits_sha256"]:
        raise ValueError("Recording run or split manifest fingerprint differs.")
    with split_path.open("rb") as stream:
        split_manifest = tomllib.load(stream)
    prompts = {}
    for name, (relative, count) in SPLITS.items():
        indexed = indexed_rows(prompt_root / relative)
        entries = next(entry["rows"] for entry in split_manifest["files"] if entry["path"] == relative)
        frozen = {f"{row['source_dataset']}/{row['source_split']}/{row['row_id']}": row["prompt_sha256"]
                  for row in entries}
        if len(indexed) != count or {key: row["prompt_sha256"] for key, row in indexed.items()} != frozen:
            raise ValueError(f"Prompt coverage or hashes differ from the frozen split: {relative}")
        prompts[name] = indexed
    if prompts["fit-base"].keys() & prompts["val"].keys():
        raise ValueError("Fit and validation prompts overlap.")

    models = {entry["key"]: entry for entry in plan["models"]}
    manifest = dict(schema_version=1, recording_run_id=RECORDING_RUN, splits_sha256=budget["splits_sha256"],
                    targets={})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".train-inputs-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        for target in TARGETS:
            directory = oracle_root / target / "train"
            manifest_path = directory / "manifest.json"
            complete = read_json(directory / "complete.json")
            record = read_json(manifest_path)
            if (complete["target"] != target or complete["oracle"] != "train"
                    or complete["manifest_sha256"] != sha256_file(manifest_path)
                    or record["target"] != models[target] or record["target"] != budget["models"][target]
                    or record["split"] != "train" or record["splits_sha256"] != budget["splits_sha256"]
                    or record["git_sha"] != budget["git_sha"] or record["dtype"] != "float16"
                    or record["decoding"] != "greedy" or record["max_new_tokens"] != MAX_NEW_TOKENS
                    or record["recording_gpu"] != complete["recording_gpu"]):
                raise ValueError(f"Unexpected training oracle provenance for {target}.")
            record["files"] = {}
            for name, (_, count) in SPLITS.items():
                source = directory / f"{name}.jsonl"
                expected = complete["files"][source.name]
                if expected["rows"] != count or sha256_file(source) != expected["sha256"]:
                    raise ValueError(f"Oracle file checksum or row count mismatch for {target}/{source.name}.")
                joined = join_rows(prompts[name], indexed_rows(source), count,
                                   record["eos_token_ids"], MAX_NEW_TOKENS)
                destination = staging / f"{target}-{name}.jsonl"
                with destination.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in joined:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                record["files"][name] = dict(rows=len(joined), sha256=sha256_file(destination),
                                             oracle_sha256=expected["sha256"])
                print(f"Verified {target}/{name}: {len(joined)} rows", flush=True)
            manifest["targets"][target] = record
        write_json(staging / MANIFEST_NAME, manifest)
        write_json(staging / "dataset-metadata.json", {
            "id": dataset_ref, "title": DATASET_SLUG, "licenses": [{"name": "other"}],
            "description": "Private SpecAdapt training and validation prompts with recorded greedy teacher answers.",
        })
        staging.rename(output)
    return manifest


def publish(directory, ref):
    api = get_api()
    if api.get_config_value(api.CONFIG_NAME_USER) != ref.split("/")[0]:
        raise RuntimeError("The authenticated Kaggle account does not own the training dataset.")
    response = api.dataset_create_new(str(directory), public=False, quiet=True, convert_to_csv=False)
    if response.error:
        raise RuntimeError(f"Could not create private Kaggle dataset {ref}: {response.error}")
    wait_cache_ready(api, ref)
    print(f"Private dataset ready: https://www.kaggle.com/datasets/{ref}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "training/dataset")
    parser.add_argument("--oracles", type=Path, default=ROOT / "results/modal" / RECORDING_RUN / "oracle-dump")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / DATASET_SLUG)
    parser.add_argument("--upload", action="store_true", help="Create the private Kaggle dataset after validation")
    args = parser.parse_args()
    owner = read_json(ROOT / "kernel-metadata.json")["id"].split("/")[0]
    ref = f"{owner}/{DATASET_SLUG}"
    build(args.prompts, args.oracles, args.output, load_plan(ROOT / "asset_plan.toml"), ref)
    print(f"Training inputs: {args.output}")
    if args.upload:
        publish(args.output, ref)


if __name__ == "__main__":
    main()
