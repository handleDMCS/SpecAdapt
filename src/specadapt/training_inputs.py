"""Training-only data contract for the join and the Kaggle job."""

from hashlib import file_digest
import json

from demo.oracle import digest
from specadapt.oracle import GreedyOracle


DATASET_SLUG = "specadapt-train-inputs"
MANIFEST_NAME = f"{DATASET_SLUG}.json"
RECORDING_RUN = "20261123T154925641802Z"
TARGETS = ("qwen_7b", "vicuna_7b")
TASKS = ("math", "code", "instruction", "summarization")
SPLITS = {"fit-base": ("train/fit/base.jsonl", 1440), "val": ("train/val.jsonl", 320)}
MAX_NEW_TOKENS = 256


def sha256_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed_rows(path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["prompt_id"]
            if key in rows:
                raise ValueError(f"Duplicate prompt ID in {path}: {key}")
            rows[key] = row
    return rows


def check_prompt(row):
    if row["prompt_sha256"] != digest(row["prompt"]):
        raise ValueError(f"Prompt hash mismatch for {row['prompt_id']}.")
    if row["task"] not in TASKS:
        raise ValueError(f"Unknown task for {row['prompt_id']}.")


def row_oracle(row, eos_ids, limit):
    oracle = GreedyOracle(row["token_ids"], eos_ids)
    oracle.validate_limit(limit)
    stop_reason = "eos" if oracle.token_ids[-1] in oracle.eos_token_ids else "length"
    if len(oracle.token_ids) > limit or row["stop_reason"] != stop_reason:
        raise ValueError(f"Invalid oracle termination for {row['prompt_id']}.")
    if not isinstance(row["text"], str):
        raise ValueError(f"Invalid teacher text for {row['prompt_id']}.")
    return oracle


def load_target(directory, target):
    manifest = read_json(directory / MANIFEST_NAME)
    if manifest["schema_version"] != 1 or manifest["recording_run_id"] != RECORDING_RUN:
        raise ValueError("Unexpected training inputs schema or recording run.")
    record = manifest["targets"][target]
    if (record["target"]["key"] != target or record["split"] != "train"
            or record["max_new_tokens"] != MAX_NEW_TOKENS):
        raise ValueError(f"Unexpected training inputs contract for {target}.")
    rows = {}
    for name, (_, count) in SPLITS.items():
        path = directory / f"{target}-{name}.jsonl"
        if sha256_file(path) != record["files"][name]["sha256"]:
            raise ValueError(f"Training input checksum mismatch for {path.name}.")
        indexed = indexed_rows(path)
        if len(indexed) != count or record["files"][name]["rows"] != count:
            raise ValueError(f"{path.name} must contain exactly {count} rows.")
        rows[name] = indexed
        for row in indexed.values():
            check_prompt(row)
            row_oracle(row, record["eos_token_ids"], MAX_NEW_TOKENS)
    if rows["fit-base"].keys() & rows["val"].keys():
        raise ValueError(f"Fit and validation overlap for {target}.")
    return record, list(rows["fit-base"].values()), list(rows["val"].values())
