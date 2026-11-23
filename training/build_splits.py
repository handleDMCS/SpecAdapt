"""Merge prompt slices from the downloaded assets into one JSONL per role."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import tomllib

from demo.assets import load_plan
from demo.oracle import digest
from specadapt.decoding import draft_prompt
from training.download_datasets import DEFAULT_RAW, DRAFTERS, ROOT, asset_dir


SEED = 42
RESERVED_TOKENS = 512
NEAR_DUPLICATE_LIMIT = 2500
TASKS = {"gsm8k": "math", "humaneval": "code", "mbpp": "code",
         "alpaca": "instruction", "xsum": "summarization"}
SPLIT_PRIORITY = {"dev": 0, "test": 1, "validation": 2, "train": 3, "prompt": 4}
COLUMNS = {"gsm8k": ["question"], "humaneval": ["prompt"], "mbpp": ["text", "test_list"],
           "alpaca": ["instruction", "input"], "xsum": ["document"]}
DEFAULT_OUTPUT = ROOT / "training" / "dataset"

# Slice positions refer to each retained, shuffled source split.
FIT_LAYOUT = {
    "smoke": {
        ("gsm8k", "train"): slice(0, 10), ("mbpp", "prompt"): slice(0, 10),
        ("alpaca", "train"): slice(100, 110), ("xsum", "validation"): slice(0, 10),
    },
    "train/fit/base": {
        ("gsm8k", "train"): slice(10, 370), ("mbpp", "train"): slice(0, 360),
        ("alpaca", "train"): slice(110, 470), ("xsum", "train"): slice(0, 360),
    },
    "train/val": {
        ("gsm8k", "train"): slice(370, 450), ("mbpp", "validation"): slice(0, 80),
        ("alpaca", "train"): slice(470, 550), ("xsum", "train"): slice(360, 440),
    },
    "train/fit/extra": {
        ("gsm8k", "train"): slice(450, 1530), ("mbpp", "train"): slice(360, None),
        ("alpaca", "train"): slice(550, 1630), ("xsum", "train"): slice(440, 1520),
    },
}


@dataclass(frozen=True, slots=True)
class Prompt:
    dataset: str
    split: str
    row_id: int
    text: str
    normalized_sha256: str


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def render_prompt(dataset, row):
    if dataset == "gsm8k":
        return row["question"].strip()
    if dataset == "humaneval":
        return row["prompt"].strip()
    if dataset == "mbpp":
        return f"{row['text'].strip()}\n\nExample test:\n{row['test_list'][0].strip()}"
    if dataset == "alpaca":
        instruction = row["instruction"]
        return f"{instruction}\n\nInput:\n{row['input']}" if row["input"].strip() else instruction
    return f"Summarize the following article in one sentence.\n\n{row['document'].strip()}"


def load_drafters(plan, raw_dir):
    from transformers import AutoTokenizer

    tokenizers, contexts = [], {}
    models = {entry["key"]: entry for entry in plan["models"]}
    for key in DRAFTERS:
        directory = asset_dir(raw_dir, "tokenizers", models[key])
        with (directory / "config.json").open(encoding="utf-8") as stream:
            contexts[key] = json.load(stream)["max_position_embeddings"]
        tokenizers.append(AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True))
    return tokenizers, contexts


def eligible_prompts(plan, raw_dir, dev_ids, tokenizers, token_limit):
    import pyarrow.parquet as parquet

    for entry in plan["datasets"]:
        dataset = entry["key"]
        directory = asset_dir(raw_dir, "datasets", entry)
        row_offsets = defaultdict(int)
        for filename in sorted(entry["files"]):
            split = Path(filename).parent.name
            print(f"Checking prompts in {dataset}/{split}", flush=True)
            with parquet.ParquetFile(directory / filename) as source:
                for batch in source.iter_batches(batch_size=256, columns=COLUMNS[dataset]):
                    first_id = row_offsets[split]
                    row_offsets[split] += batch.num_rows
                    rendered = [(row_id, render_prompt(dataset, row))
                                for row_id, row in enumerate(batch.to_pylist(), first_id)]
                    rendered = [(row_id, text) for row_id, text in rendered if text.strip()]
                    if not rendered:
                        continue
                    inputs = [draft_prompt(text) for _, text in rendered]
                    token_ids = [tokenizer(inputs, add_special_tokens=True, truncation=False,
                                           verbose=False)["input_ids"] for tokenizer in tokenizers]
                    for (row_id, text), *encodings in zip(rendered, *token_ids, strict=True):
                        if all(len(encoding) <= token_limit for encoding in encodings):
                            normalized_hash = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
                            is_dev = (dataset, split) == ("gsm8k", "test") and row_id in dev_ids
                            yield is_dev, Prompt(dataset, split, row_id, text, normalized_hash)


def deduplicate_and_shuffle(prompts, dev_ids):
    winners = {}
    for is_dev, row in prompts:
        priority = (SPLIT_PRIORITY["dev" if is_dev else row.split],
                    row.dataset == "mbpp", row.row_id, row.dataset)
        current = winners.get(row.normalized_sha256)
        if current is None or priority < current[0]:
            winners[row.normalized_sha256] = (priority, row)

    dev, sources = [], defaultdict(list)
    for priority, row in winners.values():
        if priority[0] == SPLIT_PRIORITY["dev"]:
            dev.append(row)
        else:
            sources[row.dataset, row.split].append(row)
    dev.sort(key=lambda row: dev_ids.index(row.row_id))
    for rows in sources.values():
        rows.sort(key=lambda row: row.row_id)
        random.Random(SEED).shuffle(rows)
    return dev, sources


def near_duplicate(words, references):
    for other in references:
        size, other_size = len(words), len(other)
        if 5 * min(size, other_size) >= 4 * max(size, other_size):
            common = len(words & other)
            if 5 * common >= 4 * (size + other_size - common):
                return True
    return False


def select_splits(dev, sources):
    slices = {("dev", "gsm8k", "test"): dev}
    test_words = defaultdict(list)
    for group, offset in (("test/core", 0), ("test/expansion", 1)):
        for dataset, split, count in (("gsm8k", "test", 50), ("humaneval", "test", 25),
                                       ("mbpp", "test", 25), ("alpaca", "train", 50),
                                       ("xsum", "test", 50)):
            selected = sources[dataset, split][offset * count:(offset + 1) * count]
            slices[group, dataset, split] = selected
            test_words[TASKS[dataset]].extend(set(normalize(row.text).split()) for row in selected)

    for (dataset, split), rows in sources.items():
        if split == "test":
            continue
        # Preserve the frozen Alpaca test prefix and the documented slice offsets.
        start = 100 if dataset == "alpaca" and split == "train" else 0
        kept = [row for row in rows[start:NEAR_DUPLICATE_LIMIT]
                if not near_duplicate(set(normalize(row.text).split()), test_words[TASKS[dataset]])]
        rows[start:NEAR_DUPLICATE_LIMIT] = kept

    for group, source_slices in FIT_LAYOUT.items():
        for (dataset, split), positions in source_slices.items():
            slices[group, dataset, split] = sources[dataset, split][positions]
    return slices


def validate_splits(slices):
    keys, hashes = set(), set()
    for (group, dataset, split), rows in slices.items():
        for row in rows:
            key = (row.dataset, row.split, row.row_id)
            if key in keys or row.normalized_sha256 in hashes:
                raise ValueError(f"split overlap at {group}/{dataset}_{split}, source row {row.row_id}")
            keys.add(key)
            hashes.add(row.normalized_sha256)

        expected = None
        minimum = None
        if group == "dev":
            expected = 12
        elif group.startswith("test/"):
            expected = 25 if TASKS[dataset] == "code" else 50
        elif dataset != "mbpp":
            expected = {"smoke": 10, "train/fit/base": 360, "train/val": 80, "train/fit/extra": 1080}[group]
        else:
            minimum = {"smoke": 6, "train/fit/base": 340, "train/val": 70}.get(group)
        if expected is not None and len(rows) != expected:
            raise ValueError(f"{group}/{dataset}_{split} has {len(rows)} rows, expected {expected}")
        if minimum is not None and len(rows) < minimum:
            raise ValueError(f"{group}/{dataset}_{split} has {len(rows)} rows, requires at least {minimum}")


def write_splits(output_dir, slices, plan, contexts, token_limit):
    rows_by_role = defaultdict(list)
    for (role, _, _), rows in slices.items():
        rows_by_role[role].extend(rows)

    manifest = ["schema_version = 2", f"seed = {SEED}", f"max_prompt_tokens = {token_limit}"]
    for entry in plan["datasets"]:
        manifest.extend([f"\n[sources.{entry['key']}]", f"repo = {json.dumps(entry['repo'])}",
                         f"revision = {json.dumps(entry['revision'])}"])
    for entry in plan["models"]:
        if entry["key"] in contexts:
            manifest.extend([f"\n[drafters.{entry['key']}]", f"repo = {json.dumps(entry['repo'])}",
                             f"revision = {json.dumps(entry['revision'])}",
                             f"max_position_embeddings = {contexts[entry['key']]}"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".splits-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        for role, rows in rows_by_role.items():
            relative_path = f"{role}.jsonl"
            path = staging / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest.extend(["\n[[files]]", f"path = {json.dumps(relative_path)}", "rows = ["])
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    dataset, split = row.dataset, row.split
                    prompt_hash = digest(row.text)
                    stream.write(json.dumps(dict(prompt_id=f"{dataset}/{split}/{row.row_id}", task=TASKS[dataset],
                                                 source_dataset=dataset, source_split=split, row_id=row.row_id,
                                                 prompt=row.text, prompt_sha256=prompt_hash), ensure_ascii=False) + "\n")
                    manifest.append(f'  {{ source_dataset = "{dataset}", source_split = "{split}", '
                                    f'row_id = {row.row_id}, prompt_sha256 = "{prompt_hash}", '
                                    f'normalized_sha256 = "{row.normalized_sha256}" }},')
            manifest.append("]")
        (staging / "splits.toml").write_text("\n".join(manifest) + "\n", encoding="utf-8", newline="\n")
        staging.rename(output_dir)
    return {role: len(rows) for role, rows in rows_by_role.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW,
                        help=f"where the downloader stored the original files (default: {DEFAULT_RAW})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"where to create the dataset (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.exit(1, f"Output already exists: {output_dir}. Choose a new --output-dir to rebuild.\n")
    try:
        plan = load_plan(ROOT / "asset_plan.toml")
        dev_ids = tomllib.loads((ROOT / "pair2_demo.toml").read_text(encoding="utf-8"))["prompt_ids"]
        if len(dev_ids) != 12 or len(set(dev_ids)) != 12:
            raise ValueError("pair2_demo.toml must designate 12 distinct dev rows")
        tokenizers, contexts = load_drafters(plan, args.raw_dir)
        token_limit = min(contexts.values()) - RESERVED_TOKENS
        if token_limit <= 0:
            raise ValueError("draft context must leave room for the 512 reserved tokens")
        prompts = eligible_prompts(plan, args.raw_dir, set(dev_ids), tokenizers, token_limit)
        dev, sources = deduplicate_and_shuffle(prompts, dev_ids)
        slices = select_splits(dev, sources)
        validate_splits(slices)
        counts = write_splits(output_dir, slices, plan, contexts, token_limit)
    except FileNotFoundError as error:
        parser.exit(1, f"Missing file: {error.filename}. Run the dataset downloader first.\n")
    except ValueError as error:
        parser.exit(1, f"Dataset build failed: {error}\n")

    print(f"Wrote {sum(counts.values())} prompts across {len(counts)} JSONL files and splits.toml to {output_dir}")
    for role, count in counts.items():
        print(f"  {role}.jsonl: {count}")


if __name__ == "__main__":
    main()
