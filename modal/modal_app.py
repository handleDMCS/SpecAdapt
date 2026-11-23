"""Run SpecAdapt's read-only checks in a Modal CPU container."""

from collections import Counter
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import modal


ROOT = Path(__file__).resolve().parents[1]
MOUNT = PurePosixPath("/vol/specadapt")
ASSET_PLAN = PurePosixPath("/opt/specadapt/asset_plan.toml")
HF_SECRET_NAME = "huggingface-secret"
SPEND_CAP = Decimal("25.00")
PAIRS = {"P1": ("llama_68m", "vicuna_7b"), "P2": ("llama_68m", "qwen_7b"),
         "P4": ("llama_160m", "qwen_7b")}
MAX_TOKENS = 256
GPU_TIMEOUT = 2400
SYNC_TIMEOUT = 1800
# Modal published rates per GPU, physical CPU core, and GiB-second.
CPU_RATE = Decimal("0.0000131")
RAM_RATE = Decimal("0.00000222")
SYNC_RATE = 4 * CPU_RATE + 8 * RAM_RATE
GPU_RATE = Decimal("0.000306") + 2 * CPU_RATE + 24 * RAM_RATE
TASKS = {"gsm8k": "math", "humaneval": "code", "mbpp": "code",
         "alpaca": "instruction", "xsum": "summarization"}

# Counts by (original dataset, original split). A tuple is an inclusive range.
TEST_COUNTS = {("gsm8k", "test"): 50, ("humaneval", "test"): 25, ("mbpp", "test"): 25,
               ("alpaca", "train"): 50, ("xsum", "test"): 50}
ROLE_COUNTS = {
    "dev.jsonl": {("gsm8k", "test"): 12},
    "smoke.jsonl": {("gsm8k", "train"): 10, ("mbpp", "prompt"): (6, 10),
                    ("alpaca", "train"): 10, ("xsum", "validation"): 10},
    "train/fit/base.jsonl": {("gsm8k", "train"): 360, ("mbpp", "train"): (340, 360),
                             ("alpaca", "train"): 360, ("xsum", "train"): 360},
    "train/fit/extra.jsonl": {("gsm8k", "train"): 1080, ("mbpp", "train"): (0, 14),
                              ("alpaca", "train"): 1080, ("xsum", "train"): 1080},
    "train/val.jsonl": {("gsm8k", "train"): 80, ("mbpp", "validation"): (70, 80),
                        ("alpaca", "train"): 80, ("xsum", "train"): 80},
    "test/core.jsonl": TEST_COUNTS,
    "test/expansion.jsonl": TEST_COUNTS,
}

app = modal.App("specadapt-dry-run")
image = (modal.Image.debian_slim(python_version="3.12")
         .add_local_file(ROOT / "asset_plan.toml", str(ASSET_PLAN))
         .add_local_file(ROOT / "pair2_demo.toml", "/opt/specadapt/pair2_demo.toml"))
volume = modal.Volume.from_name("specadapt").with_mount_options(read_only=True)
secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
write_volume = modal.Volume.from_name("specadapt")
sync_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub==0.36.0")
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
             .uv_pip_install("transformers[sentencepiece]==4.57.1", "accelerate==1.11.0", "sentencepiece==0.2.1")
             .env({"PYTHONPATH": "/opt/specadapt/src:/opt/specadapt/modal", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_dir(ROOT / "src", "/opt/specadapt/src", ignore=["**/__pycache__/**"])
             .add_local_file(ROOT / "modal/modal_app.py", "/opt/specadapt/modal/modal_app.py")
             .add_local_file(ROOT / "asset_plan.toml", str(ASSET_PLAN))
             .add_local_file(ROOT / "pair2_demo.toml", "/opt/specadapt/pair2_demo.toml"))


def read_toml(path):
    with path.open("rb") as stream:
        return tomllib.load(stream)


def verify_dataset(root, plan, dev_ids):
    manifest = read_toml(root / "splits.toml")
    if manifest.get("schema_version") != 2 or manifest.get("seed") != 42:
        raise ValueError("Expected a hybrid dataset manifest with schema 2 and seed 42.")
    expected_sources = {entry["key"]: {"repo": entry["repo"], "revision": entry["revision"]}
                        for entry in plan["datasets"]}
    if manifest["sources"] != expected_sources:
        raise ValueError("Dataset revisions differ from asset_plan.toml.")
    draft_keys = {"llama_68m", "llama_160m"}
    expected_drafts = {entry["key"]: (entry["repo"], entry["revision"]) for entry in plan["models"]
                       if entry["key"] in draft_keys}
    drafts = manifest["drafters"]
    if {key: (entry["repo"], entry["revision"]) for key, entry in drafts.items()} != expected_drafts:
        raise ValueError("Drafter revisions differ from asset_plan.toml.")
    token_limit = min(entry["max_position_embeddings"] for entry in drafts.values()) - 512
    if token_limit <= 0 or token_limit != manifest["max_prompt_tokens"]:
        raise ValueError("The manifest's prompt limit disagrees with the drafter contexts.")

    files = {entry["path"]: entry["rows"] for entry in manifest["files"]}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.jsonl")}
    if set(files) != ROLE_COUNTS.keys() or actual != ROLE_COUNTS.keys() or len(manifest["files"]) != len(files):
        raise ValueError("Expected exactly the seven hybrid role files in the manifest and on the Volume.")
    seen_ids, seen_hashes = set(), set()
    counts = {}
    for path, expected in ROLE_COUNTS.items():
        origins = files[path]
        counts_by_source = Counter()
        with (root / path).open(encoding="utf-8") as stream:
            for number, (line, origin) in enumerate(zip(stream, origins, strict=True), 1):
                row = json.loads(line)
                dataset, split, row_id = (row["source_dataset"], row["source_split"], row["row_id"])
                source = (dataset, split)
                prompt_id = f"{dataset}/{split}/{row_id}"
                prompt = row["prompt"]
                prompt_hash = sha256(json.dumps(prompt, ensure_ascii=False).encode("utf-8")).hexdigest()
                normalized_hash = sha256(re.sub(r"\s+", " ", prompt.strip().lower()).encode("utf-8")).hexdigest()
                expected_origin = dict(source_dataset=dataset, source_split=split, row_id=row_id,
                                       prompt_sha256=prompt_hash, normalized_sha256=normalized_hash)
                if (source not in expected or type(row_id) is not int or row_id < 0 or not prompt.strip()
                        or row["task"] != TASKS[dataset] or row["prompt_id"] != prompt_id
                        or row["prompt_sha256"] != prompt_hash or origin != expected_origin):
                    raise ValueError(f"{path}, line {number} disagrees with the layout or manifest.")
                if prompt_id in seen_ids or normalized_hash in seen_hashes:
                    raise ValueError(f"{path}, line {number} repeats a source ID or normalized prompt.")
                seen_ids.add(prompt_id)
                seen_hashes.add(normalized_hash)
                counts_by_source[source] += 1
        for source, bounds in expected.items():
            lower, upper = (bounds, bounds) if isinstance(bounds, int) else bounds
            if not lower <= counts_by_source[source] <= upper:
                raise ValueError(f"Unexpected count for {source} in {path}: {counts_by_source[source]}.")
        counts[path] = sum(counts_by_source.values())
    if [row["row_id"] for row in files["dev.jsonl"]] != dev_ids:
        raise ValueError("Dev rows differ from pair2_demo.toml.")
    return counts


def verify_huggingface():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"Set HF_TOKEN in the {HF_SECRET_NAME} Modal secret.")
    request = Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected Hugging Face authentication status: HTTP {response.status}.")
    except HTTPError as error:
        raise RuntimeError(f"Hugging Face authentication failed with HTTP {error.code}.") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Could not reach Hugging Face to verify the token.") from None


def read_spend(path):
    total = Decimal(0)
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        return total
    with stream:
        for number, line in enumerate(stream, 1):
            try:
                amount = Decimal(str(json.loads(line, parse_float=Decimal)["usd"]))
            except (ValueError, KeyError, TypeError, InvalidOperation):
                raise ValueError(f"Invalid USD amount in the spend ledger at line {number}.") from None
            if not amount.is_finite() or amount < 0:
                raise ValueError(f"Invalid USD amount in the spend ledger at line {number}.")
            total += amount
    return total


def within_budget(spent, projected):
    return spent.is_finite() and projected.is_finite() and spent >= 0 and projected >= 0 and spent + projected <= SPEND_CAP


def run_context():
    return {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "plan": read_toml(ROOT / "asset_plan.toml"),
            "dev_ids": read_toml(ROOT / "pair2_demo.toml")["prompt_ids"]}


def require_budget(root, seconds, rate):
    spent = read_spend(root / "ledger.jsonl")
    projected = seconds * rate
    if not within_budget(spent, projected):
        raise RuntimeError(f"Projected cost ${projected:.2f} plus ${spent:.2f} exceeds the ${SPEND_CAP} cap.")


def append_spend(root, stage, target, started, rate, *, gpu=False):
    seconds = Decimal(str(time.monotonic() - started))
    row = dict(stage=stage, target=target, gpu_seconds=float(seconds) if gpu else 0,
               usd=str((seconds * rate).quantize(Decimal("0.000001"))))
    with (root / "ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    write_volume.commit()
    return row


def pinned_asset_dir(root, category, entry):
    return root / "assets" / category / entry["key"] / entry["revision"]


def validate_asset(directory, entry):
    manifest = json.loads((directory / "specadapt-files.json").read_text(encoding="utf-8"))
    if manifest["repo"] != entry["repo"] or manifest["revision"] != entry["revision"]:
        raise ValueError(f"Asset provenance mismatch for {entry['key']}.")
    sizes = manifest["files"]
    if set(sizes) != set(entry["files"]):
        raise ValueError(f"Incomplete asset manifest for {entry['key']}.")
    for name, size in sizes.items():
        if size <= 0 or (directory / name).stat().st_size != size:
            raise ValueError(f"Incorrect file size for {entry['key']}/{name}.")


def sync_assets(root, plan):
    from huggingface_hub import snapshot_download

    keys = {key for pair in PAIRS.values() for key in pair}
    models = [entry for entry in plan["models"] if entry["key"] in keys]
    for category, entries in (("models", models), ("datasets", plan["datasets"])):
        for entry in entries:
            directory = pinned_asset_dir(root, category, entry)
            marker = directory / "specadapt-files.json"
            if not marker.exists():
                print(f"Syncing {entry['key']} at {entry['revision']}", flush=True)
                snapshot_download(entry["repo"], revision=entry["revision"],
                                  repo_type="model" if category == "models" else "dataset",
                                  allow_patterns=entry["files"], local_dir=directory)
                sizes = {name: (directory / name).stat().st_size for name in entry["files"]}
                if any(size <= 0 for size in sizes.values()):
                    raise ValueError(f"Empty file in {entry['key']}.")
                with marker.open("x", encoding="utf-8") as stream:
                    json.dump(dict(repo=entry["repo"], revision=entry["revision"], files=sizes), stream)
                write_volume.commit()
            validate_asset(directory, entry)
    return sorted(keys)


def load_model(root, entry):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = pinned_asset_dir(root, "models", entry)
    validate_asset(directory, entry)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=torch.float16,
                                               device_map={"": 0}, attn_implementation="sdpa").eval()
    return model, tokenizer


def read_prompts(root, *files):
    from demo.oracle import digest

    manifest = read_toml(root / "dataset/splits.toml")
    origins_by_file = {entry["path"]: entry["rows"] for entry in manifest["files"]}
    rows = []
    for filename in files:
        with (root / "dataset" / filename).open(encoding="utf-8") as stream:
            for line, origin in zip(stream, origins_by_file[filename], strict=True):
                row = json.loads(line)
                fields = ("prompt_sha256", "source_dataset", "source_split", "row_id")
                if (digest(row["prompt"]) != row["prompt_sha256"]
                        or any(row[field] != origin[field] for field in fields)):
                    raise ValueError(f"Prompt {row['prompt_id']} disagrees with the manifest in {filename}.")
                rows.append(row)
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated prompt ID in the selected inputs.")
    return rows


@app.function(image=image, volumes={str(MOUNT): volume}, secrets=[secret], cpu=0.25, memory=256, timeout=60)
def dry_run():
    mount = Path(MOUNT)
    spent = read_spend(mount / "ledger.jsonl")
    if not within_budget(spent, Decimal(0)):
        raise RuntimeError(f"The spend ledger exceeds the ${SPEND_CAP} cap.")
    remaining = SPEND_CAP - spent
    if not within_budget(spent, remaining) or within_budget(spent, remaining + Decimal("0.01")):
        raise RuntimeError("The spend guard failed its boundary checks.")

    asset_plan = Path(ASSET_PLAN)
    plan = read_toml(asset_plan)
    dev_ids = read_toml(asset_plan.with_name("pair2_demo.toml"))["prompt_ids"]
    counts = verify_dataset(mount / "dataset", plan, dev_ids)
    verify_huggingface()
    return {"files": counts, "rows": sum(counts.values()), "huggingface_auth": "passed",
            "spend_guard": "passed", "ledger_usd": str(spent), "remaining_usd": str(remaining)}


@app.function(image=sync_image, volumes={str(MOUNT): write_volume}, secrets=[secret], cpu=4, memory=8192,
              timeout=SYNC_TIMEOUT)
def sync_wave1(context):
    root = Path(MOUNT)
    require_budget(root, SYNC_TIMEOUT, SYNC_RATE)
    counts = verify_dataset(root / "dataset", context["plan"], context["dev_ids"])
    started = time.monotonic()
    try:
        models = sync_assets(root, context["plan"])
    finally:
        cost = append_spend(root, "asset_sync", "wave1", started, SYNC_RATE)
    return dict(models=models, rows=sum(counts.values()), cost=cost, git_sha=context["git_sha"])


@app.local_entrypoint()
def main(sync: bool = False):
    result = sync_wave1.remote(run_context()) if sync else dry_run.remote()
    print(json.dumps(result, indent=2))
