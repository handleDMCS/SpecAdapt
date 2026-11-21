"""Reproducible P2 demo on Kaggle, with target-only output checks."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tomllib
import time
from contextlib import ExitStack


def setup():
    # Install before importing transformers; never install/download on the laptop.
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this experiment through kaggle_run.py on Kaggle.")
    requirements = {"transformers": "4.57.1", "accelerate": "1.11.0",
                    "bitsandbytes": "0.48.1", "sentencepiece": "0.2.1"}
    missing = []
    for name, version in requirements.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing.append(f"{name}=={version}")
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def summarize(rows):
    proposed = sum(row["proposed"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    tokens = sum(row["generated_tokens"] for row in rows)
    seconds = sum(row["seconds"] for row in rows)
    rounds = sum(row["rounds"] for row in rows)
    bridge_seconds = sum(row["bridge_seconds"] for row in rows)
    draft_steps = sum(row["draft_steps"] for row in rows)
    draft_seconds = sum(row["draft_seconds"] for row in rows)
    return dict(prompts=len(rows), generated_tokens=tokens, seconds=seconds,
                tokens_per_second=tokens / seconds, accepted=accepted, proposed=proposed,
                acceptance_rate=accepted / proposed if proposed else 0,
                accepted_per_round=accepted / rounds,
                mean_prompt_acceptance=sum(row["acceptance_rate"] for row in rows) / len(rows),
                exact_matches=sum(row.get("exact_match", True) for row in rows),
                cache_rounds=sum(row["cache_rounds"] for row in rows),
                bridge_ms_per_token=1000 * bridge_seconds / tokens,
                bridge_ms_per_round=1000 * bridge_seconds / rounds,
                draft_ms_per_step=1000 * draft_seconds / draft_steps if draft_steps else None,
                bridge_fraction_of_draft_step=(bridge_seconds / rounds) / (draft_seconds / draft_steps)
                if draft_steps and draft_seconds else None)


def run_demo(mode, model_paths, dataset_paths):
    setup()
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from specadapt.decoding import generate

    root = Path(__file__).resolve().parents[2]
    config_bytes = (root / "pair2_demo.toml").read_bytes()
    config = tomllib.loads(config_bytes.decode())
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("The pair 2 demo requires a Kaggle GPU.")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"This benchmark is pinned to T4; got {gpu}.")
    print(f"P2 {mode}: {gpu}, one GPU, 4-bit NF4 target, fp16 draft, greedy, K={config['k']}", flush=True)
    target_tokenizer = AutoTokenizer.from_pretrained(model_paths["qwen_7b"], local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(model_paths["llama_68m"], local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        model_paths["qwen_7b"], local_files_only=True, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        torch_dtype=torch.float16, attn_implementation="sdpa").eval()
    draft = AutoModelForCausalLM.from_pretrained(model_paths["llama_68m"], local_files_only=True,
                                               torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0").eval()
    plan = tomllib.loads((root / "asset_plan.toml").read_text())
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((root / "src").rglob("*.py")))).hexdigest()
    metadata = dict(pair="P2", task="gsm8k", split="test", seed=config["seed"], gpu=gpu,
                    config_hash=hashlib.sha256(config_bytes).hexdigest(), source_hash=source_hash,
                    config=config, models=[m for m in plan["models"] if m["key"] in {"qwen_7b", "llama_68m"}],
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "bitsandbytes", "accelerate", "sentencepiece")})
    questions = pq.read_table(Path(dataset_paths["gsm8k"]) / "main/test/0000.parquet", columns=["question"]).to_pylist()
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    kwargs = dict(max_new_tokens=config["max_new_tokens"], k=config["k"])
    selected_modes = ["baseline", "m1"] if mode == "compare" else [mode]
    print("Warming up decoding paths on a separate synthetic prompt.", flush=True)
    for warm_mode in ["target", *selected_modes]:
        generate(target, draft, target_tokenizer, draft_tokenizer, "What is 2 + 2?",
                 mode=warm_mode, max_new_tokens=12, k=config["k"])
    seed_cache = None
    if "m1" in selected_modes:
        from specadapt.bridge import FrozenContinuationCache
        seed_start = time.perf_counter()
        provenance = dict(target=next(m for m in plan["models"] if m["key"] == "qwen_7b"),
                          dataset=next(d for d in plan["datasets"] if d["key"] == "gsm8k"),
                          split="train", ids=config["cache_seed_train_ids"],
                          max_new_tokens=config["cache_seed_tokens"], versions=metadata["versions"])
        fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache_file = root / "m1_seed_cache.toml"
        if cache_file.exists():
            cached = tomllib.loads(cache_file.read_text())
            if cached["fingerprint"] != fingerprint:
                raise ValueError("Frozen M1 cache provenance differs; regenerate it in the pinned environment.")
            sequences = cached["sequences"]
            cache_origin = "bundled_train_cache"
        else:
            train = pq.read_table(Path(dataset_paths["gsm8k"]) / "main/train/0000.parquet",
                                  columns=["question"]).to_pylist()
            test_questions = {questions[i]["question"] for i in config["prompt_ids"]}
            sequences = []
            for train_id in config["cache_seed_train_ids"]:
                question = train[train_id]["question"]
                if question in test_questions:
                    raise ValueError("A cache training question overlaps the test prompts.")
                seeded = generate(target, draft, target_tokenizer, draft_tokenizer, question,
                                  mode="target", max_new_tokens=config["cache_seed_tokens"], k=config["k"])
                sequences.append(seeded["token_ids"])
                print(f"M1 cache warmup: GSM8K train/{train_id}, {len(seeded['token_ids'])} verified tokens", flush=True)
            cache_origin = "generated_from_train"
        cache_text = ("# Frozen Qwen greedy outputs on GSM8K TRAIN, never test references.\n"
                      + "fingerprint = " + json.dumps(fingerprint) + "\n"
                      + "provenance_json = " + json.dumps(json.dumps(provenance, sort_keys=True)) + "\n"
                      + "sequences = " + json.dumps(sequences) + "\n")
        (output_dir / "m1_seed_cache.toml").write_text(cache_text, encoding="utf-8")
        seed_cache = FrozenContinuationCache(sequences, k=config["k"])
        metadata["m1_cache"] = dict(origin=cache_origin, fingerprint=fingerprint, provenance=provenance,
                                    setup_seconds=time.perf_counter() - seed_start,
                                    tokens=sum(map(len, sequences)), entries=len(seed_cache.entries))
        print("M1 cache setup: " + json.dumps(metadata["m1_cache"]), flush=True)
    rows = {current: [] for current in selected_modes}
    references = []
    with ExitStack() as stack:
        streams = {current: stack.enter_context((output_dir / f"{current}.jsonl").open("w", encoding="utf-8"))
                   for current in selected_modes}
        for current in selected_modes:
            (output_dir / f"{current}_metadata.json").write_text(json.dumps(metadata, indent=2))
        for index, prompt_id in enumerate(config["prompt_ids"]):
            prompt = questions[prompt_id]["question"]
            modes = ["target", *selected_modes]
            offset = index % len(modes)
            modes = modes[offset:] + modes[:offset]
            result = {}
            for current in modes:
                result[current] = generate(target, draft, target_tokenizer, draft_tokenizer,
                                           prompt, mode=current, seed_cache=seed_cache, **kwargs)
            reference = result["target"]
            reference["prompt_id"] = prompt_id
            references.append(reference)
            for current in selected_modes:
                row = result[current]
                row.update(prompt_id=prompt_id, exact_match=row["token_ids"] == reference["token_ids"],
                           target_seconds=reference["seconds"], speedup=reference["seconds"] / row["seconds"],
                           pair="P2", seed=config["seed"], config_hash=metadata["config_hash"],
                           source_hash=source_hash, gpu=gpu, versions=metadata["versions"])
                streams[current].write(json.dumps({**row, "target_reference": reference}) + "\n")
                streams[current].flush()
                rows[current].append(row)
                print(f"{current} GSM8K test/{prompt_id}: acceptance={row['acceptance_rate']:.1%}, "
                      f"accepted/round={row['accepted_per_round']:.3f}, tok/s={row['tokens_per_second']:.2f}, "
                      f"speedup={row['speedup']:.2f}x, exact={row['exact_match']}, cache={row['cache_rounds']}", flush=True)
    for current in selected_modes:
        summary = {current: summarize(rows[current]), "target": summarize(references), "metadata": metadata}
        summary[current]["speedup_vs_target"] = summary["target"]["seconds"] / summary[current]["seconds"]
        (output_dir / f"{current}_summary.json").write_text(json.dumps(summary, indent=2))
        print("SUMMARY " + json.dumps({key: value for key, value in summary.items() if key != "metadata"}), flush=True)
        if not all(row["exact_match"] for row in rows[current]):
            print("OUTPUT CHECK: numerical drift detected; inspect saved target/speculative token IDs.", flush=True)
