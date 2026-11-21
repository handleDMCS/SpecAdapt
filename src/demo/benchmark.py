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
    replay = all(row.get("verification") == "oracle" for row in rows)
    if not replay and any(row.get("verification") == "oracle" for row in rows):
        raise ValueError("Cannot mix live and oracle measurements.")
    proposed = sum(row["proposed"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    tokens = sum(row["generated_tokens"] for row in rows)
    seconds = sum(row["replay_seconds" if replay else "seconds"] for row in rows)
    rounds = sum(row["rounds"] for row in rows)
    bridge_seconds = sum(row["bridge_seconds"] for row in rows)
    draft_steps = sum(row["draft_steps"] for row in rows)
    draft_seconds = sum(row["draft_seconds"] for row in rows)
    result = dict(prompts=len(rows), generated_tokens=tokens, seconds=seconds,
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
    if replay:
        result["verification"] = "oracle"
        result["replay_seconds"] = result.pop("seconds")
        result.pop("tokens_per_second")
        result.pop("exact_matches")
    return result


def run_demo(mode, model_paths, dataset_paths, *, oracle_path=None, overrides=None):
    run_started = time.perf_counter()
    replay = oracle_path is not None
    if mode not in {"baseline", "m1", "compare", "record"} or (replay and mode == "record"):
        raise ValueError(f"Invalid benchmark mode: {mode}")
    setup()
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    from specadapt.decoding import CachedModel, draft_prompt, generate
    from demo.oracle import dumps, load_for_prompts, make_contract, make_reference

    root = Path(__file__).resolve().parents[2]
    config_bytes = (root / "pair2_demo.toml").read_bytes()
    config = tomllib.loads(config_bytes.decode())
    if overrides:
        if not replay or set(overrides) - {"k", "max_new_tokens", "prompt_ids"}:
            raise ValueError("Only replay workload overrides are supported.")
        config.update(overrides)
    if (type(config["k"]) is not int or config["k"] <= 0 or
            type(config["max_new_tokens"]) is not int or config["max_new_tokens"] <= 0):
        raise ValueError("k and max_new_tokens must be positive integers.")
    if (not config["prompt_ids"] or len(set(config["prompt_ids"])) != len(config["prompt_ids"]) or
            any(type(i) is not int or i < 0 for i in config["prompt_ids"])):
        raise ValueError("prompt_ids must be nonempty, unique nonnegative integers.")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("The pair 2 demo requires a Kaggle GPU.")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"This benchmark is pinned to T4; got {gpu}.")
    verification = "oracle" if replay else "live"
    print(f"P2 {mode}: {gpu}, verification={verification}, greedy, K={config['k']}, "
          f"max_new_tokens={config['max_new_tokens']}", flush=True)
    target_tokenizer = AutoTokenizer.from_pretrained(model_paths["qwen_7b"], local_files_only=True)
    draft_tokenizer = (AutoTokenizer.from_pretrained(model_paths["llama_68m"], local_files_only=True)
                       if mode != "record" else target_tokenizer)
    plan = tomllib.loads((root / "asset_plan.toml").read_text())
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((root / "src").rglob("*.py")))).hexdigest()
    metadata = dict(pair="P2", task="gsm8k", split="test", seed=config["seed"], gpu=gpu,
                    config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), source_hash=source_hash,
                    config=config, models=[m for m in plan["models"] if m["key"] in {"qwen_7b", "llama_68m"}],
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "bitsandbytes", "accelerate", "sentencepiece")})
    questions = pq.read_table(Path(dataset_paths["gsm8k"]) / "main/test/0000.parquet", columns=["question"]).to_pylist()
    prompts = {i: questions[i]["question"] for i in config["prompt_ids"]}
    eos = GenerationConfig.from_pretrained(model_paths["qwen_7b"], local_files_only=True).eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    contract = make_contract(plan, target_tokenizer, eos, metadata["versions"], gpu)
    metadata["verification"] = verification
    metadata["oracle_contract"] = contract
    if replay:
        oracles, fingerprint = load_for_prompts(oracle_path, contract, prompts, target_tokenizer, config["max_new_tokens"])
        metadata["oracle_fingerprint"] = fingerprint
        print(f"Oracle validated: {fingerprint}. No target model weights will be loaded.", flush=True)
    target = None
    if not replay:
        target = AutoModelForCausalLM.from_pretrained(
            model_paths["qwen_7b"], local_files_only=True, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
            torch_dtype=torch.float16, attn_implementation="sdpa").eval()
    draft = None
    if mode != "record":
        draft = AutoModelForCausalLM.from_pretrained(model_paths["llama_68m"], local_files_only=True,
                                                   torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0").eval()
    metadata["loaded_models"] = (["qwen_7b"] if target is not None else []) + (["llama_68m"] if draft is not None else [])
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    kwargs = dict(max_new_tokens=config["max_new_tokens"], k=config["k"])
    selected_modes = ["baseline", "m1"] if mode == "compare" else ([] if mode == "record" else [mode])
    output_names = {current: f"oracle_{current}" if replay else current for current in selected_modes}
    print("Warming up decoding paths on a separate synthetic prompt.", flush=True)
    if replay:
        # Warm only draft compute; do not expose a test reference to the proposer.
        with torch.inference_mode():
            warm_ids = draft_tokenizer.encode(draft_prompt("What is 2 + 2?"), add_special_tokens=True)
            CachedModel(draft).propose(warm_ids, 12, {draft_tokenizer.eos_token_id})
    else:
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
            if replay:
                raise ValueError("Oracle M1 requires the training-derived m1_seed_cache.toml. "
                                 "Generate it in a full M1 run; test references must never seed it.")
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
    oracle_records = []
    decoding_started = time.perf_counter()
    with ExitStack() as stack:
        streams = {current: stack.enter_context((output_dir / f"{output_names[current]}.jsonl").open("w", encoding="utf-8"))
                   for current in selected_modes}
        for current in selected_modes:
            (output_dir / f"{output_names[current]}_metadata.json").write_text(json.dumps(metadata, indent=2))
        for index, prompt_id in enumerate(config["prompt_ids"]):
            prompt = questions[prompt_id]["question"]
            modes = selected_modes[:] if replay else ["target", *selected_modes]
            offset = index % len(modes)
            modes = modes[offset:] + modes[:offset]
            result = {}
            for current in modes:
                result[current] = generate(target, draft, target_tokenizer, draft_tokenizer,
                                           prompt, mode=current, seed_cache=seed_cache,
                                           oracle=oracles[prompt_id] if replay else None, **kwargs)
            if not replay:
                reference = result["target"]
                reference["prompt_id"] = prompt_id
                references.append(reference)
                oracle_records.append(make_reference(prompt_id, prompt, target_tokenizer, reference["token_ids"],
                                                     config["max_new_tokens"], eos))
            for current in selected_modes:
                row = result[current]
                if replay:
                    row.update(prompt_id=prompt_id, oracle_fingerprint=metadata["oracle_fingerprint"],
                               config_hash=metadata["config_hash"], source_hash=source_hash,
                               pair="P2", seed=config["seed"], gpu=gpu, versions=metadata["versions"])
                    streams[current].write(json.dumps(row) + "\n")
                    streams[current].flush()
                    rows[current].append(row)
                    print(f"oracle {current} GSM8K test/{prompt_id}: acceptance={row['acceptance_rate']:.1%}, "
                          f"accepted/round={row['accepted_per_round']:.3f}, replay={row['replay_seconds']:.2f}s, "
                          f"cache={row['cache_rounds']}", flush=True)
                    continue
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
    metadata["decoding_wall_seconds"] = time.perf_counter() - decoding_started
    metadata["benchmark_wall_seconds"] = time.perf_counter() - run_started
    if not replay:
        document = dict(schema_version=1, contract=contract, references=oracle_records,
                        source=dict(source_hash=source_hash, versions=metadata["versions"]))
        (output_dir / "target_references.toml").write_text(dumps(document), encoding="utf-8")
    if mode == "record":
        (output_dir / "target_summary.json").write_text(json.dumps({"target": summarize(references), "metadata": metadata}, indent=2))
    for current in selected_modes:
        summary = {current: summarize(rows[current]), "metadata": metadata}
        if not replay:
            summary["target"] = summarize(references)
            summary[current]["speedup_vs_target"] = summary["target"]["seconds"] / summary[current]["seconds"]
        (output_dir / f"{output_names[current]}_metadata.json").write_text(json.dumps(metadata, indent=2))
        (output_dir / f"{output_names[current]}_summary.json").write_text(json.dumps(summary, indent=2))
        print("SUMMARY " + json.dumps({key: value for key, value in summary.items() if key != "metadata"}), flush=True)
        if not replay and not all(row["exact_match"] for row in rows[current]):
            print("OUTPUT CHECK: numerical drift detected; inspect saved target/speculative token IDs.", flush=True)
