"""Measure Wave 1 live smoke performance, then replay on Kaggle."""

from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from time import monotonic
import tomllib


def kaggle_replay(*, input_name="smoke_replay_input.toml", result_name="smoke_replay.json", validation=False):
    import torch
    from demo.benchmark import setup

    setup()
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from demo.oracle import digest, tokenizer_digest
    from specadapt.bridge import decode
    from specadapt.decoding import generate, target_prompt
    from specadapt.oracle import GreedyOracle

    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"Smoke replay requires a Kaggle T4, got {gpu}.")
    with Path(__file__).with_name(input_name).open("rb") as stream:
        payload = json.loads(tomllib.load(stream)["payload"])
    if validation and payload.get("split") != "train/val":
        raise ValueError("The cloned oracle smoke run requires a validation-only payload.")
    plan = payload["plan"]
    models = {entry["key"]: entry for entry in plan["models"]}
    results = {}
    for report in payload["smoke"]:
        target = report["target"]
        if validation and len({row["prompt_id"] for row in report["references"]}) != 40:
            raise ValueError(f"Expected 40 distinct validation prompts for {target}.")
        config = models[target]
        files = [name for name in config["files"] if name == "config.json" or name.startswith("tokenizer")
                 or name in {"merges.txt", "vocab.json", "special_tokens_map.json"}]
        target_tokenizer = AutoTokenizer.from_pretrained(snapshot_download(
            config["repo"], revision=config["revision"], allow_patterns=files), local_files_only=True)
        if tokenizer_digest(target_tokenizer) != report["tokenizer_sha256"]:
            raise ValueError(f"Target tokenizer fingerprint mismatch for {target}.")
        for pair, summary in report["pairs"].items():
            if not summary["eligible"]:
                continue
            draft_spec = models[summary["draft"]]
            directory = snapshot_download(draft_spec["repo"], revision=draft_spec["revision"],
                                          allow_patterns=draft_spec["files"])
            draft_tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            draft = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=torch.float16,
                                                       device_map={"": 0}, attn_implementation="sdpa").eval()
            accepted, proposed = 0, 0
            matched = 0
            for record in report["references"]:
                prompt = record["prompt"]
                inputs = target_tokenizer.encode(target_prompt(target_tokenizer, prompt), add_special_tokens=False)
                if digest(prompt) != record["prompt_sha256"] or inputs != record["input_ids"]:
                    raise ValueError(f"Prompt or input token mismatch for {record['prompt_id']}.")
                replay = generate(None, draft, target_tokenizer, draft_tokenizer, prompt, mode="baseline", k=3,
                                  same_tokenizer=pair == "P1", max_new_tokens=record["max_new_tokens"],
                                  oracle=GreedyOracle(record["token_ids"], report["eos_token_ids"]))
                accepted += replay["accepted"]
                proposed += replay["proposed"]
                if validation:
                    if replay["token_ids"] != record["token_ids"] or decode(target_tokenizer, record["token_ids"]) != record["text"]:
                        raise ValueError(f"Replay or decoded text differs for {pair}/{record['prompt_id']}.")
                    matched += 1
            rate = accepted / proposed if proposed else 0.0
            results[pair] = dict(accepted=accepted, proposed=proposed, acceptance=rate,
                                 recording_gpu=report["recording_gpu"], replay_gpu=gpu,
                                 git_sha=payload["git_sha"], draft_revision=draft_spec["revision"],
                                 target_revision=config["revision"])
            if validation:
                results[pair].update(split="train/val", recording_run_id=payload["run_id"], target=target,
                                     oracle_sha256=report["oracle_sha256"], checked_prompts=matched, passed=matched == 40)
                print(f"{pair}: verified {matched} validation traces; replay acceptance={rate:.6f}", flush=True)
            else:
                delta = abs(rate - summary["acceptance"])
                results[pair].update(live_acceptance=summary["acceptance"], delta=delta, passed=delta <= 0.005)
                print(f"{pair}: live={summary['acceptance']:.6f}, replay={rate:.6f}, delta={delta:.6f}", flush=True)
            del draft
            torch.cuda.empty_cache()
    destination = Path(result_name)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not results or any(not row["passed"] for row in results.values()):
        raise RuntimeError("The Kaggle smoke replay failed.")
    return results


def batch_probe(model, tokenizer, rows, eos, max_tokens):
    import torch
    from specadapt.decoding import target_prompt

    prompt_ids = [tokenizer.encode(target_prompt(tokenizer, row["prompt"]), add_special_tokens=False)
                  for row in rows[:32]]
    prompt_ids.sort(key=len)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    sizes = []

    def run(batch):
        width = len(batch[-1])
        inputs = torch.tensor([[pad] * (width - len(ids)) + ids for ids in batch], device=model.device)
        mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in batch], device=model.device)
        output = model.generate(input_ids=inputs, attention_mask=mask, do_sample=False, max_new_tokens=max_tokens,
                                pad_token_id=pad, eos_token_id=eos)
        count = 0
        for tokens in output[:, width:].tolist():
            count += next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))
        sizes.append(len(batch))
        return count

    pending = [prompt_ids[:16], prompt_ids[16:]]
    started = monotonic()
    tokens = 0
    while pending:
        batch = pending.pop(0)
        try:
            tokens += run(batch)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            pending[:0] = [batch[:middle], batch[middle:]]
        else:
            continue
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = monotonic() - started
    return dict(rows=32, tokens=tokens, seconds=elapsed, tokens_per_second=tokens / elapsed, batch_sizes=sizes)


if Path("/kaggle/working").is_dir():
    if __name__ == "__main__":
        kaggle_replay()
else:
    import modal
    from modal_app import (GPU_RATE, GPU_TIMEOUT, MAX_TOKENS, MOUNT, PAIRS, ROOT, append_spend, gpu_image,
                           load_model, read_prompts, require_budget, run_context, write_volume)

    app = modal.App("specadapt-stage1-smoke")

    @app.function(image=gpu_image, volumes={str(MOUNT): write_volume}, gpu="A10", cpu=2, memory=24576,
                  timeout=GPU_TIMEOUT)
    def smoke_target(context, target_key):
        import importlib.metadata
        import torch
        from demo.oracle import make_reference, tokenizer_digest
        from specadapt.decoding import generate

        root = Path(MOUNT)
        require_budget(root, GPU_TIMEOUT, GPU_RATE)
        models = {entry["key"]: entry for entry in context["plan"]["models"]}
        prompts = read_prompts(root, "smoke.jsonl")
        if len(prompts) != 40:
            raise ValueError(f"Expected 40 smoke prompts, found {len(prompts)}.")
        started = monotonic()
        try:
            target, tokenizer = load_model(root, models[target_key])
            raw_eos = target.generation_config.eos_token_id
            eos = [raw_eos] if isinstance(raw_eos, int) else raw_eos
            generate(target, None, tokenizer, None, prompts[0]["prompt"], mode="target", max_new_tokens=8)
            baseline = [generate(target, None, tokenizer, None, row["prompt"], mode="target",
                                 max_new_tokens=MAX_TOKENS) for row in prompts]
            b0_steps = sum(row["target_steps"] for row in baseline)
            target_step = sum(row["target_model_seconds"] for row in baseline) / b0_steps
            single_rate = sum(row["generated_tokens"] for row in baseline) / sum(row["seconds"] for row in baseline)
            references = [dict(make_reference(row["prompt_id"], row["prompt"], tokenizer, result["token_ids"],
                                              MAX_TOKENS, eos), prompt=row["prompt"])
                          for row, result in zip(prompts, baseline, strict=True)]
            probe = batch_probe(target, tokenizer, prompts, eos, MAX_TOKENS)
            pairs = {}
            for pair, (draft_key, verify_key) in PAIRS.items():
                if verify_key != target_key:
                    continue
                draft, draft_tokenizer = load_model(root, models[draft_key])
                if pair == "P1" and draft_tokenizer.get_vocab() != tokenizer.get_vocab():
                    raise ValueError("P1 token IDs differ between the draft and target vocabularies.")
                results = [generate(target, draft, tokenizer, draft_tokenizer, row["prompt"], mode="baseline", k=3,
                                    same_tokenizer=pair == "P1", max_new_tokens=MAX_TOKENS) for row in prompts]
                steps = sum(row["draft_steps"] for row in results)
                draft_step = sum(row["draft_seconds"] + row["bridge_seconds"] for row in results) / steps
                accepted, proposed = (sum(row[key] for row in results) for key in ("accepted", "proposed"))
                pairs[pair] = dict(draft=draft_key, ratio=draft_step / target_step, eligible=draft_step <= target_step / 2,
                                   accepted=accepted, proposed=proposed, acceptance=accepted / proposed if proposed else 0.0,
                                   same_output_count=sum(a["token_ids"] == b["token_ids"]
                                                         for a, b in zip(baseline, results, strict=True)))
                print(f"{pair}: {pairs[pair]}", flush=True)
                del draft
                torch.cuda.empty_cache()
            with (root / "dataset/splits.toml").open("rb") as stream:
                splits_sha = file_digest(stream, "sha256").hexdigest()
            report = dict(target=target_key, target_revision=models[target_key]["revision"],
                          tokenizer_sha256=tokenizer_digest(tokenizer), pairs=pairs, references=references,
                          eos_token_ids=eos, single_tokens_per_second=single_rate, batch=probe,
                          splits_sha256=splits_sha, recording_gpu=torch.cuda.get_device_name(0), replay_gpu=None,
                          git_sha=context["git_sha"], versions={name: importlib.metadata.version(name)
                              for name in ("torch", "transformers", "accelerate", "sentencepiece")})
        finally:
            cost = append_spend(root, "smoke", target_key, started, GPU_RATE, gpu=True)
        return dict(report, cost=cost)

    @app.local_entrypoint()
    def main(run_id: str = ""):
        context = run_context()
        if run_id and not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
            raise ValueError("Expected a previously printed smoke run ID.")
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = ROOT / "results/modal" / run_id
        out.mkdir(parents=True, exist_ok=True)
        print(f"Smoke run ID: {run_id}", flush=True)
        reports = []
        for target in dict.fromkeys(target for _, target in PAIRS.values()):
            destination = out / f"smoke-{target}.json"
            if destination.exists():
                report = json.loads(destination.read_text(encoding="utf-8"))
                pinned = next(entry["revision"] for entry in context["plan"]["models"] if entry["key"] == target)
                if (report["target"] != target or report["target_revision"] != pinned
                        or report["git_sha"] != context["git_sha"]):
                    raise ValueError(f"Existing smoke output has different provenance for {target}.")
            else:
                report = smoke_target.remote(context, target)
                with destination.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2)
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items() if k != "references"}, indent=2), flush=True)

        payload = dict(context, smoke=reports)
        scratch = ROOT / "modal/smoke_replay_input.toml"
        with scratch.open("x", encoding="utf-8") as stream:
            stream.write("payload = " + json.dumps(json.dumps(payload, ensure_ascii=True)) + "\n")
        try:
            subprocess.run(["uv", "run", "--locked", "python", "-m", "demo.kaggle_run", "--no-cache",
                            "modal/stage1_smoke.py", "--wait-timeout", "1800"], cwd=ROOT, check=True)
        finally:
            scratch.unlink()
