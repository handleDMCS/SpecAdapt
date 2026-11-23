"""Versioned, prompt-bound target references that can travel in the TOML bundle."""

import hashlib
import json
from pathlib import Path
import tomllib

from specadapt.decoding import target_prompt
from specadapt.oracle import GreedyOracle

TARGET_SETTINGS = dict(decoding="greedy_argmax", quantization="nf4", double_quant=True,
                       compute_dtype="float16", attention="sdpa", stopping="eos_or_token_limit")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def tokenizer_digest(tokenizer):
    tokenizer_state = dict(vocab=tokenizer.get_vocab(), special_tokens=tokenizer.special_tokens_map,
                           chat_template=tokenizer.chat_template,
                           backend=tokenizer.backend_tokenizer.to_str())
    return digest(tokenizer_state)


def make_contract(plan, tokenizer, eos_token_ids, versions, gpu):
    return dict(target=next(m for m in plan["models"] if m["key"] == "qwen_7b"),
                dataset=next(d for d in plan["datasets"] if d["key"] == "gsm8k"),
                split="test", tokenizer_sha256=tokenizer_digest(tokenizer),
                eos_token_ids=sorted(eos_token_ids), settings=TARGET_SETTINGS,
                versions=versions, gpu=gpu)


def make_reference(prompt_id, prompt, tokenizer, token_ids, max_new_tokens, eos_token_ids):
    oracle = GreedyOracle(token_ids, eos_token_ids)
    oracle.validate_limit(max_new_tokens)
    if len(token_ids) > max_new_tokens:
        raise ValueError("Reference exceeds its recording token limit.")
    return dict(prompt_id=prompt_id, prompt_sha256=digest(prompt),
                input_ids=tokenizer.encode(target_prompt(tokenizer, prompt), add_special_tokens=False),
                token_ids=list(token_ids), max_new_tokens=max_new_tokens,
                stop_reason="eos" if token_ids[-1] in eos_token_ids else "length")


def validate_document(document):
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported oracle schema.")
    contract = document["contract"]
    required = {"target", "dataset", "split", "tokenizer_sha256", "eos_token_ids", "settings", "versions", "gpu"}
    if set(contract) != required or contract["settings"] != TARGET_SETTINGS or contract["split"] != "test":
        raise ValueError("Incomplete or unsupported oracle contract.")
    records = document["references"]
    if not records or len({r["prompt_id"] for r in records}) != len(records):
        raise ValueError("Oracle needs nonempty, unique prompt records.")
    for record in records:
        if type(record["prompt_id"]) is not int or record["prompt_id"] < 0:
            raise ValueError("Invalid oracle prompt ID.")
        if not record["input_ids"] or any(type(t) is not int or t < 0 for t in record["input_ids"]):
            raise ValueError("Invalid oracle prompt tokens.")
        oracle = GreedyOracle(record["token_ids"], contract["eos_token_ids"])
        oracle.validate_limit(record["max_new_tokens"])
        if len(record["token_ids"]) > record["max_new_tokens"]:
            raise ValueError("Reference exceeds its recording token limit.")
        reason = "eos" if oracle.token_ids[-1] in oracle.eos_token_ids else "length"
        if record["stop_reason"] != reason:
            raise ValueError("Oracle termination metadata disagrees with its tokens.")
    return document


def dumps(document):
    validate_document(document)
    lines = ["# Greedy TEST references: verifier only; never seed the draft or M1 cache.",
             "schema_version = 1", f"fingerprint = {json.dumps(digest(document))}",
             f"contract_json = {json.dumps(json.dumps(document['contract'], ensure_ascii=False), ensure_ascii=False)}",
             f"source_json = {json.dumps(json.dumps(document['source']), ensure_ascii=False)}"]
    for record in document["references"]:
        lines.append("\n[[references]]")
        lines.extend(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in record.items())
    return "\n".join(lines) + "\n"


def loads(text):
    data = tomllib.loads(text)
    document = dict(schema_version=data["schema_version"], contract=json.loads(data["contract_json"]),
                    source=json.loads(data["source_json"]), references=data["references"])
    if digest(document) != data["fingerprint"]:
        raise ValueError("Oracle fingerprint mismatch; the reference file was changed or damaged.")
    return validate_document(document)


def load_for_prompts(path, contract, prompts, tokenizer, max_new_tokens):
    document = loads(Path(path).read_text(encoding="utf-8"))
    if document["contract"] != contract:
        changed = [key for key in contract if document["contract"].get(key) != contract[key]]
        raise ValueError(f"Oracle provenance differs ({', '.join(changed)}); record new references.")
    records = {record["prompt_id"]: record for record in document["references"]}
    oracles = {}
    for prompt_id, prompt in prompts.items():
        if prompt_id not in records:
            raise ValueError(f"Oracle has no reference for prompt {prompt_id}.")
        record = records[prompt_id]
        inputs = tokenizer.encode(target_prompt(tokenizer, prompt), add_special_tokens=False)
        if record["prompt_sha256"] != digest(prompt) or record["input_ids"] != inputs:
            raise ValueError(f"Oracle prompt/tokenization mismatch for prompt {prompt_id}.")
        oracle = GreedyOracle(record["token_ids"], contract["eos_token_ids"])
        oracle.validate_limit(max_new_tokens)
        oracles[prompt_id] = oracle
    return oracles, digest(document)
