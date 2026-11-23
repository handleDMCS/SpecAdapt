"""Download the pinned original Parquet files and draft tokenizer assets."""

import argparse
from pathlib import Path

from demo.assets import load_plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "training" / "raw"
DRAFTERS = ("llama_68m", "llama_160m")


def asset_dir(raw_dir, category, entry):
    return raw_dir / category / entry["key"] / entry["revision"]


def download_assets(plan, raw_dir):
    from huggingface_hub import hf_hub_download

    for entry in plan["datasets"]:
        directory = asset_dir(raw_dir, "datasets", entry)
        for filename in entry["files"]:
            print(f"Dataset {entry['repo']}: {filename}", flush=True)
            hf_hub_download(entry["repo"], filename, repo_type="dataset",
                            revision=entry["revision"], local_dir=directory)

    for entry in plan["models"]:
        if entry["key"] not in DRAFTERS:
            continue
        directory = asset_dir(raw_dir, "tokenizers", entry)
        print(f"Tokenizer and config: {entry['repo']}", flush=True)
        for filename in entry["files"]:
            if filename.startswith("tokenizer") or filename in {"config.json", "special_tokens_map.json"}:
                hf_hub_download(entry["repo"], filename, revision=entry["revision"], local_dir=directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW,
                        help=f"where to store the original files (default: {DEFAULT_RAW})")
    args = parser.parse_args()
    download_assets(load_plan(ROOT / "asset_plan.toml"), args.raw_dir)
    print(f"Original datasets and tokenizer assets are ready in {args.raw_dir.resolve()}")


if __name__ == "__main__":
    main()
