"""Analyze downloaded Kaggle result ZIPs; does not run models or benchmarks.

Usage: uv run --locked python -m demo.compare_demo BASELINE.zip M1.zip
"""

import argparse
import json
from pathlib import Path
import random
import zipfile


def read_run(path, mode):
    with zipfile.ZipFile(path) as archive:
        if json.loads(archive.read("submission.json"))["status"] != "complete":
            raise ValueError(f"Submission did not complete: {path}")
        candidates = [name for name in archive.namelist() if name.endswith(f"/{mode}.jsonl")]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one {mode}.jsonl in {path}")
        rows = [json.loads(line) for line in archive.read(candidates[0]).decode().splitlines()]
        meta_path = candidates[0].removesuffix(".jsonl") + "_metadata.json"
        metadata = json.loads(archive.read(meta_path))
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prompt IDs")
    if {row["prompt_id"] for row in rows} != set(metadata["config"]["prompt_ids"]):
        raise ValueError("The run is missing configured prompt IDs")
    return {row["prompt_id"]: row for row in rows}, metadata


def compare(baseline_path, m1_path):
    baseline, baseline_meta = read_run(baseline_path, "baseline")
    m1, m1_meta = read_run(m1_path, "m1")
    if baseline.keys() != m1.keys():
        raise ValueError("Both runs must contain the same prompt IDs")
    for key in ("config_hash", "gpu", "models", "versions"):
        if baseline_meta[key] != m1_meta[key]:
            raise ValueError(f"Incompatible runs: {key} differs")
    differences = [m1[key]["acceptance_rate"] - baseline[key]["acceptance_rate"] for key in baseline]
    rng = random.Random(42)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                   for _ in range(baseline_meta["config"]["bootstrap_samples"]))
    from demo.benchmark import summarize
    result = {"baseline": summarize(list(baseline.values())), "m1": summarize(list(m1.values())),
              "paired_mean_acceptance_gain_pp": 100 * sum(differences) / len(differences),
              "paired_bootstrap_95_ci_pp": [100 * means[int(len(means) * q)] for q in (0.025, 0.975)],
              "same_target_reference_prompts": sum(baseline[key]["target_reference"]["token_ids"] ==
                                                   m1[key]["target_reference"]["token_ids"] for key in baseline),
              "same_speculative_output_prompts": sum(baseline[key]["token_ids"] == m1[key]["token_ids"]
                                                      for key in baseline),
              "baseline_archive": str(baseline_path), "m1_archive": str(m1_path)}
    result["throughput_ratio_m1_over_baseline"] = result["m1"]["tokens_per_second"] / result["baseline"]["tokens_per_second"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("m1", type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.m1)
    output = Path("results/pair2_comparison.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
