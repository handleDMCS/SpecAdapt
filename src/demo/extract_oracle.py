"""Extract prompt-bound references from a completed submission JSON; no model loading.

Usage: uv run python -m demo.extract_oracle results/submission-TIMESTAMP.json
"""

import argparse
import json
from pathlib import Path

from demo.oracle import dumps, loads


def extract(source, destination):
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("Only completed submissions can supply an oracle.")
    candidates = [value for name, value in report["outputs"].items()
                  if Path(name).name == "target_references.toml"]
    if len(candidates) != 1 or not isinstance(candidates[0], str):
        raise ValueError("Expected one target_references.toml artifact. Older runs lack prompt/tokenizer "
                         "provenance; run src/demo/record_oracle.py or a new full demo first.")
    document = loads(candidates[0])
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an established reference.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(dumps(document))
    print(f"Extracted {len(document['references'])} references to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, default=Path("target_references.toml"))
    args = parser.parse_args()
    extract(args.submission, args.output)
