"""Draft-only P2 replay; settings are in oracle_demo.toml, not script arguments."""
from pathlib import Path
import tomllib

from demo.benchmark import run_demo

root = Path(__file__).resolve().parents[2]
settings = tomllib.loads((root / "oracle_demo.toml").read_text(encoding="utf-8"))
reference_path = (root / settings["reference_file"]).resolve()
if not reference_path.is_relative_to(root):
    raise ValueError("reference_file must stay inside the bundled project.")
run_demo(settings["mode"], MODEL_PATHS, DATASET_PATHS, oracle_path=reference_path,
         overrides={key: settings[key] for key in ("max_new_tokens", "k", "prompt_ids") if key in settings})
