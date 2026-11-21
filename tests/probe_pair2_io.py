"""Optional Kaggle-only diagnostic for model input read latency (no model load)."""
from pathlib import Path
import time

input_root = Path("/kaggle/input")
if not input_root.is_dir():
    raise RuntimeError("Submit this diagnostic to Kaggle.")
weights = next(input_root.rglob("model-00001-of-00004.safetensors"))
started = time.perf_counter()
total = 0
with weights.open("rb") as stream:
    for _ in range(16):
        block = stream.read(8 * 1024 * 1024)
        total += len(block)
elapsed = time.perf_counter() - started
print(f"Qwen input: read {total / 1024**2:.0f} MiB in {elapsed:.3f}s "
      f"({total / 1024**2 / elapsed:.1f} MiB/s)", flush=True)
