import json
import os
from pathlib import Path
import subprocess
import sys
import time

def load_env():
    """Load .env file into environment variables without extra dependencies."""
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

def run(cmd, capture=False):
    if capture:
        res = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=os.environ)
        return res.stdout.strip()
    return subprocess.run(cmd, shell=True, env=os.environ).returncode

def main():
    load_env()

    # 1. Read target kernel from kernel-metadata.json
    meta_path = Path("kernel-metadata.json")
    if not meta_path.exists():
        print("❌ kernel-metadata.json not found.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    kernel_id = meta.get("id")

    if not kernel_id or "<" in kernel_id:
        print("❌ Update 'id' in kernel-metadata.json with your actual Kaggle username first!")
        sys.exit(1)

    # 2. Push job to Kaggle
    print(f"🚀 Pushing to Kaggle: {kernel_id} ...")
    if run("kaggle kernels push -p .") != 0:
        print("❌ Push failed. Check your .env credentials.")
        sys.exit(1)

    # 3. Poll until done
    print("⏳ Waiting for GPU execution...")
    while True:
        status_msg = run(f"kaggle kernels status {kernel_id}", capture=True)
        print(f"   [{time.strftime('%H:%M:%S')}] {status_msg}")

        if "complete" in status_msg.lower():
            print("✅ Execution completed!")
            break
        elif "error" in status_msg.lower():
            print(f"❌ Kernel run failed:\n{status_msg}")
            sys.exit(1)

        time.sleep(10)

    # 4. Fetch results
    out_dir = Path("results/cuda_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Pulling results to {out_dir}/ ...")
    run(f"kaggle kernels output {kernel_id} -p {out_dir}")

    # 5. Display output log
    log_file = out_dir / "cuda_spec.txt"
    if log_file.exists():
        print("\n" + "=" * 45)
        print("📊 REMOTE GPU SPECIFICATION:")
        print("=" * 45)
        print(log_file.read_text())
    else:
        print(f"⚠️ cuda_spec.txt not found. Files downloaded to {out_dir}:")
        for f in out_dir.iterdir():
            print(f" - {f.name}")

if __name__ == "__main__":
    main()