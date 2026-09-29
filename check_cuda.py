import subprocess

def main():
    lines = []
    try:
        import torch
        lines.append(f"PyTorch Version: {torch.__version__}")
        lines.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU Name: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"VRAM: {vram:.2f} GB")
    except ImportError:
        lines.append("PyTorch not installed.")

    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        lines.append("\n--- nvidia-smi ---\n" + smi)
    except Exception as e:
        lines.append(f"\nnvidia-smi failed: {e}")

    output_text = "\n".join(lines)
    print(output_text)
    
    with open("cuda_spec.txt", "w") as f:
        f.write(output_text)

if __name__ == "__main__":
    main()