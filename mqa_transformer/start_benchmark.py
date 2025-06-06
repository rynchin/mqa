from config import load_config
import torch
if torch.cuda.is_available():
    from mqa_transformer.benchmark import local_benchmark
else:
    from benchmark import local_benchmark
from pathlib import Path

if __name__ == "__main__":
    size = "large" # "large"

    config_path = Path(__file__).parent.parent / "config.yml"

    model_cfg, bench_cfg = load_config(config_path, size=size)

    if not torch.cuda.is_available():
        print("CUDA not available, now running on CPU")
        bench_cfg["device"] = "cpu"

    result = local_benchmark(model_cfg, bench_cfg)

    print(f"Results for {size}:")
    for k, v in result.items():
        print(f"{k}: {v}")
