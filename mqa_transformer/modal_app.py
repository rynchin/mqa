import modal
import torch
if torch.cuda.is_available():
    from mqa_transformer.benchmark import local_benchmark
else:
    from benchmark import local_benchmark
import json

app = modal.App(name="mqa-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "pyyaml")
)

# modal gpu
@app.function(gpu="A10G", image=image, timeout=600)
def benchmark_model(model_config_json: str, bench_config_json: str):
    model_config = json.loads(model_config_json)
    bench_config = json.loads(bench_config_json)

    return local_benchmark(model_config, bench_config)