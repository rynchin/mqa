import torch
from torch import Tensor
from typing import Optional, List
if torch.cuda.is_available():
    from mqa_transformer.model import Model
else:
    from model import Model
import time

# Benchmark Step
def benchmark_step(model, input_ids: Tensor, start_pos: int = 0, k_cache: Optional[List[Optional[Tensor]]] = None, v_cache: Optional[List[Optional[Tensor]]] = None):
    if torch.cuda.is_available(): # gpu
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    
    with torch.no_grad():
        if torch.cuda.is_available():
            stream = torch.cuda.current_stream()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record(stream)

            # Forward pass with cache
            output, k_new, v_new = model(input_ids, start_pos, k_cache, v_cache)

            end_event.record(stream)

            # sync and compute ellapsed
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            
            # peak gpu
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
        else: # cpu
            start_time = time.time()
            output, k_new, v_new = model(input_ids, start_pos, k_cache, v_cache)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            peak_mem = 0

    return latency_ms, peak_mem, (output, k_new, v_new)
    
def benchmark(model, input_ids: Tensor, use_kv_cache:bool = False):
    batch, T = input_ids.shape

    if use_kv_cache:
        # 
        k_cache = [None] * len(model.blocks)
        v_cache = [None] * len(model.blocks)

        total_latency = 0
        total_mem = 0

        for t in range(T):
            input_t = input_ids[:, t:t+1]  # (batch, 1)
            start_pos = t

            latency_ms, mem, (output, k_cache, v_cache) = benchmark_step(model, input_t, start_pos, k_cache, v_cache)

            total_latency += latency_ms
            total_mem = max(total_mem, mem)  # peak mem over steps

        return total_latency, total_mem, None
    else:
        # full forward
        latency, mem, (output, k, v) = benchmark_step(model, input_ids, start_pos=0, k_cache=None, v_cache=None)
        return latency, mem, output

def local_benchmark(model_config: dict, bench_config: dict):
    # model init
    model = Model(
        vocab_size=model_config["vocab_size"],
        dim=model_config["dim"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        max_seq=model_config["max_seq"],
        p=model_config["dropout"],
        use_mqa=model_config["mqa"]
    ).to(bench_config["device"])

    # random input for benchmark
    input_ids = torch.randint(0, model_config["vocab_size"], (bench_config["batch_size"], model_config["max_seq"])).to(bench_config["device"])

    # benchmarking
    latencies = []
    memories = []

    for _ in range(bench_config["trials"]):
        latency, mem, _ = benchmark(model, input_ids, model_config["use_kv_cache"])
        latencies.append(latency)
        memories.append(mem)

    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(memories) / len(memories)

    return {
        "Average Latency": avg_latency,
        "Average peak memory": avg_memory,
        "# of trials": bench_config["trials"]
    }