# Multi-Query Attention (MQA) with KV-cache Implementation

First introduced [Shazeer, 2019](https://arxiv.org/pdf/1911.02150), MQA maintains a shared set of keys and values across all heads. In autoregressive generation, this reduces the size of the KV-cache by roughly a factor of `n_heads` per layer per token compared to MHA, reducing memory bandwidth during inference.

## Efficiency Benchmark

| Model Size | Attention Type | Average Latency (ms) | Average Peak Memory (MB) |
| ---------- | -------------- | -------------------- | ------------------------ |
| Small      | MQA            | 3023.94              | 629.45                   |
| Large      | MQA            | 2973.20              | 629.45                   |
| Small      | MHA            | 3561.46              | 760.68                   |
| Large      | MHA            | 3303.48              | 760.68                   |

- Observed 17% reduction in peak memory reflects the expected ≈`n_heads`× shrinkage of the KV-cache in MQA
- Here, the large model shows slightly lower latency than the small model. This is an artifact of `batch_size` = 1 and short sequences, where larger matmuls better utilize GPU tensor cores.

Ran using Modal's NVIDIA A10G GPU.

## Project Info
Run benchmark on local CPU: `python mqa_transformer/start_benchmark.py`

Run benchmark on GPU (Modal): `python mqa_transformer/run_modal.py`

Model Implementation w/ MQA: [model.py](./mqa_transformer/model.py)

Configuration: The benchmark configuration settings are defined in [config.yml](./config.yml)
