# Multi-Query Attention (MQA) with KV-cache Implementation

## Benchmark

| Model Size | Attention Type | Average Latency (ms) | Average Peak Memory (MB) |
| ---------- | -------------- | -------------------- | ------------------------ |
| Small      | MQA            | 3023.94              | 629.45                   |
| Large      | MQA            | 2973.20              | 629.45                   |
| Small      | MHA            | 3561.46              | 760.68                   |
| Large      | MHA            | 3303.48              | 760.68                   |

ran on NVIDIA A10G

[View Config File](./config.yml)
