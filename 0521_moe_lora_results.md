# MoE LoRA Decode Benchmark — 2026-05-22

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` · **bs=8** · **output_tokens=200** · 5 bench iters · rank=16 · max_loras=2 · H100 80GB

**n_active:** distinct LoRA adapters in batch (0 = enable_lora, all base model)

> MoE LoRA buffers ~1.96 GB/slot; max_loras=2 on 80 GB H100 with 30B model. gpu_util=0.86 for cudagraph+LoRA.

## TP1 Eager

| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |
|---|---:|---:|---:|
| baseline tp1 eager | 99.5 | 28.5 | 228.1 |
| baseline triton tp1 eager | 169.9 | 22.9 | 183.2 |
| per_expert tp1 eager n_active=0 | ERR | ERR | ERR |
| per_expert tp1 eager n_active=1 | ERR | ERR | ERR |
| per_expert tp1 eager n_active=2 | ERR | ERR | ERR |
| sglang_shared tp1 eager n_active=0 | ERR | ERR | ERR |
| sglang_shared tp1 eager n_active=1 | ERR | ERR | ERR |
| sglang_shared tp1 eager n_active=2 | ERR | ERR | ERR |

## TP1 CUDA Graph

| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |
|---|---:|---:|---:|
| baseline tp1 cudagraph | ERR | ERR | ERR |
| baseline triton tp1 cudagraph | ERR | ERR | ERR |
| per_expert tp1 cudagraph n_active=0 | ERR | ERR | ERR |
| per_expert tp1 cudagraph n_active=1 | ERR | ERR | ERR |
| per_expert tp1 cudagraph n_active=2 | ERR | ERR | ERR |
| sglang_shared tp1 cudagraph n_active=0 | ERR | ERR | ERR |
| sglang_shared tp1 cudagraph n_active=1 | ERR | ERR | ERR |
| sglang_shared tp1 cudagraph n_active=2 | ERR | ERR | ERR |

## TP2 Eager

| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |
|---|---:|---:|---:|
| baseline tp2 eager | ERR | ERR | ERR |
| baseline triton tp2 eager | ERR | ERR | ERR |
| per_expert tp2 eager n_active=0 | ERR | ERR | ERR |
| per_expert tp2 eager n_active=1 | ERR | ERR | ERR |
| per_expert tp2 eager n_active=8 | ERR | ERR | ERR |

## TP2 CUDA Graph

| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |
|---|---:|---:|---:|
| baseline tp2 cudagraph | ERR | ERR | ERR |
| baseline triton tp2 cudagraph | ERR | ERR | ERR |
| per_expert tp2 cudagraph n_active=0 | ERR | ERR | ERR |
| per_expert tp2 cudagraph n_active=1 | ERR | ERR | ERR |
| per_expert tp2 cudagraph n_active=8 | ERR | ERR | ERR |
