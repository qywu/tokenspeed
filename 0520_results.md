# LoRA Decode Benchmark — 2026-05-20

**Model:** `Qwen/Qwen3-8B` · **bs=8** · **output\_tokens=200** · 5 bench iters · rank=16 · n\_slots=8 · H100 80GB
**Adapters:** `togethercomputer/Qwen3-8B-LoRA-Password-Adapters`
**n\_active:** distinct LoRA adapters in the batch (0 = enable\_lora but all requests use base model)

---

## TP1 — All Adapter Types

| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |
|---|---:|---:|---:|
| baseline (no LoRA) · eager | 40.1 | 53.7 | 429.5 |
| baseline (no LoRA) · cudagraph | 27.7 | 141.4 | 1131.0 |
| **attn** · eager · n\_active=0 | 40.6 | 52.9 | 423.2 |
| **attn** · eager · n\_active=1 | 55.5 | 36.7 | 293.8 |
| **attn** · eager · n\_active=8 | 56.2 | 35.9 | 287.2 |
| **attn** · cudagraph · n\_active=0 | 27.2 | 134.7 | 1077.6 |
| **attn** · cudagraph · n\_active=1 | 35.9 | 133.8 | 1070.2 |
| **attn** · cudagraph · n\_active=8 | 35.4 | 133.6 | 1068.8 |
| **mlp** · eager · n\_active=0 | 38.8 | 54.1 | 433.0 |
| **mlp** · eager · n\_active=1 | 55.2 | 37.1 | 296.7 |
| **mlp** · eager · n\_active=8 | 55.5 | 36.2 | 289.6 |
| **mlp** · cudagraph · n\_active=0 | 28.2 | 134.5 | 1075.5 |
| **mlp** · cudagraph · n\_active=1 | 36.9 | 133.4 | 1066.5 |
| **mlp** · cudagraph · n\_active=8 | 37.0 | 133.3 | 1066.3 |
| **lm\_head** · eager · n\_active=0 | 39.4 | 53.5 | 428.2 |
| **lm\_head** · eager · n\_active=1 | 40.1 | 51.8 | 414.4 |
| **lm\_head** · eager · n\_active=8 | 40.3 | 51.5 | 411.9 |
| **lm\_head** · cudagraph · n\_active=0 | 28.1 | 133.9 | 1071.0 |
| **lm\_head** · cudagraph · n\_active=1 | 28.8 | 134.3 | 1074.2 |
| **lm\_head** · cudagraph · n\_active=8 | 28.7 | 134.0 | 1071.9 |

---

## TP1 vs TP2 — lm\_head LoRA

| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |
|---|---:|---:|---:|
| baseline tp1 · eager | 40.1 | 53.9 | 430.9 |
| baseline tp1 · cudagraph | 28.2 | 141.3 | 1130.4 |
| baseline tp2 · eager | 97.0 | 47.9 | 382.9 |
| baseline tp2 · cudagraph | 29.1 | 206.6 | **1651.9** |
| lm\_head tp1 · cudagraph · n\_active=0 | 28.0 | 134.5 | 1075.7 |
| lm\_head tp1 · cudagraph · n\_active=1 | 28.8 | 134.3 | 1074.1 |
| lm\_head tp1 · cudagraph · n\_active=8 | 28.9 | 134.0 | 1071.9 |
| lm\_head tp2 · cudagraph · n\_active=0 | 29.6 | 194.8 | 1557.7 |
| lm\_head tp2 · cudagraph · n\_active=1 | 29.7 | 194.6 | 1556.0 |
| lm\_head tp2 · cudagraph · n\_active=8 | 28.8 | 194.3 | 1553.4 |

---

## Summary

| | eager tput | cudagraph tput | LoRA overhead (cudagraph) | TTFT (cudagraph) |
|---|---:|---:|---:|---:|
| baseline tp1 | 429.5 | 1131.0 | — | 27–28 ms |
| attn LoRA tp1 | ~290 (−32%) | ~1069 (−5%) | −5% | 35–36 ms (+8 ms) |
| mlp LoRA tp1 | ~293 (−32%) | ~1066 (−6%) | −6% | 37 ms (+9 ms) |
| lm\_head LoRA tp1 | ~413 (−4%) | ~1073 (−5%) | −5% | 29 ms (+1 ms) |
| baseline tp2 | 382.9 | 1651.9 | — | 29 ms |
| lm\_head LoRA tp2 | — | ~1555 (−6%) | −6% | 29–30 ms |

**TP2 vs TP1 cudagraph speedup:** 1.46× (NCCL all-reduce prevents ideal 2×)

### Key findings

- **Eager mode**: attn/mlp LoRA costs ~32% throughput (Triton segmented-GEMM runs 36× per step, once per layer); lm\_head LoRA costs only ~4% (single matmul applied once)
- **Cudagraph**: all adapter types converge to ~5–6% overhead vs baseline — graph capture amortises per-layer Python launch cost
- **TTFT**: attn/mlp add ~8–9 ms even with cudagraph (LoRA kernels baked into the prefill graph across 36 layers); lm\_head adds <2 ms
- **n\_active 1→8**: negligible throughput difference under cudagraph (within 0.3%); in eager, ~2–3% degradation going from 1 to 8 distinct adapters
