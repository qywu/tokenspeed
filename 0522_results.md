# MoE LoRA Optimization Results — 2026-05-22 (updated 2026-05-23)

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` · **bs=8** · **output\_tokens=200** · H100 80GB
**LoRA:** rank=16 · max\_loras=2 · TP=2 · CUDA graph mode
**Adapter format:** sglang\_shared (shared outer A, per-expert B for gate/up; per-expert A, shared B for down)

---

## Final Results (with fused Triton kernels)

| Configuration | tput (tok/s) | step (ms) | overhead |
|---|---:|---:|---:|
| **baseline** (no LoRA, triton) | **1394** | **5.74** | — |
| **n\_active=0** (LoRA loaded, inactive) | 1398 | 5.75 | **+0.01ms ✓** |
| **n\_active=1** (fused kernels) | **1107** | **7.22** | **+1.48ms** |

n\_active=0 matches baseline — loading an adapter costs nothing in decode.
n\_active=1 overhead: **1.48ms** = 26% of baseline step time.

---

## Decode Throughput Progress

Starting from 809 tok/s (no Triton fused kernels, plain PyTorch LoRA):

| Optimization | tput | step | overhead | Δ overhead |
|---|---:|---:|---:|---:|
| Baseline (no fused kernels) | 809 | 9.89ms | 4.12ms | — |
| + flat gate/up kernel | 818 | 9.78ms | 3.99ms | −130μs |
| + flat down shrink kernel | 827 | 9.68ms | 3.93ms | −60μs |
| + buffer+slot (no gather copies) | 927 | 8.63ms | 2.90ms | −1.03ms |
| + flat\_a\_gemm + scalings buffer | **1107** | **7.22ms** | **1.50ms** | **−1.40ms** |

**Total: +36.8% tput, −63.6% LoRA overhead (4.12ms → 1.50ms)**

---

## Fused Triton Kernels

All kernels live in `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe_lora/__init__.py`.
Integration is in `python/tokenspeed/runtime/lora/moe_lora.py`.

### 1. `compact_gate_up_expand` — flat per-expert GEMV (decode gate/up)

Replaces the all-experts GEMM + candidates.gather + route\_delta chain (3 separate ops):
```python
# Old (3 ops, reads all 128 experts' B data = 12.6 MB):
candidates = (lora_a_m @ w13_B.permute(2,0,1).reshape(r, E*I2)).view(m, E, I2)
delta = candidates.gather(1, safe_ids.unsqueeze(-1).expand(...))
_add_route_delta(gate_up_output, delta, ...)

# New (1 op, reads only active experts' B = ~5 MB, −60% bandwidth):
compact_gate_up_expand(lora_a_m, w13_B_buffer, slot_idx, safe_ids, gate_up_output, scalings)
```

Grid: `(I2//BLOCK_I, m*k)` — one block per flat-pair position. Computes `tok = pid_s // K`
directly inside the kernel. CUDA-graph compatible: reads `w13_B_buffer[slot]` and
`scalings[slot]` from device tensors without separate gather copies.

**Microbenchmark:** 20μs vs 69μs (3.4×) for the gate/up B expand step.

### 2. `flat_a_gemm` — A GEMM from buffer

Computes `lora_a_m = hidden @ w13_A_buffer[slot, 0, :, :].T` directly from the weight
buffer, eliminating:
- `w13_A = w13_A_buffers[layer][slot_idx].squeeze(0)` — 22μs gather copy
- `hidden @ w13_A[0].T` — 25μs cuBLAS GEMM (inefficient for m=8)

Grid: `(m, R//BLOCK_R)` — one block per token. With m=8 and R=32 fitting in L1 cache
across the 8 blocks, the kernel runs in ~5-8μs total.

**Savings:** 47μs/layer × 48 = **2.26ms** isolated.

### 3. `flat_down_shrink` — per-expert shrink from buffer

Replaces `_select_expert_weights(down_A, safe_ids) + einsum("mki,mkri->mkr", ...)`:
- Avoids the `(m*k, r, INTER)` = 1.5 MB intermediate tensor
- Reads `down_A_buffer[slot, exp, :, :]` directly for each flat pair

**Microbenchmark:** 23μs vs 54μs (2.4×).

### 4. `flat_down_expand` — shared B expand + scale + add

Fuses `lora_a @ down_B[slot, 0].T × topk_weight × scaling → down_output` in one kernel,
reading `down_B_buffer[slot]` and `scalings[slot]` directly from device memory.

### Key design decisions

**No gather copies:** All 4 kernels receive the full `(n_slots, ...)` weight buffer and
a `slot_ptr` GPU scalar. The kernel computes `buffer + slot * stride + ...` internally.
This eliminates 4 buffer gather copies per layer (previously ~64μs/layer × 48 = **3.08ms**).

**CUDA-graph safe:** `slot_ptr = bi.weight_indices[:1].clamp(0)` is a GPU tensor mutated
before each `graph.replay()`, so different adapters work without re-capturing the graph.

**Scalings in kernel:** `_flat_gate_up_expand_kernel` and `_flat_down_expand_kernel` load
`scalings[slot]` from the full `(n_slots,)` scalings buffer, eliminating 2 more
`scalings[slot_idx]` gather ops per layer (~19μs each × 2 × 48 = **1.82ms**).

---

## Earlier Optimizations (prefill / TTFT)

### Shared A/B fast path (sglang\_shared format)
When `w13_A.shape[0] == 1` (shared outer), use a single matmul instead of an
`O(m·k·r·h)` gather tensor. Saves 2.2 GB of intermediate tensor creation per prefill.

### Remove `torch.any(valid)` GPU→CPU sync
96 GPU→CPU stalls per prefill (48 layers × 2 ops) stalled the CPU-GPU pipeline.
**Impact: −35ms TTFT** (108ms → 73ms for sglang\_shared n=1 prefill).

### Vectorised scatter operations
`_add_route_delta` (−56%) and `_route_rows_from_cache` (−68%) replaced boolean-index
tensor creation with `scatter_` + slice.
**Impact: −11ms** on route scatter ops in prefill.

### CUDA graph: force has\_active\_lora=True during capture
During LoRA CUDA graph capture, `has_active_lora=True` and `single_lora_slot=0` are
forced so LoRA Triton kernels ARE recorded in the decode graph. Dynamic slot selection
uses `bi.weight_indices[:1].clamp(0)` (GPU tensor updated before each replay) so the
same graph serves any loaded adapter.

---

## Correctness

All correctness tests pass: `16 tests, 90 subtests` covering sglang\_shared and
per\_expert formats under sequential, batched, high-concurrency, and mixed-LoRA/base
scenarios (test\_qwen3\_moe\_per\_expert\_lora + test\_qwen3\_lora\_password\_adapters).
