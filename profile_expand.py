"""Profile the decode expand kernel: bandwidth, FLOP utilization, config sweep.

Identifies the bottleneck (instruction-bound vs memory-bound) and sweeps
BLOCK_K up to 64/128 — larger BLOCK_K eliminates the inner K-loop entirely
for rank=64/128 adapters, removing loop overhead and k-mask instructions.

Usage:
    python profile_expand.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent / "tokenspeed-kernel" / "python"))

from tokenspeed_kernel._triton import triton as tok_triton
from tokenspeed_kernel.ops.lora.triton.kernel_utils import _resolve_token_positions

# ── minimal batch-info stub ────────────────────────────────────────────────────


@dataclass
class BI:
    bs: int
    max_len: int = 1
    seg_lens: torch.Tensor = None
    seg_indptr: torch.Tensor = None
    weight_indices: torch.Tensor = None
    lora_ranks: torch.Tensor = None
    scalings: torch.Tensor = None
    permutation: torch.Tensor = None

    def __post_init__(self):
        d = "cuda"
        self.seg_lens = torch.ones(self.bs, dtype=torch.int32, device=d)
        self.seg_indptr = torch.arange(self.bs + 1, dtype=torch.int32, device=d)
        self.weight_indices = torch.ones(self.bs, dtype=torch.int32, device=d)
        self.lora_ranks = torch.tensor([0, self.bs], dtype=torch.int32, device=d)
        self.scalings = torch.tensor([0.0, 1.0], dtype=torch.float32, device=d)


# ── inline expand kernel with configurable BLOCK_K ────────────────────────────


@triton.jit
def _expand_probe(
    x,
    weights,
    output,
    N,
    K,
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    K_real = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.max_contiguous(tl.arange(0, BLOCK_K), BLOCK_K)

    x_ptrs = (
        x
        + (seg_start + s_offset)[:, None] * x_stride_0
        + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    s_mask = s_offset[:, None] < seg_len
    n_mask = n_offset[None, :] < N
    partial = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_real, BLOCK_K)):
        k_rem = K_real - k * BLOCK_K
        x_tile = tl.load(
            x_ptrs,
            mask=s_mask & (k_offset[None, :] < k_rem),
            other=0.0,
            eviction_policy="evict_first",
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_rem) & n_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        partial += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial *= scaling
    partial = partial.to(x.dtype.element_ty)
    out_ptr = (
        output
        + (seg_start + s_offset)[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    out_mask = s_mask & n_mask
    partial += tl.load(out_ptr, mask=out_mask, other=0.0)
    tl.store(out_ptr, partial, mask=out_mask)


def run_probe(x, weights, output, bi, BLOCK_S, BLOCK_N, BLOCK_K, num_warps, num_stages):
    N, K = weights.shape[-2], weights.shape[-1]
    max_len = bi.max_len
    grid = (triton.cdiv(max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N), bi.bs)
    _expand_probe[grid](
        x,
        weights,
        output,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        bi.seg_lens,
        bi.seg_indptr,
        bi.weight_indices,
        bi.lora_ranks,
        bi.scalings,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ── metrics ────────────────────────────────────────────────────────────────────


def theoretical_bandwidth_gb(n_segs, N, K):
    """Min memory read in GB for one expand call."""
    w_bytes = n_segs * N * K * 2  # weights: n_segs adapter tiles
    x_bytes = n_segs * K * 2  # x: 1 row per segment
    out_bytes = n_segs * N * 2 * 2  # output read+write
    return (w_bytes + x_bytes + out_bytes) / 1e9


def flops(n_segs, N, K):
    return n_segs * 2 * N * K  # 2 × N × K per token


def bench_cfg(fn, warmup=15, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep) * 1e-3  # → seconds


# ── main sweep ─────────────────────────────────────────────────────────────────


def sweep(n_segs: int, rank: int, N: int, label: str) -> None:
    dev, dt = "cuda", torch.bfloat16
    bi = BI(bs=n_segs)
    bi.lora_ranks = torch.tensor([0, rank], dtype=torch.int32, device=dev)
    x = torch.randn(n_segs, rank, device=dev, dtype=dt)
    w = torch.randn(2, N, rank, device=dev, dtype=dt)
    o = torch.zeros(n_segs, N, device=dev, dtype=dt)

    h100_bw = 3.35e12  # bytes/s
    h100_tflops = 2e15  # bf16 tensor core peak

    bw_floor = theoretical_bandwidth_gb(n_segs, N, rank) / h100_bw * 1e6  # µs
    flop_floor = flops(n_segs, N, rank) / h100_tflops * 1e6  # µs

    print(f"\n{'='*72}")
    print(f"  {label}  n_segs={n_segs}  rank={rank}  N={N}")
    print(f"  Bandwidth floor: {bw_floor:.1f}µs  |  FLOP floor: {flop_floor:.2f}µs")
    print(
        f"  {'BLOCK_S':>7}  {'BLOCK_N':>7}  {'BLOCK_K':>7}  {'warps':>5}  {'stg':>3}  {'µs':>8}  {'BW%':>6}  {'K-iters':>8}"
    )
    print(f"  {'-'*66}")

    configs = [
        # (BLOCK_S, BLOCK_N, BLOCK_K, num_warps, num_stages)
        # Current best from autotune:
        (16, 64, 16, 8, 3),
        (16, 64, 32, 8, 3),
        # Larger BLOCK_K — KEY EXPERIMENT:
        #   rank=64 → BLOCK_K=64: 1 K-iteration, no k-mask, no loop overhead
        #   rank=128 → BLOCK_K=128: same
        (16, 64, 64, 8, 1),
        (16, 64, 64, 4, 1),
        (16, 64, 64, 8, 2),
        (16, 128, 64, 4, 1),
        (16, 128, 64, 8, 1),
        (16, 64, 128, 8, 1) if rank >= 128 else None,
        (16, 128, 128, 4, 1) if rank >= 128 else None,
        # Wider BLOCK_N to reduce CTA count:
        (16, 128, 16, 4, 2),
        (16, 128, 32, 4, 2),
        (32, 64, 16, 4, 2),
        (32, 64, 32, 4, 2),
    ]

    best_t = float("inf")
    best_cfg = None

    for cfg in configs:
        if cfg is None:
            continue
        BS, BN, BK, nw, ns = cfg
        if BK > rank:  # BLOCK_K larger than actual K makes no sense
            continue
        try:
            t_s = bench_cfg(lambda: run_probe(x, w, o.clone(), bi, BS, BN, BK, nw, ns))
            t_us = t_s * 1e6
            bw_pct = bw_floor / t_us * 100
            k_iters = (rank + BK - 1) // BK
            marker = " ←" if t_us < best_t else ""
            if t_us < best_t:
                best_t = t_us
                best_cfg = cfg
            print(
                f"  {BS:>7}  {BN:>7}  {BK:>7}  {nw:>5}  {ns:>3}  {t_us:>7.1f}µ  {bw_pct:>5.1f}%  {k_iters:>8}{marker}"
            )
        except Exception as e:
            print(f"  {BS:>7}  {BN:>7}  {BK:>7}  {nw:>5}  {ns:>3}  FAILED: {e}")

    print(
        f"\n  Best: BLOCK_S={best_cfg[0]} BLOCK_N={best_cfg[1]} BLOCK_K={best_cfg[2]} warps={best_cfg[3]} stages={best_cfg[4]} → {best_t:.1f}µs"
    )
    print(
        f"  Current autotune: {bench_cfg(lambda: run_probe(x, w, o.clone(), bi, 16, 64, 16, 8, 3))*1e6:.1f}µs"
    )


if __name__ == "__main__":
    for n_segs in (16, 32, 64):
        sweep(n_segs=n_segs, rank=64, N=4096, label="o_proj rank=64")
    sweep(n_segs=32, rank=128, N=4096, label="o_proj rank=128")
    sweep(n_segs=32, rank=16, N=4096, label="o_proj rank=16")
