"""Benchmark: our shrink/expand kernels vs sglang csgmv variants.

Inlines sglang kernels (Apache-2.0) so sglang doesn't need to be
installed.  All kernels are autotuned with the same config space.

Shrink (LoRA-A):  x (s, K) @ W^T (K, N)  →  out (s, N)
  N = stack_num * rank  (small),  K = in_dim  (large, 4096+)
  Key diff in chunked_sgmv_shrink: K and N are constexpr
  → K-loop trip count is compile-time constant.

Expand (LoRA-B):  x (s, num_slices*R) @ W (R, out_dim)  →  out (s, out_dim)
  R = rank  (small),  out_dim large
  Key diff in chunked_sgmv_expand: strides and MAX_RANK are constexpr.

When rank == max_rank the x layouts are identical between ours and sglang.

Usage:
    python bench_chunked_sgmv.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl

# ── make the local kernel package importable ──────────────────────────────────
sys.path.insert(
    0,
    str(Path(__file__).parent / "tokenspeed-kernel" / "python"),
)

from tokenspeed_kernel.ops.lora.triton.lora_expand import lora_expand_fwd
from tokenspeed_kernel.ops.lora.triton.lora_gate_up_expand import (
    lora_gate_up_expand_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_qkv_expand import lora_qkv_expand_fwd
from tokenspeed_kernel.ops.lora.triton.lora_shrink import lora_shrink_fwd

# ── minimal batch-info dataclass ──────────────────────────────────────────────


@dataclass
class BatchInfo:
    bs: int
    max_len: int
    seg_lens: torch.Tensor
    seg_indptr: torch.Tensor
    weight_indices: torch.Tensor
    lora_ranks: torch.Tensor
    scalings: torch.Tensor
    permutation: torch.Tensor | None = None
    # sglang compat
    num_segments: int = 0
    use_cuda_graph: bool = False


def make_batch(
    s_per_seg: int, n_segs: int, rank: int, with_perm: bool = False
) -> BatchInfo:
    dev = "cuda"
    seg_lens = torch.full((n_segs,), s_per_seg, dtype=torch.int32, device=dev)
    seg_indptr = torch.arange(n_segs + 1, dtype=torch.int32, device=dev) * s_per_seg
    # all segs route to slot 1 (real adapter), slot 0 = no-adapter sentinel
    weight_indices = torch.ones(n_segs, dtype=torch.int32, device=dev)
    lora_ranks = torch.tensor([0, rank], dtype=torch.int32, device=dev)
    scalings = torch.tensor([0.0, 1.0], dtype=torch.float32, device=dev)
    perm = None
    if with_perm:
        s_total = n_segs * s_per_seg
        perm = torch.arange(s_total, dtype=torch.int64, device=dev)
    return BatchInfo(
        bs=n_segs,
        max_len=s_per_seg,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        permutation=perm,
        num_segments=n_segs,
    )


# ── inlined sglang chunked_sgmv_expand (Apache-2.0) ──────────────────────────
# Source: github.com/sgl-project/sglang  python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py
# Local change: replaced sglang imports with triton directly; added @triton.autotune.


@triton.jit(do_not_specialize=["num_segs", "output_stride_0", "output_stride_1"])
def _chunked_lora_expand_kernel(
    x,
    weights,
    output,
    output_stride_0,
    output_stride_1,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    scalings,
    slice_offsets,
    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK
    x_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(axis=2)
    if pid_s >= num_segs:
        return

    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    slice_id = tl.program_id(axis=1)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)
    scaling = tl.load(scalings + w_index)

    cur_rank = tl.minimum(MAX_RANK, cur_rank)

    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    pid_n = tl.program_id(axis=0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (
        x
        + slice_id * cur_rank * x_stride_1
        + (s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK
    w_stride_2: tl.constexpr = 1
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(cur_rank, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < cur_rank - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < cur_rank - k * BLOCK_K)
            & (n_offset[None, :] < slice_end),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)

    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_expand_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: BatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: torch.Tensor | None,
) -> torch.Tensor:
    assert x.is_contiguous() and weights.is_contiguous()
    M = x.shape[0]
    OUT_DIM = weights.shape[1]
    MAX_RANK = weights.shape[2]
    num_slices = len(slice_offsets) - 1
    assert x.shape[1] == num_slices * MAX_RANK

    num_segs = batch_info.num_segments

    BM, BN, BK = 16, 64, 16
    grid = (triton.cdiv(max_slice_size, BN), num_slices, batch_info.bs)
    output = (
        torch.zeros((M, OUT_DIM), device=x.device, dtype=x.dtype)
        if base_output is None
        else base_output
    )
    _chunked_lora_expand_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=num_segs,
        scalings=batch_info.scalings,
        slice_offsets=slice_offsets,
        NUM_SLICES=num_slices,
        OUTPUT_DIM=OUT_DIM,
        MAX_RANK=MAX_RANK,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=2,
    )
    return output


# ── inlined sglang sgemm_lora_a (Apache-2.0) ─────────────────────────────────
# Source: github.com/sgl-project/sglang  python/sglang/srt/lora/triton_ops/sgemm_lora_a.py
# Local change: replaced sglang imports; added @triton.autotune (original uses fixed sizes).


@triton.jit
def _sgemm_lora_a_kernel(
    x,
    weights,
    output,
    N,
    K,
    stack_num,
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
    sorted_token_ids,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    N = tl.minimum(N, rank * stack_num)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    if SORTED_BY_ADAPTER:
        s_physical = tl.load(
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len, other=0
        )
    else:
        s_physical = seg_start + s_offset
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_a_fwd(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    N = weights.shape[-2]
    assert x.is_contiguous() and weights.is_contiguous()
    max_len = batch_info.max_len
    BS, BN, BK = 16, 32, 128
    grid = (triton.cdiv(max_len, BS) * triton.cdiv(N, BN), batch_info.bs)
    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    sorted_by_adapter = batch_info.permutation is not None
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        N,
        K,
        stack_num,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S=BS,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )
    return output


# ── inlined sglang chunked_sgmv_shrink (Apache-2.0) ──────────────────────────
# Source: github.com/sgl-project/sglang  python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py
# Local change: replaced sglang imports; added @triton.autotune.
# Key structural diff vs sgemm_lora_a: K, N, and all strides are constexpr.


@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_kernel(
    x,
    weights,
    output,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K
    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1
    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return
    pid_n = tl.program_id(0)
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    cur_n = tl.minimum(N, rank * NUM_SLICES)

    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_shrink_fwd(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[-2]  # num_slices * rank
    assert x.is_contiguous() and weights.is_contiguous()
    num_segs = batch_info.num_segments
    BM, BN, BK = 16, 32, 128
    grid = (triton.cdiv(N, BN), batch_info.bs)
    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _chunked_lora_shrink_kernel[grid](
        x,
        weights,
        output,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        num_segs,
        N=N,
        K=K,
        NUM_SLICES=num_slices,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )
    return output


# ── inlined sglang sgemm_lora_b (Apache-2.0) ─────────────────────────────────
# Source: github.com/sgl-project/sglang  python/sglang/srt/lora/triton_ops/sgemm_lora_b.py
# Structurally identical to our lora_expand; only difference is fixed BLOCK_N=256.


@triton.jit
def _sgemm_lora_b_kernel(
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
    sorted_token_ids,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scalings,
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
    K = tl.minimum(K, rank)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    if SORTED_BY_ADAPTER:
        s_physical = tl.load(
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len, other=0
        )
    else:
        s_physical = seg_start + s_offset
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )
    n_mask = n_offset[None, :] < N
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs, mask=(k_offset[:, None] < K - k * BLOCK_K) & n_mask, other=0.0
        )
        partial_sum += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_b_fwd(x, weights, batch_info, base_output=None):
    S, R = x.shape
    N = weights.shape[-2]
    assert x.is_contiguous() and weights.is_contiguous()
    # Original sglang fixed configs: BLOCK_S=16, BLOCK_N=256, BLOCK_K=16
    BS, BN, BK = 16, 256, 16
    max_len = batch_info.max_len
    grid = (triton.cdiv(max_len, BS) * triton.cdiv(N, BN), batch_info.bs)
    output = (
        torch.zeros((S, N), device=x.device, dtype=x.dtype)
        if base_output is None
        else base_output
    )
    sorted_by_adapter = batch_info.permutation is not None
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        N,
        R,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S=BS,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=2,
        scalings=batch_info.scalings,
    )
    return output


# ── benchmark helpers ─────────────────────────────────────────────────────────


def bench(fn, label: str, warmup: int = 25, rep: int = 100) -> float:
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    print(f"  {label:<42s}  {ms*1000:7.1f} µs")
    return ms


def run_shrink_scenario(
    label: str,
    s_per_seg: int,
    n_segs: int,
    rank: int,
    hidden: int,
    intermediate_per_tp: int,
) -> None:
    dev, dt = "cuda", torch.bfloat16
    s = s_per_seg * n_segs
    bi_ours = make_batch(s_per_seg, n_segs, rank, with_perm=False)
    bi_sglang = make_batch(s_per_seg, n_segs, rank, with_perm=True)

    print(f"\n{'='*60}")
    print(f"  SHRINK {label}")
    print(f"  s_per_seg={s_per_seg}  n_segs={n_segs}  rank={rank}  s_total={s}")
    print(f"{'='*60}")

    for stack_num, in_dim, tag in [
        (3, hidden, "QKV shrink      in=hidden  stack=3"),
        (2, hidden, "gate/up shrink  in=hidden  stack=2"),
        (1, hidden, "o/down shrink   in=hidden  stack=1"),
        (1, intermediate_per_tp, "down shrink     in=inter   stack=1"),
    ]:
        N = stack_num * rank
        x = torch.randn((s, in_dim), device=dev, dtype=dt)
        w = torch.randn((2, N, in_dim), device=dev, dtype=dt)
        print(f"\n[{tag}]  K={in_dim}")
        bench(
            lambda x=x, w=w: lora_shrink_fwd(x, w, bi_ours, stack_num=stack_num),
            "ours lora_shrink_fwd",
        )
        bench(
            lambda x=x, w=w: sgemm_lora_a_fwd(x, w, bi_sglang, stack_num=stack_num),
            "sglang sgemm_lora_a (autotuned)",
        )
        bench(
            lambda x=x, w=w: chunked_sgmv_shrink_fwd(
                x, w, bi_sglang, num_slices=stack_num
            ),
            "sglang chunked_sgmv_shrink",
        )


def run_scenario(
    label: str,
    s_per_seg: int,
    n_segs: int,
    rank: int,
    hidden: int,
    intermediate_per_tp: int,
    q_per_tp: int,
    kv_per_tp: int,
) -> None:
    dev, dt = "cuda", torch.bfloat16
    max_rank = rank  # rank == max_rank so x layouts are identical

    s = s_per_seg * n_segs
    bi_ours = make_batch(s_per_seg, n_segs, rank, with_perm=False)
    bi_sglang = make_batch(
        s_per_seg, n_segs, rank, with_perm=True
    )  # sglang always needs perm

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  s_per_seg={s_per_seg}  n_segs={n_segs}  rank={rank}  s_total={s}")
    print(f"{'='*60}")

    # ── plain expand (o_proj / down_proj): 1 slice, out_dim=hidden ──
    print("\n[plain expand]  out_dim=hidden")
    x1 = torch.randn((s, max_rank), device=dev, dtype=dt)
    w1 = torch.randn((2, hidden, max_rank), device=dev, dtype=dt)
    o1 = torch.zeros((s, hidden), device=dev, dtype=dt)
    so1 = torch.tensor([0, hidden], dtype=torch.int32, device=dev)

    bench(
        lambda: lora_expand_fwd(x1, w1, bi_ours, base_output=o1.clone()),
        "ours lora_expand_fwd",
    )
    bench(
        lambda: sgemm_lora_b_fwd(x1, w1, bi_sglang, base_output=o1.clone()),
        "sglang sgemm_lora_b (BN=256)",
    )
    bench(
        lambda: chunked_sgmv_expand_fwd(x1, w1, bi_sglang, so1, hidden, o1.clone()),
        "sglang chunked_sgmv (1 slice)",
    )

    # ── QKV expand: 3 slices ──
    qkv_out = q_per_tp + 2 * kv_per_tp
    max_qkv = max(q_per_tp, kv_per_tp)
    x3 = torch.randn((s, 3 * max_rank), device=dev, dtype=dt)
    w3 = torch.randn((2, qkv_out, max_rank), device=dev, dtype=dt)
    o3 = torch.zeros((s, qkv_out), device=dev, dtype=dt)
    off3 = torch.tensor(
        [0, q_per_tp, q_per_tp + kv_per_tp, q_per_tp + 2 * kv_per_tp],
        dtype=torch.int32,
        device=dev,
    )

    print(f"\n[QKV expand]  q={q_per_tp}  kv={kv_per_tp}")
    bench(
        lambda: lora_qkv_expand_fwd(
            x3, w3, bi_ours, off3, max_qkv, base_output=o3.clone()
        ),
        "ours lora_qkv_expand_fwd",
    )
    bench(
        lambda: chunked_sgmv_expand_fwd(x3, w3, bi_sglang, off3, max_qkv, o3.clone()),
        "sglang chunked_sgmv (3 slices)",
    )

    # ── gate/up expand: 2 slices ──
    x2 = torch.randn((s, 2 * max_rank), device=dev, dtype=dt)
    w2 = torch.randn((2, 2 * intermediate_per_tp, max_rank), device=dev, dtype=dt)
    o2 = torch.zeros((s, 2 * intermediate_per_tp), device=dev, dtype=dt)
    so2 = torch.tensor(
        [0, intermediate_per_tp, 2 * intermediate_per_tp], dtype=torch.int32, device=dev
    )

    print(f"\n[gate/up expand]  intermediate_per_tp={intermediate_per_tp}")
    bench(
        lambda: lora_gate_up_expand_fwd(
            x2, w2, bi_ours, intermediate_per_tp, base_output=o2.clone()
        ),
        "ours lora_gate_up_expand_fwd",
    )
    bench(
        lambda: chunked_sgmv_expand_fwd(
            x2, w2, bi_sglang, so2, intermediate_per_tp, o2.clone()
        ),
        "sglang chunked_sgmv (2 slices)",
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Qwen3-8B-like shapes at TP=2
    HIDDEN = 4096
    INTERMEDIATE = 12288
    INTER_PER_TP = INTERMEDIATE // 2  # 6144
    Q_PER_TP = 2048
    KV_PER_TP = 512
    RANK = 64

    # ── 1. Sequence-length sweep (fixed n_segs=32 decode, n_segs=4 prefill) ──
    for s_per_seg, n_segs, tag in [
        (1, 32, "DECODE      s=1    n_segs=32"),
        (1, 64, "DECODE      s=1    n_segs=64"),
        (128, 4, "PREFILL     s=128  n_segs=4"),
        (512, 2, "PREFILL     s=512  n_segs=2"),
    ]:
        run_scenario(
            tag,
            s_per_seg=s_per_seg,
            n_segs=n_segs,
            rank=RANK,
            hidden=HIDDEN,
            intermediate_per_tp=INTER_PER_TP,
            q_per_tp=Q_PER_TP,
            kv_per_tp=KV_PER_TP,
        )

    # ── 2. Adapter-count sweep (decode, s_per_seg=1, vary n_segs) ──
    print(f"\n\n{'#'*60}")
    print(f"  ADAPTER COUNT SWEEP  (decode s=1, rank={RANK})")
    print(f"{'#'*60}")
    dev, dt = "cuda", torch.bfloat16
    qkv_out = Q_PER_TP + 2 * KV_PER_TP
    max_qkv = max(Q_PER_TP, KV_PER_TP)
    off3 = torch.tensor(
        [0, Q_PER_TP, Q_PER_TP + KV_PER_TP, Q_PER_TP + 2 * KV_PER_TP],
        dtype=torch.int32,
        device=dev,
    )
    so1 = torch.tensor([0, HIDDEN], dtype=torch.int32, device=dev)

    print(
        f"\n{'n_segs':>8}  {'ours expand':>14}  {'sgemm_b BN256':>14}  {'csgmv 1sl':>12}  {'ours qkv':>12}  {'csgmv 3sl':>12}"
    )
    print("-" * 82)
    for n_segs in (1, 2, 4, 8, 16, 32, 64, 128):
        s = n_segs
        bi_o = make_batch(1, n_segs, RANK, with_perm=False)
        bi_s = make_batch(1, n_segs, RANK, with_perm=True)
        x1 = torch.randn((s, RANK), device=dev, dtype=dt)
        w1 = torch.randn((2, HIDDEN, RANK), device=dev, dtype=dt)
        o1 = torch.zeros((s, HIDDEN), device=dev, dtype=dt)
        x3 = torch.randn((s, 3 * RANK), device=dev, dtype=dt)
        w3 = torch.randn((2, qkv_out, RANK), device=dev, dtype=dt)
        o3 = torch.zeros((s, qkv_out), device=dev, dtype=dt)

        def t(fn):
            return triton.testing.do_bench(fn, warmup=25, rep=200) * 1000

        t_ours_exp = t(lambda: lora_expand_fwd(x1, w1, bi_o, base_output=o1.clone()))
        t_sgemm_b = t(lambda: sgemm_lora_b_fwd(x1, w1, bi_s, base_output=o1.clone()))
        t_csgmv_1 = t(
            lambda: chunked_sgmv_expand_fwd(x1, w1, bi_s, so1, HIDDEN, o1.clone())
        )
        t_ours_qkv = t(
            lambda: lora_qkv_expand_fwd(
                x3, w3, bi_o, off3, max_qkv, base_output=o3.clone()
            )
        )
        t_csgmv_3 = t(
            lambda: chunked_sgmv_expand_fwd(x3, w3, bi_s, off3, max_qkv, o3.clone())
        )

        print(
            f"{n_segs:>8}  {t_ours_exp:>13.1f}µ  {t_sgemm_b:>13.1f}µ  {t_csgmv_1:>11.1f}µ  {t_ours_qkv:>11.1f}µ  {t_csgmv_3:>11.1f}µ"
        )
