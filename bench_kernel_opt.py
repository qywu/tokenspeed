"""Before/after benchmark for kernel micro-optimisations + sort-by-adapter.

Tests decode shrink and expand with mixed adapters — the scenario where
sort-by-adapter actually helps (adjacent CTAs share the same weight tile).

Usage:
    python bench_kernel_opt.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent / "tokenspeed-kernel" / "python"))

from tokenspeed_kernel.ops.lora.triton.lora_expand import lora_expand_fwd
from tokenspeed_kernel.ops.lora.triton.lora_shrink import lora_shrink_fwd


@dataclass
class BatchInfo:
    bs: int
    max_len: int
    num_segments: int
    seg_lens: torch.Tensor
    seg_indptr: torch.Tensor
    weight_indices: torch.Tensor
    lora_ranks: torch.Tensor
    scalings: torch.Tensor
    permutation: torch.Tensor | None = None


def make_mixed_batch(
    n_segs: int,
    n_unique_adapters: int,
    rank: int,
    sorted_by_adapter: bool,
    device: str = "cuda",
) -> BatchInfo:
    """n_segs decode segments, round-robin across n_unique_adapters adapters."""
    # slots: [1, 2, ..., n_unique, 1, 2, ...] cycling
    slots = torch.tensor(
        [(i % n_unique_adapters) + 1 for i in range(n_segs)], dtype=torch.int32, device=device
    )
    if sorted_by_adapter:
        sort_order = torch.argsort(slots, stable=True)
        slots = slots[sort_order]
        perm = sort_order.to(torch.int64)
    else:
        perm = None

    seg_lens   = torch.ones(n_segs, dtype=torch.int32, device=device)
    seg_indptr = torch.arange(n_segs + 1, dtype=torch.int32, device=device)
    n_slots    = n_unique_adapters + 1
    lora_ranks = torch.zeros(n_slots, dtype=torch.int32, device=device)
    lora_ranks[1:] = rank
    scalings   = torch.ones(n_slots, dtype=torch.float32, device=device)
    scalings[0] = 0.0

    return BatchInfo(
        bs=n_segs, max_len=1, num_segments=n_segs,
        seg_lens=seg_lens, seg_indptr=seg_indptr,
        weight_indices=slots, lora_ranks=lora_ranks,
        scalings=scalings, permutation=perm,
    )


def bench(fn, warmup=25, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep) * 1000


def run(n_segs: int, n_unique: int, rank: int, hidden: int) -> None:
    dev, dt = "cuda", torch.bfloat16
    n_slots = n_unique + 1
    s = n_segs

    bi_unsorted = make_mixed_batch(n_segs, n_unique, rank, sorted_by_adapter=False)
    bi_sorted   = make_mixed_batch(n_segs, n_unique, rank, sorted_by_adapter=True)

    # Shrink: x (s, hidden) → lora_a (s, rank)
    x_sh = torch.randn((s, hidden),             device=dev, dtype=dt)
    w_sh = torch.randn((n_slots, rank, hidden),  device=dev, dtype=dt)

    # Expand: lora_a (s, rank) → output (s, hidden) fused-add
    x_ex = torch.randn((s, rank),               device=dev, dtype=dt)
    w_ex = torch.randn((n_slots, hidden, rank),  device=dev, dtype=dt)
    o_ex = torch.zeros((s, hidden),             device=dev, dtype=dt)

    print(f"\nn_segs={n_segs}  n_unique={n_unique}  rank={rank}  hidden={hidden}")
    print(f"  {'kernel':<28}  {'unsorted':>10}  {'sorted':>10}  {'speedup':>8}")
    print(f"  {'-'*62}")

    for label, fn_u, fn_s in [
        ("shrink",
         lambda: lora_shrink_fwd(x_sh, w_sh, bi_unsorted, stack_num=1),
         lambda: lora_shrink_fwd(x_sh, w_sh, bi_sorted,   stack_num=1)),
        ("expand (o_proj)",
         lambda: lora_expand_fwd(x_ex, w_ex, bi_unsorted, base_output=o_ex.clone()),
         lambda: lora_expand_fwd(x_ex, w_ex, bi_sorted,   base_output=o_ex.clone())),
    ]:
        tu = bench(fn_u)
        ts = bench(fn_s)
        print(f"  {label:<28}  {tu:>9.1f}µ  {ts:>9.1f}µ  {tu/ts:>7.2f}x")


if __name__ == "__main__":
    # Qwen3-8B TP=2, rank=64
    HIDDEN, RANK = 4096, 64

    for n_unique in (2, 4, 8, 16):
        run(n_segs=32,  n_unique=n_unique, rank=RANK, hidden=HIDDEN)
    for n_segs in (16, 32, 64, 128):
        run(n_segs=n_segs, n_unique=4,    rank=RANK, hidden=HIDDEN)
