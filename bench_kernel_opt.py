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
from tokenspeed_kernel.ops.lora.triton.lora_expand_decode import lora_expand_decode_fwd
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
    sort_order: torch.Tensor | None = None
    group_slots: torch.Tensor | None = None
    group_starts: torch.Tensor | None = None
    group_sizes: torch.Tensor | None = None
    num_groups: int = 0


def make_mixed_batch(
    n_segs: int,
    n_unique_adapters: int,
    rank: int,
    device: str = "cuda",
) -> BatchInfo:
    """n_segs decode segments, round-robin across n_unique_adapters adapters."""
    slots_list = [(i % n_unique_adapters) + 1 for i in range(n_segs)]
    slots = torch.tensor(slots_list, dtype=torch.int32, device=device)

    seg_lens = torch.ones(n_segs, dtype=torch.int32, device=device)
    seg_indptr = torch.arange(n_segs + 1, dtype=torch.int32, device=device)
    n_slots = n_unique_adapters + 1
    lora_ranks = torch.zeros(n_slots, dtype=torch.int32, device=device)
    lora_ranks[1:] = rank
    scalings = torch.ones(n_slots, dtype=torch.float32, device=device)
    scalings[0] = 0.0

    # Build group metadata (same logic as prepare_loras)
    sort_order_cpu = sorted(range(n_segs), key=lambda i: slots_list[i])
    groups: list[list[int]] = []
    for pos, orig in enumerate(sort_order_cpu):
        slot = slots_list[orig]
        if not groups or groups[-1][0] != slot:
            groups.append([slot, pos, 1])
        else:
            groups[-1][2] += 1
    ng = len(groups)
    sort_order_gpu = torch.tensor(sort_order_cpu, dtype=torch.int64, device=device)
    group_slots_gpu = torch.tensor(
        [g[0] for g in groups], dtype=torch.int32, device=device
    )
    group_starts_gpu = torch.tensor(
        [g[1] for g in groups], dtype=torch.int32, device=device
    )
    group_sizes_gpu = torch.tensor(
        [g[2] for g in groups], dtype=torch.int32, device=device
    )

    return BatchInfo(
        bs=n_segs,
        max_len=1,
        num_segments=n_segs,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        weight_indices=slots,
        lora_ranks=lora_ranks,
        scalings=scalings,
        sort_order=sort_order_gpu,
        group_slots=group_slots_gpu,
        group_starts=group_starts_gpu,
        group_sizes=group_sizes_gpu,
        num_groups=ng,
    )


def bench(fn, warmup=25, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep) * 1000


def run(n_segs: int, n_unique: int, rank: int, hidden: int) -> None:
    dev, dt = "cuda", torch.bfloat16
    n_slots = n_unique + 1
    s = n_segs

    bi = make_mixed_batch(n_segs, n_unique, rank, device=dev)

    x_ex = torch.randn((s, rank), device=dev, dtype=dt)
    w_ex = torch.randn((n_slots, hidden, rank), device=dev, dtype=dt)
    o_ex = torch.zeros((s, hidden), device=dev, dtype=dt)

    t_base = bench(lambda: lora_expand_fwd(x_ex, w_ex, bi, base_output=o_ex.clone()))
    t_grouped = bench(
        lambda: lora_expand_decode_fwd(x_ex, w_ex, bi, base_output=o_ex.clone())
    )

    print(
        f"n_segs={n_segs:>3}  n_unique={n_unique:>2}  rank={rank:>3}  hidden={hidden:>5}  |"
        f"  base={t_base:>6.1f}µ  grouped={t_grouped:>6.1f}µ  {t_base/t_grouped:>5.2f}x"
    )


if __name__ == "__main__":
    # Qwen3-8B TP=2
    HIDDEN, RANK = 4096, 64

    print(
        f"\n{'n_segs':>7}  {'n_unique':>9}  {'rank':>5}  {'hidden':>7}  |  {'base':>8}  {'grouped':>9}  speedup"
    )
    print("-" * 75)
    for n_unique in (1, 2, 4, 8, 16, 32):
        run(n_segs=32, n_unique=n_unique, rank=RANK, hidden=HIDDEN)
    print()
    for n_segs in (8, 16, 32, 64, 128):
        run(n_segs=n_segs, n_unique=4, rank=RANK, hidden=HIDDEN)
    print()
    for rank in (16, 32, 64, 128):
        run(n_segs=32, n_unique=4, rank=rank, hidden=HIDDEN)
