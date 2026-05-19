# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Decode-optimised LoRA-B expand: groups same-adapter segments for tensor-core efficiency.

Problem with the standard decode expand kernel
----------------------------------------------
For decode batches (``s_per_seg=1``), the kernel grid is
``(cdiv(N, BLOCK_N), bs)`` — one CTA per ``(N-tile, segment)``.  With
``BLOCK_S=16`` but only 1 valid token per CTA, tensor cores run at 1/16
throughput: the ``(16, BLOCK_K) @ (BLOCK_K, BLOCK_N)`` dot product uses
only its first row.  At ``bs=32`` and ``N=4096``, this is 2048 CTAs each
doing 1/16 useful work.

Solution: grouped expand
------------------------
Sort segments by adapter slot (done on CPU in ``prepare_loras`` — free),
then build adapter groups.  The grouped kernel has grid
``(cdiv(N, BLOCK_N), num_unique_adapters)``.  Each CTA processes ALL tokens
in one adapter group in ``BLOCK_S``-wide GEMM tiles.  With ``BLOCK_S=16``
and an adapter group of 16 tokens, the dot product is fully packed.

For ``bs=32`` and 4 unique adapters (8 tokens each):
* Old: 2048 CTAs, each 1/16 efficient = 128 effective CTAs of work
* New:  256 CTAs (64 × 4), each 8/16 efficient  = 128 effective CTAs
* Grid launch cost: 8× fewer CTAs → measurable end-to-end improvement

For ``bs=32`` all same adapter:
* Old: 2048 CTAs, each 1/16 efficient
* New:  128 CTAs (64 × 1), fully packed
* 16× fewer CTAs, full tensor-core utilisation

The x gather and output scatter (small copies for decode) take ~100ns each
and are negligible vs the kernel improvement.

Adapter group metadata (``sort_order``, ``group_slots``, ``group_starts``,
``group_sizes``, ``num_groups``) is pre-computed in ``prepare_loras`` and
stored in :class:`LoraBatchInfo` so no GPU-CPU sync is needed at forward time.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

_DECODE_EXPAND_CONFIGS = [
    triton.Config(
        {"BLOCK_S": s, "BLOCK_N": n, "BLOCK_K": k},
        num_warps=w,
        num_stages=stages,
        maxnreg=mr,
    )
    for s in (16, 32)
    for n in (32, 64, 128)
    for k in (16, 32)
    for w in (4, 8)
    for stages in (1, 2, 3)
    for mr in (None, 128, 160)
]


@triton.autotune(
    configs=_DECODE_EXPAND_CONFIGS,
    key=["N", "MAX_RANK"],
    restore_value=["out_sorted"],
)
@triton.jit
def _lora_expand_decode_kernel(
    x_sorted,  # (bs, MAX_RANK) contiguous — sorted by adapter group
    weights,  # (n_slots, N, MAX_RANK) contiguous
    out_sorted,  # (bs, N) contiguous — add-into (pre-filled with base_output)
    group_slots,  # (num_groups,) int32
    group_starts,  # (num_groups,) int32 — first row in x_sorted for this group
    group_sizes,  # (num_groups,) int32 — number of tokens in this group
    scalings,  # (n_slots,) float32
    lora_ranks,  # (n_slots,) int32
    N: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Strides are constexpr because x_sorted and out_sorted are freshly
    # allocated contiguous tensors with known shapes.
    x_stride_0: tl.constexpr = MAX_RANK
    x_stride_1: tl.constexpr = 1
    w_stride_0: tl.constexpr = N * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK  # row stride of (N, MAX_RANK) slice
    w_stride_2: tl.constexpr = 1
    out_stride_0: tl.constexpr = N
    out_stride_1: tl.constexpr = 1

    group_id = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    w_index = tl.load(group_slots + group_id)
    g_size = tl.load(group_sizes + group_id)
    if g_size == 0:
        return
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    g_start = tl.load(group_starts + group_id)
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(MAX_RANK, rank)

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.max_contiguous(tl.arange(0, BLOCK_K), BLOCK_K)
    n_mask = n_offset[None, :] < N

    # Process the group in BLOCK_S-wide GEMM tiles.  When the group size is a
    # multiple of BLOCK_S (e.g. 16 tokens with BLOCK_S=16) every tile is
    # fully packed and tensor cores run at 100% efficiency.
    for tile_s in range(0, tl.cdiv(g_size, BLOCK_S)):
        s_offset = tl.arange(0, BLOCK_S)
        abs_s = g_start + tile_s * BLOCK_S + s_offset
        s_mask = (s_offset < g_size - tile_s * BLOCK_S)[:, None]

        x_ptrs = x_sorted + abs_s[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
        w_ptrs = (weights + w_index * w_stride_0) + (
            k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
        )

        partial = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
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
                eviction_policy="evict_last",  # shared across all tiles of this group
            )
            partial += tl.dot(x_tile, w_tile)
            x_ptrs += BLOCK_K * x_stride_1
            w_ptrs += BLOCK_K * w_stride_2

        partial *= scaling
        partial = partial.to(x_sorted.dtype.element_ty)

        out_ptrs = (
            out_sorted
            + abs_s[:, None] * out_stride_0
            + n_offset[None, :] * out_stride_1
        )
        out_mask = s_mask & n_mask
        partial += tl.load(out_ptrs, mask=out_mask, other=0.0)
        tl.store(out_ptrs, partial, mask=out_mask)


def lora_expand_decode_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-optimised expand using adapter-grouped GEMM tiles.

    Requires ``batch_info`` to have pre-computed group metadata fields
    (``sort_order``, ``group_slots``, ``group_starts``, ``group_sizes``,
    ``num_groups``) populated by :meth:`LoraManager.prepare_loras`.

    Input / output shapes are identical to :func:`lora_expand_fwd`.
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()

    bs = batch_info.bs
    S, R = x.shape
    N = weights.shape[-2]
    dev, dt = x.device, x.dtype

    sort_order = batch_info.sort_order[:bs]
    num_groups = batch_info.num_groups

    # Gather x (and base_output when supplied) into adapter-sorted order.
    x_sorted = x[sort_order].contiguous()

    if base_output is None:
        out_sorted = torch.zeros((S, N), device=dev, dtype=dt)
    else:
        out_sorted = base_output[sort_order].clone()

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), num_groups)

    _lora_expand_decode_kernel[grid](
        x_sorted,
        weights,
        out_sorted,
        batch_info.group_slots[:num_groups],
        batch_info.group_starts[:num_groups],
        batch_info.group_sizes[:num_groups],
        batch_info.scalings,
        batch_info.lora_ranks,
        N=N,
        MAX_RANK=R,
    )

    # Scatter sorted output back to original token order.
    if base_output is None:
        output = torch.empty((S, N), device=dev, dtype=dt)
    else:
        output = base_output
    output[sort_order] = out_sorted
    return output


load_kernel_cache(_lora_expand_decode_kernel)
