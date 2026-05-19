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

"""LoRA adapter weight manager (segment-grouped Triton path).

Adapted from sglang/Punica's S-LoRA design.

Memory layout
-------------
For each layer the manager owns:

* ``qkv_A_buffers[layer]``: ``(n_slots, 3 * max_rank, hidden)`` — fused
  q_proj/k_proj/v_proj A matrices, stack-major (q first, then k, then v).
* ``qkv_B_buffers[layer]``: ``(n_slots, q_per_tp + 2 * kv_per_tp, max_rank)``
  — fused output-side, ``[q_per_tp | kv_per_tp | kv_per_tp]`` along dim 1.
* ``o_A_buffers[layer]``:   ``(n_slots, max_rank, in_per_tp)`` — row-parallel
  A, sharded along input dim.
* ``o_B_buffers[layer]``:   ``(n_slots, hidden, max_rank)`` — full B.

Slot 0 is the no-adapter sentinel (rank 0, scaling 0).  The Triton
kernels short-circuit on slot 0, so the captured CUDA graph stays a no-op
when no request uses an adapter.

Tensor parallelism
------------------
* QKV is column-parallel: A is full, B is sharded along output dim
  (``q_per_tp + 2 * kv_per_tp``).  No collective inside the LoRA path.
* O is row-parallel: A is sharded along input dim, B is full.  The host
  module (qwen3 ``o_proj``) runs with ``reduce_results=False`` and has its
  partial sum all-reduced downstream by ``post_attention_layernorm``; the
  LoRA delta rides that same reduction (full ``B @ lora_a`` is added to the
  partial output and the downstream reduce sums it ``tp_size`` times — see
  ``apply_o_lora`` for the resulting numerical caveat).
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch
from tokenspeed_kernel.ops.lora.cutedsl import (
    lora_expand_batched_slots_cutedsl_fwd,
    lora_expand_single_slot_cutedsl_fwd,
    lora_gate_up_batched_slots_cutedsl_fwd,
    lora_gate_up_single_slot_cutedsl_fwd,
    lora_qkv_single_slot_cutedsl_fwd,
)
from tokenspeed_kernel.ops.lora.triton import (
    lora_expand_fwd,
    lora_expand_grouped_v2_fwd,
    lora_expand_prefill_fwd,
    lora_gate_up_expand_fwd,
    lora_qkv_expand_fwd,
    lora_shrink_fwd,
    lora_shrink_prefill_fwd,
)

from tokenspeed.runtime.lora.adapter_io import (
    PEFT_MODULES,
    read_adapter_scaling,
    resolve_adapter_weight_path,
)
from tokenspeed.runtime.lora.lora_batch import LoraBatchInfo, build_decode_lora_groups
from tokenspeed.runtime.lora.lora_buffers import LoraWeightBuffers
from tokenspeed.runtime.lora.lora_cache import LoraCpuCache
from tokenspeed.runtime.lora.moe_lora import MoeLoraBuffers, MoeLoraContext
from tokenspeed.runtime.utils import get_colorful_logger

# Segments longer than this use the prefill (chunked-SGMV) expand kernel,
# which specialises strides and loop counts at compile time.  Shorter
# segments (decode) use the decode-tuned kernels.  Threshold chosen from
# benchmarks: chunked-SGMV wins above ~32 tokens/segment at rank ≥ 64.
_CHUNKED_THRESHOLD = 32

# The CuTeDSL single-slot expand path lowers LoRA-B expand to dense GEMM-adds.
# Thresholds are based on H100 full-path measurements, including the Triton
# shrink that still feeds the CuTeDSL expand.
_CUTEDSL_SINGLE_SLOT_DECODE_MIN_OUT_DIM = 3072
_CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM = 1024
_CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM = 2048
_CUTEDSL_SINGLE_SLOT_LOW_OUT_MIN_TOKENS = 256
_CUTEDSL_SINGLE_SLOT_LOW_OUT_DECODE_MIN_TOKENS = 64
_CUTEDSL_MULTI_SLOT_MIN_OUT_DIM = 3072
_CUTEDSL_MULTI_SLOT_LOW_OUT_DIM = 2048
_CUTEDSL_SINGLE_SLOT_SMALL_PREFILL_MIN_TOKENS = 128
_CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM = 1024
_CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_MIN_TOKENS = 256
_CUTEDSL_SINGLE_SLOT_GATE_UP_LOW_OUT_MIN_TOKENS = 512
_CUTEDSL_GATE_UP_SMALL_OUT_DIM = 4096
_CUTEDSL_GATE_UP_MEDIUM_OUT_DIM = 8192
_CUTEDSL_GATE_UP_LARGE_OUT_DIM = 12288
_TRITON_GROUPED_DECODE_MIN_GROUP_SIZE = 32

logger = get_colorful_logger(__name__)


# ── Manager ─────────────────────────────────────────────────────────────────


def _use_cutedsl_single_slot_expand(
    bi: LoraBatchInfo,
    total_tokens: int,
    out_dim: int,
    lora_rank: int,
    input_dim: int = 4096,
) -> bool:
    """Return whether the single-slot CuTeDSL expand is faster than Triton.

    The dense CuTeDSL path wins for single-adapter prefill shapes once the
    output tile and token count are large enough; smaller output tiles stay on
    Triton.
    """
    if bi.single_lora_slot <= 0:
        return False
    if input_dim < 4096:
        if input_dim < 3072:
            if input_dim < 2048 and bi.max_len == 1:
                if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
                    return (lora_rank >= 64 and total_tokens >= 64) or (
                        lora_rank >= 32 and total_tokens >= 128
                    )
                return (
                    out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM
                    and lora_rank >= 64
                    and total_tokens >= 128
                )
            if bi.max_len == 1:
                if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
                    return lora_rank >= 64 and total_tokens >= 64
                return (
                    out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM
                    and lora_rank >= 64
                    and total_tokens >= 128
                )
            if bi.max_len <= _CHUNKED_THRESHOLD:
                return False
            if input_dim >= 1536 and input_dim < 2048:
                if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
                    return (lora_rank >= 64 and total_tokens >= 512) or (
                        lora_rank >= 32 and total_tokens >= 1024
                    )
                if out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM:
                    return lora_rank >= 64 and total_tokens >= 512
                return False
            if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
                return lora_rank >= 64 and total_tokens >= 512
            if out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM:
                return lora_rank >= 64 and total_tokens >= 1024
            return False
        if bi.max_len == 1:
            if out_dim < _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
                return (
                    out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM
                    and lora_rank >= 64
                    and total_tokens >= 64
                )
            return (lora_rank >= 64 and total_tokens >= 32) or (
                lora_rank >= 16 and total_tokens >= 128
            )
        if bi.max_len <= _CHUNKED_THRESHOLD:
            return False
        if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
            return lora_rank >= 16 and total_tokens >= 512
        if out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM:
            return lora_rank >= 64 and total_tokens >= 512
        return False
    if bi.max_len == 1:
        if out_dim >= _CUTEDSL_SINGLE_SLOT_DECODE_MIN_OUT_DIM:
            if lora_rank > 8 and lora_rank < 32:
                return total_tokens >= 64
            return lora_rank >= 8 and total_tokens >= 32
        if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
            if lora_rank >= 128:
                return total_tokens >= 32
            if lora_rank >= 32:
                return total_tokens >= 64
            if lora_rank >= 8 and out_dim == _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
                return total_tokens >= 128
            return lora_rank >= 16 and total_tokens >= 128
        return (
            out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM
            and lora_rank >= 64
            and total_tokens >= _CUTEDSL_SINGLE_SLOT_LOW_OUT_DECODE_MIN_TOKENS
        )
    if out_dim >= _CUTEDSL_SINGLE_SLOT_LOW_RANK_MIN_OUT_DIM and out_dim < (
        _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM
    ):
        if input_dim >= 8192:
            return bi.max_len > _CHUNKED_THRESHOLD and (
                (lora_rank >= 16 and total_tokens >= 256)
                or (lora_rank >= 8 and total_tokens >= 512)
            )
        return bi.max_len > _CHUNKED_THRESHOLD and (
            (
                lora_rank >= 64
                and total_tokens >= _CUTEDSL_SINGLE_SLOT_LOW_OUT_MIN_TOKENS
            )
            or (lora_rank >= 16 and total_tokens >= 512)
            or (lora_rank >= 8 and total_tokens >= 1024)
        )
    if out_dim >= _CUTEDSL_SINGLE_SLOT_PREFILL_MIN_OUT_DIM:
        if out_dim < _CUTEDSL_SINGLE_SLOT_DECODE_MIN_OUT_DIM:
            return (
                bi.max_len > _CHUNKED_THRESHOLD
                and lora_rank >= 8
                and total_tokens >= _CUTEDSL_SINGLE_SLOT_SMALL_PREFILL_MIN_TOKENS
            )
        return (
            bi.max_len > _CHUNKED_THRESHOLD
            and lora_rank >= 8
            and total_tokens > _CHUNKED_THRESHOLD
        )
    return False


def _use_cutedsl_multi_slot_expand(
    bi: LoraBatchInfo,
    total_tokens: int,
    out_dim: int,
    input_dim: int = 4096,
) -> bool:
    """Return whether equal-length consecutive multi-slot CuTeDSL should win."""
    if input_dim < 4096:
        if not (
            input_dim >= 3072
            and bi.multi_lora_start_slot > 0
            and bi.max_len > _CHUNKED_THRESHOLD
            and total_tokens > _CHUNKED_THRESHOLD
        ):
            return False
        if (
            bi.multi_lora_count == 4
            and bi.multi_lora_segment_len >= 128
            and bi.multi_lora_rank >= 32
            and out_dim >= 4096
        ):
            return True
        return (
            bi.multi_lora_count >= 2
            and bi.multi_lora_count <= 4
            and out_dim >= 8192
            and bi.multi_lora_rank >= 16
            and bi.multi_lora_segment_len >= 128
        )
    if bi.multi_lora_start_slot <= 0:
        return False
    if bi.multi_lora_count < 2 or bi.multi_lora_count > 4:
        return False
    if out_dim < _CUTEDSL_MULTI_SLOT_LOW_OUT_DIM:
        return False
    if out_dim < 4096 and bi.multi_lora_rank < 64:
        return False
    if (
        out_dim < _CUTEDSL_MULTI_SLOT_MIN_OUT_DIM
        and bi.multi_lora_segment_len < 256
        and not (bi.multi_lora_rank >= 64 and bi.multi_lora_segment_len >= 128)
    ):
        return False
    return (
        bi.max_len > _CHUNKED_THRESHOLD
        and bi.multi_lora_rank >= 8
        and total_tokens > _CHUNKED_THRESHOLD
        and (
            (bi.multi_lora_rank >= 64 and bi.multi_lora_segment_len >= 64)
            or (
                out_dim >= 8192
                and bi.multi_lora_rank >= 16
                and bi.multi_lora_segment_len >= 128
            )
            or bi.multi_lora_segment_len >= 256
            or (bi.multi_lora_count >= 4 and bi.multi_lora_segment_len >= 128)
        )
    )


def _use_cutedsl_single_slot_gate_up(
    bi: LoraBatchInfo,
    total_tokens: int,
    output_dim: int,
    lora_rank: int,
    input_dim: int = 4096,
) -> bool:
    """Return whether the two-GEMM CuTeDSL gate/up path should beat Triton."""
    if bi.single_lora_slot <= 0:
        return False
    if input_dim < 4096:
        if input_dim < 3072:
            if input_dim < 2048:
                if bi.max_len == 1:
                    if output_dim >= 2048:
                        return (lora_rank >= 64 and total_tokens >= 64) or (
                            lora_rank >= 32 and total_tokens >= 128
                        )
                    return (
                        output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM
                        and lora_rank >= 64
                        and total_tokens >= 64
                    )
                if bi.max_len <= _CHUNKED_THRESHOLD:
                    return False
                if output_dim >= 2048:
                    return lora_rank >= 64 and total_tokens >= 512
                return (
                    output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM
                    and lora_rank >= 64
                    and total_tokens >= 1024
                )
            if bi.max_len == 1:
                if output_dim >= 2048:
                    return lora_rank >= 64 and total_tokens >= 64
                return (
                    output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM
                    and lora_rank >= 64
                    and total_tokens >= 128
                )
            if bi.max_len <= _CHUNKED_THRESHOLD:
                return False
            return (
                output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM
                and lora_rank >= 64
                and total_tokens >= 512
            )
        if bi.max_len == 1:
            if output_dim >= _CUTEDSL_GATE_UP_SMALL_OUT_DIM:
                return (lora_rank >= 64 and total_tokens >= 32) or (
                    lora_rank >= 16 and total_tokens >= 64
                )
            if output_dim >= 2048:
                return (lora_rank >= 64 and total_tokens >= 64) or (
                    lora_rank >= 16 and total_tokens >= 128
                )
            if output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM:
                return lora_rank >= 64 and total_tokens >= 64
            return False
        if bi.max_len <= _CHUNKED_THRESHOLD:
            return False
        if output_dim >= _CUTEDSL_GATE_UP_SMALL_OUT_DIM:
            return (lora_rank >= 64 and total_tokens >= 256) or (
                lora_rank >= 16 and total_tokens >= 512
            )
        if output_dim >= 2048:
            return (lora_rank >= 64 and total_tokens >= 512) or (
                lora_rank >= 16 and total_tokens >= 1024
            )
        if output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM:
            return lora_rank >= 64 and total_tokens >= 512
        return False
    if bi.max_len == 1:
        if output_dim >= _CUTEDSL_GATE_UP_SMALL_OUT_DIM:
            return lora_rank >= 8 and total_tokens >= 32
        if output_dim >= 2048:
            if lora_rank >= 8 and total_tokens >= 64:
                return True
            if lora_rank >= 16 and total_tokens >= 64:
                return True
            return (lora_rank >= 64 and total_tokens >= 32) or (
                lora_rank >= 32 and total_tokens >= 64
            )
        return output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM and (
            (lora_rank >= 64 and total_tokens >= 32)
            or (lora_rank >= 16 and total_tokens >= 128)
        )
    if bi.max_len <= _CHUNKED_THRESHOLD:
        return False
    if output_dim >= _CUTEDSL_GATE_UP_SMALL_OUT_DIM:
        if output_dim >= _CUTEDSL_GATE_UP_LARGE_OUT_DIM:
            return lora_rank >= 8 and total_tokens >= 64
        if output_dim >= _CUTEDSL_GATE_UP_MEDIUM_OUT_DIM:
            return lora_rank >= 8 and total_tokens >= 64
        if lora_rank < 64:
            return lora_rank >= 8 and total_tokens >= 256
        return (lora_rank >= 64 and total_tokens >= 80) or (
            lora_rank >= 8 and total_tokens >= 128
        )
    if output_dim >= _CUTEDSL_SINGLE_SLOT_GATE_UP_SMALL_OUT_DIM:
        if output_dim < 2048:
            return (lora_rank >= 64 and total_tokens >= 512) or (
                lora_rank >= 8 and total_tokens >= 1024
            )
        if input_dim >= 8192 and lora_rank >= 8:
            return total_tokens >= 256
        if output_dim >= 3072 and lora_rank >= 8:
            return total_tokens >= 256
        return (
            (lora_rank >= 64 and total_tokens >= 256)
            or (lora_rank >= 16 and total_tokens >= 512)
            or (lora_rank >= 8 and total_tokens >= 512)
        )
    return False


def _use_cutedsl_single_slot_qkv(
    bi: LoraBatchInfo,
    total_tokens: int,
    q_dim: int,
    kv_dim: int,
    lora_rank: int,
    input_dim: int = 4096,
) -> bool:
    """Return whether the single-slot CuTeDSL QKV path should win."""
    if bi.single_lora_slot <= 0:
        return False
    if input_dim < 4096:
        if input_dim < 3072:
            if input_dim < 2048:
                if bi.max_len == 1:
                    if lora_rank >= 64 and q_dim >= 4096 and kv_dim >= 512:
                        return total_tokens >= 64
                    if lora_rank == 32 and q_dim >= 4096 and kv_dim >= 512:
                        if q_dim >= 8192 and kv_dim >= 1024:
                            return total_tokens >= 64
                        return total_tokens >= 96
                    return (
                        lora_rank == 16
                        and q_dim >= 8192
                        and kv_dim >= 1024
                        and total_tokens >= 96
                    )
                if bi.max_len <= _CHUNKED_THRESHOLD:
                    return False
                if input_dim >= 1536 and input_dim < 2048:
                    return (
                        (
                            lora_rank >= 64
                            and q_dim >= 4096
                            and kv_dim >= 512
                            and total_tokens >= 1536
                        )
                        or (
                            lora_rank >= 32
                            and q_dim >= 4096
                            and kv_dim >= 1024
                            and total_tokens >= 3072
                        )
                        or (
                            lora_rank >= 16
                            and q_dim >= 8192
                            and kv_dim >= 1024
                            and total_tokens >= 3072
                        )
                    )
                return (
                    lora_rank >= 64
                    and q_dim >= 4096
                    and kv_dim >= 512
                    and total_tokens >= 3072
                ) or (
                    lora_rank >= 16
                    and q_dim >= 8192
                    and kv_dim >= 1024
                    and total_tokens >= 3072
                )
            if bi.max_len == 1:
                if lora_rank >= 64 and q_dim >= 4096 and kv_dim >= 512:
                    return total_tokens >= 64
                if lora_rank == 32 and q_dim >= 4096 and kv_dim >= 512:
                    if q_dim >= 8192 and kv_dim >= 1024:
                        return total_tokens >= 64
                    return total_tokens >= 96
                return (
                    lora_rank == 16
                    and q_dim >= 8192
                    and kv_dim >= 1024
                    and total_tokens >= 96
                )
            if bi.max_len <= _CHUNKED_THRESHOLD:
                return False
            return (
                (
                    lora_rank >= 64
                    and q_dim >= 4096
                    and kv_dim >= 512
                    and total_tokens >= 1536
                )
                or (
                    lora_rank >= 32
                    and q_dim >= 4096
                    and kv_dim >= 1024
                    and total_tokens >= 3072
                )
                or (
                    lora_rank >= 16
                    and q_dim >= 8192
                    and kv_dim >= 1024
                    and total_tokens >= 3072
                )
            )
        if bi.max_len == 1:
            if (
                input_dim >= 3072
                and lora_rank >= 64
                and q_dim >= 4096
                and kv_dim >= 512
            ):
                return total_tokens >= 64
            if (
                input_dim >= 3072
                and lora_rank == 32
                and q_dim >= 4096
                and kv_dim >= 512
            ):
                return total_tokens >= 96
            if (
                input_dim >= 3072
                and lora_rank == 16
                and q_dim >= 4096
                and kv_dim >= 512
            ):
                return total_tokens >= 128 or (
                    q_dim >= 8192 and kv_dim >= 1024 and total_tokens >= 96
                )
            return False
        return (
            input_dim >= 3072
            and bi.max_len > _CHUNKED_THRESHOLD
            and (total_tokens >= 1536 if lora_rank >= 32 else total_tokens >= 3072)
            and (
                lora_rank >= 64
                or (q_dim >= 8192 and kv_dim >= 1024 and lora_rank >= 32)
                or (q_dim >= 8192 and kv_dim >= 1024 and lora_rank >= 16)
            )
        )
    if q_dim < 4096 or kv_dim < 512:
        return False
    if bi.max_len == 1:
        if lora_rank >= 64:
            return total_tokens >= 32
        if lora_rank >= 32:
            if kv_dim < 1024:
                if input_dim >= 8192:
                    return total_tokens >= 96
                return total_tokens >= 128
            return total_tokens >= 64
        if lora_rank >= 16:
            if kv_dim < 1024:
                return total_tokens >= 96
            return q_dim >= 8192 or total_tokens >= 96
        return False
    if bi.max_len <= _CHUNKED_THRESHOLD:
        return False
    if lora_rank >= 64:
        return total_tokens >= 1536
    return (
        (q_dim >= 8192 and kv_dim >= 1024 and lora_rank >= 32 and total_tokens >= 1536)
        or (
            q_dim >= 4096
            and kv_dim >= 512
            and lora_rank >= 32
            and total_tokens >= (1536 if input_dim >= 8192 else 3072)
        )
        or (
            q_dim >= 8192
            and kv_dim >= 1024
            and lora_rank >= 16
            and total_tokens >= (1536 if input_dim >= 8192 else 3072)
        )
    )


def _use_cutedsl_multi_slot_gate_up(
    bi: LoraBatchInfo,
    total_tokens: int,
    output_dim: int,
) -> bool:
    """Return whether equal-length consecutive multi-slot gate/up should win."""
    if bi.multi_lora_start_slot <= 0:
        return False
    if bi.multi_lora_count < 2 or bi.multi_lora_count > 4:
        return False
    if bi.max_len <= _CHUNKED_THRESHOLD or total_tokens <= _CHUNKED_THRESHOLD:
        return False
    if bi.multi_lora_rank < 64:
        return False
    if output_dim >= _CUTEDSL_GATE_UP_LARGE_OUT_DIM:
        return bi.multi_lora_segment_len >= 256 or (
            bi.multi_lora_count >= 4 and bi.multi_lora_segment_len >= 128
        )
    if output_dim >= _CUTEDSL_GATE_UP_MEDIUM_OUT_DIM:
        if bi.multi_lora_rank >= 128 and bi.multi_lora_segment_len >= 128:
            return True
        return bi.multi_lora_segment_len >= 256 or (
            bi.multi_lora_count >= 4 and bi.multi_lora_segment_len >= 128
        )
    if output_dim >= _CUTEDSL_GATE_UP_SMALL_OUT_DIM:
        return bi.multi_lora_rank >= 128 and bi.multi_lora_segment_len >= 256
    return False


def _use_triton_grouped_decode(bi: LoraBatchInfo) -> bool:
    """Return whether grouped Triton decode expand should beat basic decode."""
    return (
        bi.single_lora_slot <= 0
        and bi.num_groups > 0
        and bi.bs // bi.num_groups >= _TRITON_GROUPED_DECODE_MIN_GROUP_SIZE
    )


class LoraManager:
    """Owns GPU-resident LoRA weights and dispatches the segmented-GEMM path.

    Public surface (used by the model + executor):

    * :meth:`load_adapter` / :meth:`unload_adapter` — adapter lifecycle.
    * :attr:`batch_info` — persistent :class:`LoraBatchInfo` whose tensor
      pointers are stable across forward steps (so they can be baked into
      the captured CUDA graph).
    * :meth:`prepare_loras` — fill the persistent batch_info for one step.
    * :meth:`apply_qkv_lora` / :meth:`apply_o_lora` — Triton-backed deltas.
    """

    def __init__(
        self,
        model_config,
        max_loras: int,
        max_lora_rank: int,
        max_num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group=None,
        max_loras_cpu: int | None = None,
    ) -> None:
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.max_num_tokens = max_num_tokens
        self.dtype = dtype
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        # Tier-2 (CPU pinned) cap.  Defaults to 4× the GPU pool so adapter
        # spill-out to disk is rare in steady state.
        self.max_loras_cpu: int = (
            max_loras_cpu if max_loras_cpu is not None else 4 * max_loras
        )
        if self.max_loras_cpu < max_loras:
            raise ValueError(
                f"max_loras_cpu ({self.max_loras_cpu}) must be ≥ "
                f"max_loras ({max_loras}); GPU-resident adapters live in "
                "the CPU pool too."
            )

        self.n_layers: int = model_config.num_hidden_layers
        hidden: int = model_config.hidden_size
        n_heads: int = model_config.num_attention_heads
        n_kv: int = model_config.num_key_value_heads
        head_dim: int = hidden // n_heads

        self.q_size_per_tp: int = (n_heads // tp_size) * head_dim
        self.kv_size_per_tp: int = max(1, n_kv // tp_size) * head_dim
        self.o_in_per_tp: int = self.q_size_per_tp
        self.hidden_size: int = hidden

        # Qwen3MLP is TP-aware: ``gate_up_proj`` is column-parallel (each rank
        # holds ``intermediate_size // tp_size`` output cols) and ``down_proj``
        # is row-parallel (each rank holds ``intermediate_size // tp_size``
        # input cols).  The LoRA deltas ride the partial outputs of those base
        # linears, and the existing downstream all-reduce sums per-rank
        # partials — see ``apply_down_lora``/``apply_gate_up_lora``.
        self.intermediate_size: int = getattr(
            model_config, "intermediate_size", 4 * hidden
        )
        self.intermediate_per_tp: int = self.intermediate_size // self.tp_size

        # CPU-side flag: True when at least one segment in the current
        # batch_info uses a real adapter (slot != 0).  CudaGraphWrapper
        # reads this to pick the with-LoRA vs no-LoRA captured graph.
        self.has_active_lora: bool = False

        # Slot 0 = no-adapter sentinel.  Real adapters take 1 .. max_loras.
        # ── Tier 1: GPU pool ─────────────────────────────────────────────
        # Slot 0 = no-adapter sentinel.  Real adapters take 1 .. max_loras.
        self._n_slots: int = max_loras + 1
        self._slot_to_name: list[str | None] = [None] * self._n_slots
        self._name_to_slot: dict[str, int] = {}
        self._gpu_lru: OrderedDict[str, None] = OrderedDict()  # alias of _lru

        # ── Tier 2: CPU pinned pool ─────────────────────────────────────
        # ``_cpu_cache[name]`` holds parsed weights in pinned host memory.
        # ``_cpu_lru`` tracks LRU order for CPU eviction back to disk.  An
        # adapter is "CPU-resident" iff its name is in ``_cpu_cache``.
        # GPU-resident adapters are also kept in ``_cpu_cache`` (we pay
        # the host RAM cost once; reload to GPU is cheap and re-evicting
        # GPU then re-promoting only needs an H2D copy, not a disk read).
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._next_id: int = 1

        # ── Tier 2/3: CPU pinned pool + disk source of truth ─────────────
        self._cpu_store = LoraCpuCache(
            capacity=self.max_loras_cpu,
            is_gpu_resident=lambda name: name in self._name_to_slot,
        )
        # Compatibility aliases for existing tests/debug tooling.
        self._cpu_cache = self._cpu_store.cache
        self._cpu_lru = self._cpu_store.lru
        self._pinned = self._cpu_store.pinned
        self._adapter_paths = self._cpu_store.adapter_paths
        self._pending_loads = self._cpu_store.pending_loads

        # Per-slot rank + scaling.  Rank 0 means "no adapter"; the Triton
        # kernels skip on rank 0, so slot 0's row is permanently zero.
        self._lora_ranks: torch.Tensor = torch.zeros(
            self._n_slots, dtype=torch.int32, device=device
        )
        self._slot_ranks: list[int] = [0] * self._n_slots
        self._slot_scalings: list[float] = [0.0] * self._n_slots
        self._scalings: torch.Tensor = torch.zeros(
            self._n_slots, dtype=torch.float32, device=device
        )

        # ── Persistent batch_info ──────────────────────────────────────────
        # All tensors are sized for the worst case so their pointers are
        # stable across forward steps; per-step updates are in-place.
        # ``num_segments`` may equal ``bs`` (one segment per token in the
        # current path — no sort-by-adapter yet).
        self._batch_info = LoraBatchInfo(
            bs=0,
            num_segments=0,
            max_len=0,
            seg_lens=torch.zeros(max_num_tokens, dtype=torch.int32, device=device),
            seg_indptr=torch.zeros(
                max_num_tokens + 1, dtype=torch.int32, device=device
            ),
            weight_indices=torch.zeros(
                max_num_tokens, dtype=torch.int32, device=device
            ),
            lora_ranks=self._lora_ranks,
            scalings=self._scalings,
            permutation=None,
        )

        # CPU staging buffers (pinned) for the per-step H2D copy.
        self._seg_lens_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int32, pin_memory=True
        )
        self._weight_indices_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int32, pin_memory=True
        )
        # Adapter-group buffers for the decode grouped expand kernel.
        # Computed on CPU in prepare_loras (no GPU sync) and transferred
        # non-blocking.  Using stable GPU addresses so decode CUDA graphs
        # can capture the pointers; num_groups on axis=1 changes per step
        # so the graph grid must be re-evaluated outside the captured region.
        _mg = self._n_slots  # upper bound: one group per loaded adapter
        self._sort_order_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int64, pin_memory=True
        )
        self._group_slots_cpu = torch.zeros(_mg, dtype=torch.int32, pin_memory=True)
        self._group_starts_cpu = torch.zeros(_mg, dtype=torch.int32, pin_memory=True)
        self._group_sizes_cpu = torch.zeros(_mg, dtype=torch.int32, pin_memory=True)
        self._sort_order_buf = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self._group_slots_buf = torch.zeros(_mg, dtype=torch.int32, device=device)
        self._group_starts_buf = torch.zeros(_mg, dtype=torch.int32, device=device)
        self._group_sizes_buf = torch.zeros(_mg, dtype=torch.int32, device=device)

        # ── GPU weight buffers ─────────────────────────────────────────────
        # Attention:
        #   qkv_A_buffers: (n_slots, 3 * max_rank, hidden) — stacked q/k/v A.
        #   qkv_B_buffers: (n_slots, q_per_tp + 2 * kv_per_tp, max_rank).
        #   o_A_buffers:   (n_slots, max_rank, o_in_per_tp).
        #   o_B_buffers:   (n_slots, hidden, max_rank).
        # MLP (TP-aware, mirrors qwen3 ``Qwen3MLP``):
        #   gate_up_A_buffers: (n_slots, 2 * max_rank, hidden)             — A replicated.
        #   gate_up_B_buffers: (n_slots, 2 * intermediate_per_tp, max_rank) — column-parallel.
        #   down_A_buffers:    (n_slots, max_rank, intermediate_per_tp)    — row-parallel.
        #   down_B_buffers:    (n_slots, hidden, max_rank)                 — B replicated.
        self._weight_buffers = LoraWeightBuffers(
            n_layers=self.n_layers,
            n_slots=self._n_slots,
            max_lora_rank=self.max_lora_rank,
            hidden_size=self.hidden_size,
            q_size_per_tp=self.q_size_per_tp,
            kv_size_per_tp=self.kv_size_per_tp,
            o_in_per_tp=self.o_in_per_tp,
            intermediate_per_tp=self.intermediate_per_tp,
            dtype=self.dtype,
            device=self.device,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        self.qkv_A_buffers = self._weight_buffers.qkv_A_buffers
        self.qkv_B_buffers = self._weight_buffers.qkv_B_buffers
        self.o_A_buffers = self._weight_buffers.o_A_buffers
        self.o_B_buffers = self._weight_buffers.o_B_buffers
        self.gate_up_A_buffers = self._weight_buffers.gate_up_A_buffers
        self.gate_up_B_buffers = self._weight_buffers.gate_up_B_buffers
        self.down_A_buffers = self._weight_buffers.down_A_buffers
        self.down_B_buffers = self._weight_buffers.down_B_buffers
        self._qkv_output_offset = self._weight_buffers.qkv_output_offset
        self._max_qkv_out_dim = self._weight_buffers.max_qkv_out_dim
        self._o_slice_offsets = self._weight_buffers.o_slice_offsets
        self._gate_up_slice_offsets = self._weight_buffers.gate_up_slice_offsets
        self._down_slice_offsets = self._weight_buffers.down_slice_offsets
        self._moe_lora_buffers = MoeLoraBuffers(
            hidden_size=self.hidden_size,
            intermediate_per_tp=self.intermediate_per_tp,
            dtype=self.dtype,
            device=self.device,
            shard_weights=self._weight_buffers.shard_weights,
        )
        # Compatibility alias for tests/debug tooling that inspected the old
        # manager-owned storage directly.
        self._moe_lora_weights = self._moe_lora_buffers.weights_by_layer

        logger.info(
            "LoraManager initialized: max_loras=%d max_rank=%d "
            "tp_rank=%d/%d device=%s dtype=%s",
            max_loras,
            max_lora_rank,
            tp_rank,
            tp_size,
            device,
            dtype,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def batch_info(self) -> LoraBatchInfo:
        return self._batch_info

    @property
    def moe_lora_context(self) -> MoeLoraContext:
        return self._moe_lora_buffers.build_context(
            batch_info=self._batch_info,
            scalings=self._scalings,
            has_active_lora=self.has_active_lora,
        )

    def load_adapter(self, name: str, path: str, pinned: bool = False) -> int:
        """Register a PEFT adapter from *path* and warm the CPU pool.

        ``path`` is recorded as the adapter's durable disk path; it must
        remain accessible for the lifetime of the manager because the CPU
        pool may evict the adapter back to disk under memory pressure.

        Returns the integer ``lora_id`` to use in subsequent
        ``prepare_loras`` calls.
        """
        if name in self._name_to_id:
            logger.warning("Adapter '%s' is already loaded; re-loading.", name)
            self._evict_by_name(name)
            self._evict_from_cpu(name)

        # Resolve the durable disk path now (used by future re-reads when
        # the CPU pool evicts these weights).
        adapter_path = path
        weight_path = resolve_adapter_weight_path(adapter_path)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Adapter weights not found at {weight_path!r} or {path!r}"
            )

        lora_id = self._next_id
        self._next_id += 1
        self._name_to_id[name] = lora_id
        self._id_to_name[lora_id] = name
        self._cpu_store.set_path(name, adapter_path, pinned=pinned)

        # Warm the CPU pool — bounded by ``max_loras_cpu``, may evict
        # other CPU-resident adapters back to disk.
        self._cpu_store.ensure(name)

        logger.info(
            "Registered adapter '%s' (lora_id=%d) from %s; CPU pool: %d/%d",
            name,
            lora_id,
            path,
            len(self._cpu_cache),
            self.max_loras_cpu,
        )
        return lora_id

    def unload_adapter(self, name: str) -> None:
        if name not in self._name_to_id:
            raise KeyError(f"Adapter '{name}' is not loaded.")
        self._evict_by_name(name)
        self._cpu_store.remove(name)
        lora_id = self._name_to_id.pop(name)
        del self._id_to_name[lora_id]
        logger.info("Unloaded adapter '%s'", name)

    def get_id(self, name: str) -> int | None:
        return self._name_to_id.get(name)

    def prepare_loras(
        self,
        lora_ids: list[int],
        per_request_token_counts: list[int] | int = 1,
    ) -> int:
        """Fill :attr:`batch_info` for the upcoming forward.

        Each request becomes one segment.  Returns the total number of
        tokens written.  All updates are in place on the persistent
        batch_info tensors so the captured CUDA graph keeps replaying
        against the same pointers.
        """
        bs = len(lora_ids)
        # Resolve names → slots; LRU bookkeeping.
        per_request_slots: list[int] = []
        for lid in lora_ids:
            if lid == 0:
                per_request_slots.append(0)
                continue
            name = self._id_to_name.get(lid)
            if name is None:
                logger.warning("Unknown lora_id %d; treating as base model.", lid)
                per_request_slots.append(0)
                continue
            slot = self._ensure_in_gpu(name)
            per_request_slots.append(slot)
            self._gpu_lru.move_to_end(name)

        # Per-request seg_lens.
        if isinstance(per_request_token_counts, int):
            seg_lens_list = [per_request_token_counts] * bs
        else:
            if len(per_request_token_counts) != bs:
                raise ValueError(
                    "per_request_token_counts length must match lora_ids length"
                )
            seg_lens_list = list(per_request_token_counts)

        total_tokens = sum(seg_lens_list)
        if total_tokens > self.max_num_tokens:
            raise ValueError(
                f"LoRA batch_info overflow: {total_tokens} > {self.max_num_tokens}"
            )
        max_len = max(seg_lens_list) if seg_lens_list else 0

        bi = self._batch_info

        # For decode batches (max_len == 1): compute adapter groups on CPU
        # so the grouped expand kernel can batch same-adapter tokens into a
        # full BLOCK_S=16 GEMM tile, recovering tensor-core efficiency.
        if max_len == 1 and bs > 1:
            sort_order, group_slots, group_starts, group_sizes = (
                build_decode_lora_groups(per_request_slots)
            )
            ng = len(group_slots)
            self._sort_order_cpu[:bs] = torch.as_tensor(sort_order, dtype=torch.int64)
            self._group_slots_cpu[:ng] = torch.as_tensor(group_slots, dtype=torch.int32)
            self._group_starts_cpu[:ng] = torch.as_tensor(
                group_starts, dtype=torch.int32
            )
            self._group_sizes_cpu[:ng] = torch.as_tensor(group_sizes, dtype=torch.int32)
            bi.sort_order = self._sort_order_buf
            bi.group_slots = self._group_slots_buf
            bi.group_starts = self._group_starts_buf
            bi.group_sizes = self._group_sizes_buf
            bi.sort_order[:bs].copy_(self._sort_order_cpu[:bs], non_blocking=True)
            bi.group_slots[:ng].copy_(self._group_slots_cpu[:ng], non_blocking=True)
            bi.group_starts[:ng].copy_(self._group_starts_cpu[:ng], non_blocking=True)
            bi.group_sizes[:ng].copy_(self._group_sizes_cpu[:ng], non_blocking=True)
            bi.num_groups = ng
        else:
            bi.sort_order = bi.group_slots = bi.group_starts = bi.group_sizes = None
            bi.num_groups = 0

        first_slot = per_request_slots[0] if per_request_slots else 0
        bi.single_lora_slot = (
            first_slot
            if first_slot != 0 and all(slot == first_slot for slot in per_request_slots)
            else -1
        )
        bi.single_lora_rank = (
            self._slot_ranks[bi.single_lora_slot] if bi.single_lora_slot > 0 else 0
        )
        bi.multi_lora_start_slot = -1
        bi.multi_lora_count = 0
        bi.multi_lora_segment_len = 0
        bi.multi_lora_rank = 0
        if (
            bs > 1
            and bi.single_lora_slot <= 0
            and max_len > _CHUNKED_THRESHOLD
            and len(set(seg_lens_list)) == 1
            and all(slot > 0 for slot in per_request_slots)
        ):
            start_slot = per_request_slots[0]
            consecutive_slots = all(
                slot == start_slot + i for i, slot in enumerate(per_request_slots)
            )
            rank = self._slot_ranks[start_slot]
            scaling = self._slot_scalings[start_slot]
            same_rank_and_scaling = all(
                self._slot_ranks[slot] == rank and self._slot_scalings[slot] == scaling
                for slot in per_request_slots
            )
            if consecutive_slots and rank > 0 and same_rank_and_scaling:
                bi.multi_lora_start_slot = start_slot
                bi.multi_lora_count = bs
                bi.multi_lora_segment_len = seg_lens_list[0]
                bi.multi_lora_rank = rank

        # Stage on CPU then a single non-blocking H2D.
        self._seg_lens_cpu[:bs] = torch.as_tensor(seg_lens_list, dtype=torch.int32)
        self._weight_indices_cpu[:bs] = torch.as_tensor(
            per_request_slots, dtype=torch.int32
        )

        bi = self._batch_info
        bi.seg_lens[:bs].copy_(self._seg_lens_cpu[:bs], non_blocking=True)
        bi.weight_indices[:bs].copy_(self._weight_indices_cpu[:bs], non_blocking=True)
        # cumsum on device — same number of segments as bs.
        bi.seg_indptr[0] = 0
        torch.cumsum(bi.seg_lens[:bs], dim=0, out=bi.seg_indptr[1 : bs + 1])
        bi.bs = bs
        bi.num_segments = bs
        bi.max_len = max_len

        # Host-side flag: True iff at least one request resolved to a real
        # adapter slot.  The CudaGraphWrapper reads this before each replay
        # to pick the no-LoRA graph variant when the whole batch is
        # base-model — saving the per-step Triton-kernel launches.
        self.has_active_lora = any(s != 0 for s in per_request_slots)
        return total_tokens

    def apply_qkv_lora(
        self,
        hidden_states: torch.Tensor,
        qkv: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Fused QKV LoRA delta: ``qkv += B @ A @ x * scaling``.

        ``hidden_states``: ``(s, hidden)`` (full input).
        ``qkv``:           ``(s, q_per_tp + 2 * kv_per_tp)`` (output of qkv_proj
        on this rank).  Updated in place via the kernel's fused-add.
        """
        if hidden_states.shape[0] == 0:
            return qkv
        bi = self._batch_info
        if bi.bs == 0:
            return qkv

        A_buf = self.qkv_A_buffers[layer_id]
        B_buf = self.qkv_B_buffers[layer_id]
        # lora_a: (s, 3 * max_rank)
        lora_a = (
            lora_shrink_prefill_fwd(hidden_states, A_buf, bi, stack_num=3)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(hidden_states, A_buf, bi, stack_num=3)
        )
        if _use_cutedsl_single_slot_qkv(
            bi,
            lora_a.shape[0],
            self.q_size_per_tp,
            self.kv_size_per_tp,
            bi.single_lora_rank,
            input_dim=hidden_states.shape[1],
        ):
            lora_qkv_single_slot_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                self.q_size_per_tp,
                self.kv_size_per_tp,
                qkv,
                apply_scaling=True,
                single_weight_index=bi.single_lora_slot,
            )
        elif bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._qkv_output_offset,
                self._max_qkv_out_dim,
                base_output=qkv,
            )
        else:
            lora_qkv_expand_fwd(
                lora_a,
                B_buf,
                bi,
                self._qkv_output_offset,
                self._max_qkv_out_dim,
                base_output=qkv,
            )
        return qkv

    def apply_o_lora(
        self,
        attn_output: torch.Tensor,
        o_output: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Row-parallel O-projection LoRA delta.

        ``attn_output``: ``(s, q_per_tp)`` per-rank attention output (input
        to o_proj).
        ``o_output``: ``(s, hidden)`` partial sum from the host o_proj
        (``reduce_results=False`` on this codebase).  Updated in place.

        Each rank computes ``B @ A_local @ x_local`` — a partial of shape
        ``(s, hidden)``.  A is sharded along its input dim and B is
        replicated, so the sum of partials over ranks equals
        ``B @ A_full @ x_full``.  The host layer's downstream fused
        all-reduce in ``post_attention_layernorm`` sums the base partial
        and the LoRA partial together, producing the correct full output.
        """
        if attn_output.shape[0] == 0:
            return o_output
        bi = self._batch_info
        if bi.bs == 0:
            return o_output

        A_buf = self.o_A_buffers[layer_id]
        B_buf = self.o_B_buffers[layer_id]
        # lora_a (partial per rank): (s, max_rank).  No internal all-reduce —
        # the partial flows into B and the result rides the downstream sum.
        lora_a = (
            lora_shrink_prefill_fwd(attn_output, A_buf, bi, stack_num=1)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(attn_output, A_buf, bi, stack_num=1)
        )
        if _use_cutedsl_single_slot_expand(
            bi,
            lora_a.shape[0],
            B_buf.shape[1],
            bi.single_lora_rank,
            input_dim=attn_output.shape[1],
        ):
            lora_expand_single_slot_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                base_output=o_output,
                apply_scaling=True,
                single_weight_index=bi.single_lora_slot,
            )
        elif _use_cutedsl_multi_slot_expand(
            bi,
            lora_a.shape[0],
            B_buf.shape[1],
            input_dim=attn_output.shape[1],
        ):
            lora_expand_batched_slots_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                base_output=o_output,
                apply_scaling=True,
            )
        elif bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._o_slice_offsets,
                self.hidden_size,
                base_output=o_output,
            )
        elif _use_triton_grouped_decode(bi):
            lora_expand_grouped_v2_fwd(lora_a, B_buf, bi, base_output=o_output)
        else:
            lora_expand_fwd(lora_a, B_buf, bi, base_output=o_output)
        return o_output

    def apply_gate_up_lora(
        self,
        hidden_states: torch.Tensor,
        gate_up: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Fused gate/up LoRA delta: ``gate_up += B @ A @ x * scaling``.

        ``hidden_states``: ``(s, hidden)``.
        ``gate_up``:       ``(s, 2 * intermediate_per_tp)`` — output of the
        column-parallel ``gate_up_proj`` (each rank holds its own output
        shard).  Updated in place via the kernel's fused-add.
        """
        if hidden_states.shape[0] == 0:
            return gate_up
        bi = self._batch_info
        if bi.bs == 0:
            return gate_up

        A_buf = self.gate_up_A_buffers[layer_id]
        B_buf = self.gate_up_B_buffers[layer_id]
        # lora_a: (s, 2 * max_rank) — gate's lora_a in [:, :r], up's in [:, r:].
        lora_a = (
            lora_shrink_prefill_fwd(hidden_states, A_buf, bi, stack_num=2)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(hidden_states, A_buf, bi, stack_num=2)
        )
        if _use_cutedsl_single_slot_gate_up(
            bi,
            lora_a.shape[0],
            self.intermediate_per_tp,
            bi.single_lora_rank,
            input_dim=hidden_states.shape[1],
        ):
            lora_gate_up_single_slot_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                self.intermediate_per_tp,
                base_output=gate_up,
                apply_scaling=True,
                single_weight_index=bi.single_lora_slot,
            )
        elif _use_cutedsl_multi_slot_gate_up(
            bi,
            lora_a.shape[0],
            self.intermediate_per_tp,
        ):
            lora_gate_up_batched_slots_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                self.intermediate_per_tp,
                base_output=gate_up,
                apply_scaling=True,
            )
        elif bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._gate_up_slice_offsets,
                self.intermediate_per_tp,
                base_output=gate_up,
            )
        else:
            lora_gate_up_expand_fwd(
                lora_a,
                B_buf,
                bi,
                self.intermediate_per_tp,
                base_output=gate_up,
            )
        return gate_up

    def apply_down_lora(
        self,
        x: torch.Tensor,
        down_output: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Down-projection LoRA delta (row-parallel under MLP TP).

        ``x``:           ``(s, intermediate_per_tp)`` — input to the
        row-parallel ``down_proj`` (this rank's input shard).
        ``down_output``: ``(s, hidden)`` — partial output of ``down_proj``
        before its all-reduce.  Updated in place.

        Each rank's delta is ``B @ A_local @ x_local``: A is sharded along
        the input dim and B is replicated, so summing per-rank deltas yields
        the full ``B @ A_full @ x_full``.  The base linear runs with
        ``reduce_results=False``; the downstream all-reduce that sums the
        base partial also sums the LoRA partials.
        """
        if x.shape[0] == 0:
            return down_output
        bi = self._batch_info
        if bi.bs == 0:
            return down_output

        A_buf = self.down_A_buffers[layer_id]
        B_buf = self.down_B_buffers[layer_id]
        lora_a = (
            lora_shrink_prefill_fwd(x, A_buf, bi, stack_num=1)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(x, A_buf, bi, stack_num=1)
        )
        if _use_cutedsl_single_slot_expand(
            bi,
            lora_a.shape[0],
            B_buf.shape[1],
            bi.single_lora_rank,
            input_dim=x.shape[1],
        ):
            lora_expand_single_slot_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                base_output=down_output,
                apply_scaling=True,
                single_weight_index=bi.single_lora_slot,
            )
        elif _use_cutedsl_multi_slot_expand(
            bi,
            lora_a.shape[0],
            B_buf.shape[1],
            input_dim=x.shape[1],
        ):
            lora_expand_batched_slots_cutedsl_fwd(
                lora_a,
                B_buf,
                bi,
                base_output=down_output,
                apply_scaling=True,
            )
        elif bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._down_slice_offsets,
                self.hidden_size,
                base_output=down_output,
            )
        elif _use_triton_grouped_decode(bi):
            lora_expand_grouped_v2_fwd(lora_a, B_buf, bi, base_output=down_output)
        else:
            lora_expand_fwd(lora_a, B_buf, bi, base_output=down_output)
        return down_output

    def apply_moe_gate_up_lora(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper; MoE-specific work lives in MoeLoraContext."""
        return self.moe_lora_context.apply_gate_up_lora(
            layer_id,
            hidden_states,
            topk_ids,
            gate_up_output,
            sorted_token_ids=sorted_token_ids,
        )

    def apply_moe_down_lora(
        self,
        layer_id: int,
        intermediate: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper; MoE-specific work lives in MoeLoraContext."""
        return self.moe_lora_context.apply_down_lora(
            layer_id,
            intermediate,
            topk_ids,
            topk_weights,
            down_output,
            sorted_token_ids=sorted_token_ids,
        )

    def set_adapter_scaling(self, name: str, scaling: float) -> None:
        slot = self._name_to_slot.get(name)
        if slot is not None:
            self._slot_scalings[slot] = scaling
            self._scalings[slot] = scaling

    # ── Slot allocation ─────────────────────────────────────────────────────

    def _ensure_in_gpu(self, name: str) -> int:
        if name in self._name_to_slot:
            return self._name_to_slot[name]
        # Tier-2 → Tier-1 promotion; may need to read from disk if the
        # CPU pool has evicted this adapter since registration.
        self._cpu_store.ensure(name)
        slot = self._find_free_slot()
        self._load_to_slot(name, slot)
        self._name_to_slot[name] = slot
        self._slot_to_name[slot] = name
        self._gpu_lru[name] = None
        return slot

    def prefetch(self, name: str) -> None:
        """Best-effort async warm of the CPU pool for *name*.

        Called from the request-admission path: when a request with a
        non-zero ``lora_id`` arrives the manager kicks off a background
        disk read so the safetensors I/O is overlapped with the previous
        forward step rather than blocking ``prepare_loras`` of the step
        that actually consumes the adapter.

        No-op when the adapter is already CPU-resident or a load is
        already in flight.  Silently ignores unknown adapters (the
        request will fall back to base via slot 0).
        """
        self._cpu_store.prefetch(name)

    def _evict_from_cpu(self, name: str) -> None:
        """Public helper, takes the lock.  Caller must ensure *name* is
        not currently GPU-resident."""
        self._cpu_store.evict(name)

    def _find_free_slot(self) -> int:
        for slot in range(1, self._n_slots):
            if self._slot_to_name[slot] is None:
                return slot
        for candidate_name in list(self._gpu_lru.keys()):
            if candidate_name in self._pinned:
                continue
            slot = self._name_to_slot[candidate_name]
            logger.debug("Evicting adapter '%s' from GPU slot %d", candidate_name, slot)
            del self._name_to_slot[candidate_name]
            self._slot_to_name[slot] = None
            del self._gpu_lru[candidate_name]
            return slot
        raise RuntimeError(
            "LoRA GPU pool is full and all adapters are pinned. "
            f"Increase max_loras (current: {self.max_loras}) or unpin an adapter."
        )

    def _load_to_slot(self, name: str, slot: int) -> None:
        cpu_weights = self._cpu_cache[name]
        rank = self._get_rank_for(name)
        scaling = self._get_scaling_for(name, rank)
        self._lora_ranks[slot] = rank
        self._slot_ranks[slot] = rank
        self._slot_scalings[slot] = scaling
        self._scalings[slot] = scaling
        self._weight_buffers.load_adapter_to_slot(cpu_weights, slot, rank)
        self._moe_lora_buffers.load_adapter_to_slot(cpu_weights, slot, rank)

        logger.debug("Loaded adapter '%s' into GPU slot %d (rank=%d)", name, slot, rank)

    def _get_rank_for(self, name: str) -> int:
        cpu_weights = self._cpu_cache.get(name, {})
        if not cpu_weights or 0 not in cpu_weights:
            return self.max_lora_rank
        # Read the rank from whichever module is present in layer 0 — the
        # adapter may target attention only, MLP only, or both.
        for mod in PEFT_MODULES:
            if mod in cpu_weights[0]:
                return cpu_weights[0][mod][0].shape[0]
        for mod, tensors in cpu_weights[0].items():
            if mod.startswith("experts."):
                return tensors[0].shape[0]
        return self.max_lora_rank

    def _get_scaling_for(self, name: str, rank: int) -> float:
        return read_adapter_scaling(self._adapter_paths.get(name), rank)

    def _evict_by_name(self, name: str) -> None:
        if name in self._name_to_slot:
            slot = self._name_to_slot.pop(name)
            self._slot_to_name[slot] = None
            self._weight_buffers.zero_slot(slot)
            self._moe_lora_buffers.clear_slot(slot)
            self._lora_ranks[slot] = 0
            self._slot_ranks[slot] = 0
            self._slot_scalings[slot] = 0.0
            self._scalings[slot] = 0.0
        self._gpu_lru.pop(name, None)
