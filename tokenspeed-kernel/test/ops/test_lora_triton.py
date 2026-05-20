from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


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


def _decode_batch(batch_size: int, rank: int, device: str) -> BatchInfo:
    return BatchInfo(
        bs=batch_size,
        max_len=1,
        seg_lens=torch.ones((batch_size,), dtype=torch.int32, device=device),
        seg_indptr=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        weight_indices=torch.ones((batch_size,), dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([0, rank], dtype=torch.int32, device=device),
        scalings=torch.ones((2,), dtype=torch.float32, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lora_expand_decode_rank_smaller_than_block_k_matches_reference():
    from tokenspeed_kernel.ops.lora.triton.lora_expand import lora_expand_fwd

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    rank = 8
    out_dim = 64
    torch.manual_seed(7)
    batch_info = _decode_batch(batch_size, rank, device)
    x = torch.randn((batch_size, rank), dtype=dtype, device=device)
    weights = torch.randn((2, out_dim, rank), dtype=dtype, device=device)
    base = torch.randn((batch_size, out_dim), dtype=dtype, device=device)

    out = lora_expand_fwd(x, weights, batch_info, base_output=base.clone())
    ref = base.float() + x.float() @ weights[1].float().T
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lora_gate_up_decode_rank_smaller_than_block_k_matches_reference():
    from tokenspeed_kernel.ops.lora.triton.lora_gate_up_expand import (
        lora_gate_up_expand_fwd,
    )

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    rank = 8
    out_dim = 64
    torch.manual_seed(8)
    batch_info = _decode_batch(batch_size, rank, device)
    x = torch.randn((batch_size, 2 * rank), dtype=dtype, device=device)
    weights = torch.randn((2, 2 * out_dim, rank), dtype=dtype, device=device)
    base = torch.randn((batch_size, 2 * out_dim), dtype=dtype, device=device)

    out = lora_gate_up_expand_fwd(
        x,
        weights,
        batch_info,
        out_dim,
        base_output=base.clone(),
    )
    ref = base.float()
    ref[:, :out_dim] += x[:, :rank].float() @ weights[1, :out_dim].float().T
    ref[:, out_dim:] += (
        x[:, rank : 2 * rank].float() @ weights[1, out_dim : 2 * out_dim].float().T
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lora_qkv_decode_rank_smaller_than_block_k_matches_reference():
    from tokenspeed_kernel.ops.lora.triton.lora_qkv_expand import lora_qkv_expand_fwd

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    rank = 8
    q_dim = 64
    kv_dim = 32
    torch.manual_seed(9)
    batch_info = _decode_batch(batch_size, rank, device)
    x = torch.randn((batch_size, 3 * rank), dtype=dtype, device=device)
    weights = torch.randn((2, q_dim + 2 * kv_dim, rank), dtype=dtype, device=device)
    base = torch.randn((batch_size, q_dim + 2 * kv_dim), dtype=dtype, device=device)
    offsets = torch.tensor(
        [0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim],
        dtype=torch.int32,
        device=device,
    )

    out = lora_qkv_expand_fwd(
        x,
        weights,
        batch_info,
        offsets,
        q_dim,
        base_output=base.clone(),
    )
    ref = base.float()
    ref[:, :q_dim] += x[:, :rank].float() @ weights[1, :q_dim].float().T
    ref[:, q_dim : q_dim + kv_dim] += (
        x[:, rank : 2 * rank].float() @ weights[1, q_dim : q_dim + kv_dim].float().T
    )
    ref[:, q_dim + kv_dim :] += (
        x[:, 2 * rank : 3 * rank].float() @ weights[1, q_dim + kv_dim :].float().T
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)
