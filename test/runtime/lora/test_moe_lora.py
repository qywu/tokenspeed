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

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.lora.lora_batch import NO_LORA_SLOT, LoraBatchInfo
from tokenspeed.runtime.lora.lora_manager import LoraManager
from tokenspeed.runtime.lora.moe_lora import MoeLoraBuffers, MoeLoraContext


def _batch_info(weight_indices: list[int]) -> LoraBatchInfo:
    bs = len(weight_indices)
    return LoraBatchInfo(
        bs=bs,
        num_segments=bs,
        max_len=1,
        seg_lens=torch.ones(bs, dtype=torch.int32),
        seg_indptr=torch.arange(bs + 1, dtype=torch.int32),
        weight_indices=torch.tensor(weight_indices, dtype=torch.int32),
        lora_ranks=torch.tensor([1], dtype=torch.int32),
        scalings=torch.tensor([0.5], dtype=torch.float32),
        permutation=None,
    )


def _context(weight_indices: list[int], *, active: bool = True) -> MoeLoraContext:
    dtype = torch.float32
    return MoeLoraContext(
        weights_by_layer={
            0: {
                0: {
                    "w13_A": torch.ones((2, 2, 2), dtype=dtype),
                    "w13_B": torch.ones((2, 4, 2), dtype=dtype),
                    "down_A": torch.ones((2, 1, 2), dtype=dtype),
                    "down_B": torch.ones((2, 2, 1), dtype=dtype),
                }
            }
        },
        batch_info=_batch_info(weight_indices),
        scalings=torch.tensor([0.5], dtype=dtype),
        has_active_lora=active,
    )


def _buffers(*, compressed_shared_outer: bool = False) -> MoeLoraBuffers:
    return MoeLoraBuffers(
        n_layers=1,
        n_slots=2,
        max_lora_rank=1,
        num_experts=2,
        hidden_size=2,
        intermediate_per_tp=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        shard_weights=lambda _module, lora_A, lora_B: (lora_A, lora_B),
        compressed_shared_outer=compressed_shared_outer,
    )


def test_moe_lora_context_applies_single_slot_gate_up_and_down():
    ctx = _context([0, 0])
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)

    gate_up = torch.zeros((2, 4))
    ctx.apply_gate_up_lora(0, hidden_states, topk_ids, gate_up)
    torch.testing.assert_close(
        gate_up,
        torch.tensor([[3.0, 3.0, 3.0, 3.0], [7.0, 7.0, 7.0, 7.0]]),
    )

    down = torch.zeros((2, 1, 2))
    ctx.apply_down_lora(
        0,
        torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
        topk_ids,
        torch.ones((2, 1)),
        down,
    )
    torch.testing.assert_close(down, torch.tensor([[[3.0, 3.0]], [[7.0, 7.0]]]))


def test_moe_lora_context_masks_mixed_base_tokens():
    ctx = _context([0, NO_LORA_SLOT])
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)
    gate_up = torch.zeros((2, 4))

    ctx.apply_gate_up_lora(0, hidden_states, topk_ids, gate_up)

    torch.testing.assert_close(
        gate_up,
        torch.tensor([[3.0, 3.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
    )


def test_moe_lora_context_noops_when_inactive():
    ctx = _context([0], active=False)
    gate_up = torch.zeros((1, 4))

    ctx.apply_gate_up_lora(
        0,
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[0]], dtype=torch.int64),
        gate_up,
    )

    torch.testing.assert_close(gate_up, torch.zeros((1, 4)))


def test_moe_lora_buffers_load_3d_shared_outer_adapter():
    buffers = _buffers()
    cpu_weights = {
        0: {
            "experts.w1": (
                torch.tensor([[[1.0, 2.0]]]),
                torch.tensor([[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]]),
            ),
            "experts.w2": (
                torch.tensor([[[5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0]]]),
                torch.tensor([[[13.0], [14.0]]]),
            ),
            "experts.w3": (
                torch.tensor([[[3.0, 4.0]]]),
                torch.tensor([[[30.0], [31.0], [32.0]], [[40.0], [41.0], [42.0]]]),
            ),
        }
    }

    buffers.load_adapter_to_slot(cpu_weights, slot=0, rank=1)
    weights = buffers.weights_by_layer[0][0]

    assert buffers.w13_A_buffers[0].shape == (2, 2, 2, 2)
    assert weights["w13_A"].data_ptr() == buffers.w13_A_buffers[0][0].data_ptr()
    assert weights["w13_A"].shape == (2, 2, 2)
    torch.testing.assert_close(
        weights["w13_A"][:, 0, :],
        torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
    )
    torch.testing.assert_close(
        weights["w13_A"][:, 1, :],
        torch.tensor([[3.0, 4.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        weights["w13_B"][:, :3, 0],
        torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
    )
    torch.testing.assert_close(
        weights["w13_B"][:, 3:, 1],
        torch.tensor([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]]),
    )
    torch.testing.assert_close(
        weights["down_A"][:, 0, :],
        torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
    )
    torch.testing.assert_close(
        weights["down_B"][:, :, 0],
        torch.tensor([[13.0, 14.0], [13.0, 14.0]]),
    )


def test_moe_lora_buffers_load_compressed_3d_shared_outer_adapter():
    buffers = _buffers(compressed_shared_outer=True)
    cpu_weights = {
        0: {
            "experts.w1": (
                torch.tensor([[[1.0, 2.0]]]),
                torch.tensor([[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]]),
            ),
            "experts.w2": (
                torch.tensor([[[5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0]]]),
                torch.tensor([[[13.0], [14.0]]]),
            ),
            "experts.w3": (
                torch.tensor([[[3.0, 4.0]]]),
                torch.tensor([[[30.0], [31.0], [32.0]], [[40.0], [41.0], [42.0]]]),
            ),
        }
    }

    buffers.load_adapter_to_slot(cpu_weights, slot=0, rank=1)
    weights = buffers.weights_by_layer[0][0]

    assert buffers.w13_A_buffers[0].shape == (2, 1, 2, 2)
    assert buffers.w13_B_buffers[0].shape == (2, 2, 6, 2)
    assert buffers.down_A_buffers[0].shape == (2, 2, 1, 3)
    assert buffers.down_B_buffers[0].shape == (2, 1, 2, 1)
    assert weights["w13_A"].shape == (1, 2, 2)
    assert weights["down_B"].shape == (1, 2, 1)

    ctx = MoeLoraContext(
        weights_by_layer=buffers.weights_by_layer,
        batch_info=_batch_info([0, 0]),
        scalings=torch.tensor([1.0], dtype=torch.float32),
        has_active_lora=True,
    )
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)
    gate_up = torch.zeros((2, 6))

    ctx.apply_gate_up_lora(0, hidden_states, topk_ids, gate_up)

    torch.testing.assert_close(
        gate_up,
        torch.tensor(
            [
                [50.0, 55.0, 60.0, 330.0, 341.0, 352.0],
                [220.0, 231.0, 242.0, 1000.0, 1025.0, 1050.0],
            ]
        ),
    )


def test_moe_lora_compressed_shared_outer_rejects_per_expert_adapter():
    buffers = _buffers(compressed_shared_outer=True)
    cpu_weights = {
        0: {
            "experts.w1": (
                torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
                torch.ones((2, 3, 1)),
            ),
            "experts.w2": (
                torch.ones((2, 1, 3)),
                torch.ones((2, 2, 1)),
            ),
            "experts.w3": (
                torch.ones((2, 1, 2)),
                torch.ones((2, 3, 1)),
            ),
        }
    }

    with pytest.raises(ValueError, match="shared-outer"):
        buffers.load_adapter_to_slot(cpu_weights, slot=0, rank=1)


def test_moe_lora_buffers_load_3d_per_expert_adapter():
    buffers = _buffers()
    cpu_weights = {
        0: {
            "experts.w1": (
                torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
                torch.tensor([[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]]),
            ),
            "experts.w2": (
                torch.tensor([[[30.0, 31.0, 32.0]], [[40.0, 41.0, 42.0]]]),
                torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),
            ),
            "experts.w3": (
                torch.tensor([[[9.0, 10.0]], [[11.0, 12.0]]]),
                torch.tensor([[[50.0], [51.0], [52.0]], [[60.0], [61.0], [62.0]]]),
            ),
        }
    }

    buffers.load_adapter_to_slot(cpu_weights, slot=0, rank=1)
    weights = buffers.weights_by_layer[0][0]

    torch.testing.assert_close(
        weights["w13_A"][:, 0, :],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        weights["w13_A"][:, 1, :],
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
    )
    torch.testing.assert_close(
        weights["down_B"][:, :, 0],
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )


def test_moe_lora_buffers_clear_slot_zeroes_preallocated_pool():
    buffers = _buffers()
    cpu_weights = {
        0: {
            "experts.w1": (
                torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
                torch.ones((2, 3, 1)),
            ),
            "experts.w2": (
                torch.ones((2, 1, 3)),
                torch.ones((2, 2, 1)),
            ),
            "experts.w3": (
                torch.ones((2, 1, 2)),
                torch.ones((2, 3, 1)),
            ),
        }
    }

    buffers.load_adapter_to_slot(cpu_weights, slot=1, rank=1)
    assert 1 in buffers.weights_by_layer[0]
    assert torch.count_nonzero(buffers.w13_A_buffers[0][1]).item() > 0

    buffers.clear_slot(1)

    assert 1 not in buffers.weights_by_layer[0]
    assert torch.count_nonzero(buffers.w13_A_buffers[0][1]).item() == 0
    assert torch.count_nonzero(buffers.w13_B_buffers[0][1]).item() == 0
    assert torch.count_nonzero(buffers.down_A_buffers[0][1]).item() == 0
    assert torch.count_nonzero(buffers.down_B_buffers[0][1]).item() == 0


def test_lora_manager_get_rank_uses_3d_moe_rank_dimension():
    manager = object.__new__(LoraManager)
    manager.max_lora_rank = 8
    manager._cpu_cache = {
        "adapter": {
            0: {
                "experts.w1": (
                    torch.empty((1, 4, 16)),
                    torch.empty((2, 32, 4)),
                )
            }
        }
    }

    assert manager._get_rank_for("adapter") == 4
