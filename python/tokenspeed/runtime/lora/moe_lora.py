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

from dataclasses import dataclass
from typing import Callable

import torch

from tokenspeed.runtime.lora.lora_batch import NO_LORA_SLOT, LoraBatchInfo

MoeLayerSlotWeights = dict[int, dict[str, torch.Tensor]]
MoeWeightsByLayer = dict[int, MoeLayerSlotWeights]


@dataclass(frozen=True)
class MoeLoraContext:
    """Narrow per-forward view of MoE LoRA state consumed by MoE backends."""

    weights_by_layer: MoeWeightsByLayer
    batch_info: LoraBatchInfo
    scalings: torch.Tensor
    has_active_lora: bool

    def apply_gate_up_lora(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply expert-scoped LoRA to routed MoE gate/up output."""
        if hidden_states.shape[0] == 0 or topk_ids.numel() == 0:
            return gate_up_output
        slots, single_slot = self._token_slots(hidden_states.shape[0])
        if single_slot == NO_LORA_SLOT and slots is None:
            return gate_up_output
        if single_slot != NO_LORA_SLOT:
            self._apply_gate_up_slot(
                layer_id,
                single_slot,
                hidden_states,
                topk_ids,
                gate_up_output,
                sorted_token_ids=sorted_token_ids,
            )
            return gate_up_output
        assert slots is not None
        for slot_t in torch.unique(slots):
            slot = int(slot_t.item())
            if slot == NO_LORA_SLOT:
                continue
            self._apply_gate_up_slot(
                layer_id,
                slot,
                hidden_states,
                topk_ids,
                gate_up_output,
                token_mask=slots == slot,
                sorted_token_ids=sorted_token_ids,
            )
        return gate_up_output

    def apply_down_lora(
        self,
        layer_id: int,
        intermediate: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply expert-scoped LoRA to routed MoE down output."""
        if intermediate.shape[0] == 0 or topk_ids.numel() == 0:
            return down_output
        num_tokens = topk_ids.shape[0]
        slots, single_slot = self._token_slots(num_tokens)
        if single_slot == NO_LORA_SLOT and slots is None:
            return down_output
        route_input = self._route_rows_from_cache(
            intermediate,
            topk_ids.numel(),
            sorted_token_ids=sorted_token_ids,
        ).view(topk_ids.shape[0], topk_ids.shape[1], -1)
        if single_slot != NO_LORA_SLOT:
            self._apply_down_slot(
                layer_id,
                single_slot,
                route_input,
                topk_ids,
                topk_weights,
                down_output,
            )
            return down_output
        assert slots is not None
        for slot_t in torch.unique(slots):
            slot = int(slot_t.item())
            if slot == NO_LORA_SLOT:
                continue
            self._apply_down_slot(
                layer_id,
                slot,
                route_input,
                topk_ids,
                topk_weights,
                down_output,
                token_mask=slots == slot,
            )
        return down_output

    def _token_slots(self, num_tokens: int) -> tuple[torch.Tensor | None, int]:
        bi = self.batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return None, NO_LORA_SLOT
        if bi.single_lora_slot != NO_LORA_SLOT:
            return None, bi.single_lora_slot
        slots = torch.repeat_interleave(
            bi.weight_indices[: bi.bs], bi.seg_lens[: bi.bs]
        )
        if slots.numel() != num_tokens:
            # Token ownership changed under TP/EP communication. Mixed LoRA
            # cannot be applied safely without transforming the slot map too.
            return None, NO_LORA_SLOT
        return slots, NO_LORA_SLOT

    def _apply_gate_up_slot(
        self,
        layer_id: int,
        slot: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up_output: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> None:
        weights = self.weights_by_layer.get(layer_id, {}).get(slot)
        if weights is None:
            return
        w13_A = weights["w13_A"]
        w13_B = weights["w13_B"]
        num_experts = max(w13_A.shape[0], w13_B.shape[0])
        valid = (topk_ids >= 0) & (topk_ids < num_experts)
        if token_mask is not None:
            valid = valid & token_mask[:, None]
        if not torch.any(valid):
            return
        safe_ids = topk_ids.clamp(0, num_experts - 1).to(torch.long)
        selected_A = self._select_expert_weights(w13_A, safe_ids)
        lora_a = torch.einsum("mh,mkrh->mkr", hidden_states, selected_A)
        selected_B = self._select_expert_weights(w13_B, safe_ids)
        delta = torch.einsum("mkr,mknr->mkn", lora_a, selected_B)
        delta = delta * self.scalings[slot]
        delta = torch.where(valid[:, :, None], delta, torch.zeros_like(delta))
        self._add_route_delta(
            gate_up_output,
            delta.reshape(-1, delta.shape[-1]),
            sorted_token_ids=sorted_token_ids,
        )

    def _apply_down_slot(
        self,
        layer_id: int,
        slot: int,
        route_input: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        weights = self.weights_by_layer.get(layer_id, {}).get(slot)
        if weights is None:
            return
        down_A = weights["down_A"]
        down_B = weights["down_B"]
        num_experts = max(down_A.shape[0], down_B.shape[0])
        valid = (topk_ids >= 0) & (topk_ids < num_experts)
        if token_mask is not None:
            valid = valid & token_mask[:, None]
        if not torch.any(valid):
            return
        safe_ids = topk_ids.clamp(0, num_experts - 1).to(torch.long)
        selected_A = self._select_expert_weights(down_A, safe_ids)
        lora_a = torch.einsum("mki,mkri->mkr", route_input, selected_A)
        selected_B = self._select_expert_weights(down_B, safe_ids)
        delta = torch.einsum("mkr,mkhr->mkh", lora_a, selected_B)
        delta = delta * topk_weights[:, :, None].to(delta.dtype)
        delta = delta * self.scalings[slot]
        delta = torch.where(valid[:, :, None], delta, torch.zeros_like(delta))
        down_output.view(topk_ids.shape[0], topk_ids.shape[1], -1).add_(delta)

    @staticmethod
    def _select_expert_weights(
        weights: torch.Tensor,
        safe_ids: torch.Tensor,
    ) -> torch.Tensor:
        if weights.shape[0] == 1:
            return weights[0].expand(*safe_ids.shape, *weights.shape[1:])
        return weights[safe_ids]

    @staticmethod
    def _add_route_delta(
        output: torch.Tensor,
        route_delta: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None,
    ) -> None:
        if sorted_token_ids is None:
            output.view(route_delta.shape[0], -1).add_(route_delta)
            return
        route_count = route_delta.shape[0]
        valid_pos = torch.arange(
            sorted_token_ids.numel(), device=sorted_token_ids.device
        )
        valid = (sorted_token_ids >= 0) & (sorted_token_ids < route_count)
        valid_pos = valid_pos[valid]
        route_ids = sorted_token_ids[valid].to(torch.long)
        output[valid_pos].add_(route_delta[route_ids])

    @staticmethod
    def _route_rows_from_cache(
        cache: torch.Tensor,
        route_count: int,
        *,
        sorted_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if sorted_token_ids is None:
            return cache.view(route_count, -1)
        rows = torch.zeros(
            (route_count, cache.shape[-1]), dtype=cache.dtype, device=cache.device
        )
        valid_pos = torch.arange(
            sorted_token_ids.numel(), device=sorted_token_ids.device
        )
        valid = (sorted_token_ids >= 0) & (sorted_token_ids < route_count)
        valid_pos = valid_pos[valid]
        route_ids = sorted_token_ids[valid].to(torch.long)
        rows[route_ids] = cache[valid_pos]
        return rows


class MoeLoraBuffers:
    """Own expert-scoped MoE LoRA weights independently from dense buffers."""

    def __init__(
        self,
        *,
        n_layers: int,
        n_slots: int,
        max_lora_rank: int,
        num_experts: int,
        hidden_size: int,
        intermediate_per_tp: int,
        dtype: torch.dtype,
        device: torch.device,
        shard_weights: Callable[
            [str, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        enabled: bool = True,
        compressed_shared_outer: bool = False,
    ) -> None:
        self.n_layers = n_layers
        self.n_slots = n_slots
        self.max_lora_rank = max_lora_rank
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_per_tp = intermediate_per_tp
        self.dtype = dtype
        self.device = device
        self._shard_weights = shard_weights
        self.enabled = enabled
        self.compressed_shared_outer = compressed_shared_outer
        self.weights_by_layer: MoeWeightsByLayer = {}
        self.w13_A_buffers: list[torch.Tensor] = []
        self.w13_B_buffers: list[torch.Tensor] = []
        self.down_A_buffers: list[torch.Tensor] = []
        self.down_B_buffers: list[torch.Tensor] = []
        self._alloc()

    def _alloc(self) -> None:
        if not self.enabled:
            return
        n = self.n_slots
        e = max(self.num_experts, 0)
        r = self.max_lora_rank
        h = self.hidden_size
        i = self.intermediate_per_tp
        w13_a_experts = 1 if self.compressed_shared_outer else e
        w13_b_experts = e
        down_a_experts = e
        down_b_experts = 1 if self.compressed_shared_outer else e
        for _ in range(self.n_layers):
            self.w13_A_buffers.append(
                torch.zeros(
                    (n, w13_a_experts, 2 * r, h),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.w13_B_buffers.append(
                torch.zeros(
                    (n, w13_b_experts, 2 * i, 2 * r),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.down_A_buffers.append(
                torch.zeros(
                    (n, down_a_experts, r, i), dtype=self.dtype, device=self.device
                )
            )
            self.down_B_buffers.append(
                torch.zeros(
                    (n, down_b_experts, h, r), dtype=self.dtype, device=self.device
                )
            )

    def load_adapter_to_slot(self, cpu_weights, slot: int, rank: int) -> None:
        has_moe = any(
            mod.startswith("experts.")
            for modules in cpu_weights.values()
            for mod in modules
        )
        if has_moe and not self.enabled:
            raise ValueError(
                "Adapter contains MoE LoRA weights, but LoRA buffer group 'moe' "
                "is disabled."
            )
        if self.num_experts <= 0:
            if has_moe:
                raise ValueError(
                    "MoE LoRA adapter requires model_config.num_experts or "
                    "model_config.num_local_experts."
                )
            return
        rank = min(rank, self.max_lora_rank)
        for layer_id, modules in cpu_weights.items():
            if not any(mod.startswith("experts.") for mod in modules):
                continue
            self._clear_layer_slot(layer_id, slot)
            if any(
                mod in modules for mod in ("experts.w1", "experts.w2", "experts.w3")
            ):
                self._load_3d_adapter_layer(layer_id, modules, slot, rank)
            else:
                self._load_2d_adapter_layer(layer_id, modules, slot, rank)

    def _load_2d_adapter_layer(self, layer_id: int, modules, slot: int, rank: int):
        expert_ids = [
            int(mod.split(".")[1]) for mod in modules if mod.startswith("experts.")
        ]
        if not expert_ids:
            return
        if self.compressed_shared_outer:
            raise ValueError(
                "Compressed MoE shared-outer storage only supports 3D "
                "experts.w1/w2/w3 adapters."
            )
        num_experts = max(expert_ids) + 1
        self._check_num_experts(layer_id, num_experts)
        w13_A, w13_B, down_A, down_B = self._slot_layer_tensors(layer_id, slot)
        r = rank
        for mod, (lora_A_full, lora_B_full) in modules.items():
            if not mod.startswith("experts."):
                continue
            _, expert_id_s, module = mod.split(".", 2)
            expert_id = int(expert_id_s)
            lora_A_shard_cpu, lora_B_shard_cpu = self._shard_weights(
                module, lora_A_full, lora_B_full
            )
            actual_rank = min(lora_A_shard_cpu.shape[0], r)
            lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            lora_B_shard = lora_B_shard_cpu[:, :actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            self._copy_projection(
                module,
                expert_id,
                actual_rank,
                lora_A_shard,
                lora_B_shard,
                w13_A,
                w13_B,
                down_A,
                down_B,
                rank=r,
            )
        self.weights_by_layer.setdefault(layer_id, {})[slot] = {
            "w13_A": w13_A,
            "w13_B": w13_B,
            "down_A": down_A,
            "down_B": down_B,
        }

    def _load_3d_adapter_layer(self, layer_id: int, modules, slot: int, rank: int):
        required = ("experts.w1", "experts.w2", "experts.w3")
        missing = [name for name in required if name not in modules]
        if missing:
            raise ValueError(
                f"3D MoE LoRA layer {layer_id} is missing modules: {missing}"
            )
        w1_A, w1_B = modules["experts.w1"]
        w2_A, w2_B = modules["experts.w2"]
        w3_A, w3_B = modules["experts.w3"]
        num_experts = self._infer_3d_num_experts((w1_A, w1_B, w2_A, w2_B, w3_A, w3_B))
        self._check_num_experts(layer_id, num_experts)
        if self.compressed_shared_outer:
            self._check_shared_outer_layer(layer_id, modules, num_experts)
        w13_A, w13_B, down_A, down_B = self._slot_layer_tensors(layer_id, slot)
        self._copy_3d_projection(
            "gate_proj", w1_A, w1_B, w13_A, w13_B, down_A, down_B, rank
        )
        self._copy_3d_projection(
            "up_proj", w3_A, w3_B, w13_A, w13_B, down_A, down_B, rank
        )
        self._copy_3d_projection(
            "down_proj", w2_A, w2_B, w13_A, w13_B, down_A, down_B, rank
        )
        self.weights_by_layer.setdefault(layer_id, {})[slot] = {
            "w13_A": w13_A,
            "w13_B": w13_B,
            "down_A": down_A,
            "down_B": down_B,
        }

    def _check_num_experts(self, layer_id: int, adapter_num_experts: int) -> None:
        if adapter_num_experts > self.num_experts:
            raise ValueError(
                f"MoE LoRA layer {layer_id} has {adapter_num_experts} experts, "
                f"but the model has {self.num_experts}."
            )

    def _slot_layer_tensors(self, layer_id: int, slot: int):
        return (
            self.w13_A_buffers[layer_id][slot],
            self.w13_B_buffers[layer_id][slot],
            self.down_A_buffers[layer_id][slot],
            self.down_B_buffers[layer_id][slot],
        )

    def _clear_layer_slot(self, layer_id: int, slot: int) -> None:
        self.w13_A_buffers[layer_id][slot].zero_()
        self.w13_B_buffers[layer_id][slot].zero_()
        self.down_A_buffers[layer_id][slot].zero_()
        self.down_B_buffers[layer_id][slot].zero_()

    @staticmethod
    def _check_shared_outer_layer(
        layer_id: int,
        modules,
        num_experts: int,
    ) -> None:
        expected = {
            "experts.w1": (1, num_experts),
            "experts.w2": (num_experts, 1),
            "experts.w3": (1, num_experts),
        }
        for module, (expected_a, expected_b) in expected.items():
            lora_A, lora_B = modules[module]
            if lora_A.shape[0] != expected_a or lora_B.shape[0] != expected_b:
                raise ValueError(
                    "Compressed MoE shared-outer storage expects "
                    f"{module} A/B dim0=({expected_a}, {expected_b}) for "
                    f"layer {layer_id}; got {tuple(lora_A.shape)}, "
                    f"{tuple(lora_B.shape)}."
                )

    @staticmethod
    def _infer_3d_num_experts(tensors: tuple[torch.Tensor, ...]) -> int:
        num_experts = 0
        for tensor in tensors:
            if tensor.dim() != 3:
                raise ValueError(
                    f"3D MoE LoRA tensors must be rank-3, got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != 1:
                num_experts = max(num_experts, int(tensor.shape[0]))
        if num_experts <= 0:
            raise ValueError("3D MoE LoRA layer has no per-expert tensor dimension")
        for tensor in tensors:
            if tensor.shape[0] not in (1, num_experts):
                raise ValueError(
                    "3D MoE LoRA dim0 must be either 1 (shared) or num_experts "
                    f"({num_experts}); got {tuple(tensor.shape)}"
                )
        return num_experts

    def _copy_3d_projection(
        self,
        module: str,
        lora_A_full: torch.Tensor,
        lora_B_full: torch.Tensor,
        w13_A: torch.Tensor,
        w13_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        rank: int,
    ) -> None:
        num_experts = max(
            w13_A.shape[0], w13_B.shape[0], down_A.shape[0], down_B.shape[0]
        )
        if self.compressed_shared_outer:
            self._copy_3d_projection_compressed(
                module,
                lora_A_full,
                lora_B_full,
                w13_A,
                w13_B,
                down_A,
                down_B,
                rank,
                num_experts,
            )
            return
        for expert_id in range(num_experts):
            lora_A = self._select_3d_expert_tensor(lora_A_full, expert_id)
            lora_B = self._select_3d_expert_tensor(lora_B_full, expert_id)
            lora_A_shard_cpu, lora_B_shard_cpu = self._shard_weights(
                module, lora_A, lora_B
            )
            actual_rank = min(lora_A_shard_cpu.shape[0], rank)
            lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            lora_B_shard = lora_B_shard_cpu[:, :actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            self._copy_projection(
                module,
                expert_id,
                actual_rank,
                lora_A_shard,
                lora_B_shard,
                w13_A,
                w13_B,
                down_A,
                down_B,
                rank=rank,
                a_expert_id=self._dst_expert_id(module, "A", expert_id),
                b_expert_id=self._dst_expert_id(module, "B", expert_id),
            )

    def _copy_3d_projection_compressed(
        self,
        module: str,
        lora_A_full: torch.Tensor,
        lora_B_full: torch.Tensor,
        w13_A: torch.Tensor,
        w13_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        rank: int,
        num_experts: int,
    ) -> None:
        if module in ("gate_proj", "up_proj"):
            shared_A = self._select_3d_expert_tensor(lora_A_full, 0)
            first_B = self._select_3d_expert_tensor(lora_B_full, 0)
            lora_A_shard_cpu, _ = self._shard_weights(module, shared_A, first_B)
            actual_rank = min(lora_A_shard_cpu.shape[0], rank)
            lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            if module == "gate_proj":
                w13_A[0, :actual_rank, :].copy_(lora_A_shard, non_blocking=True)
            else:
                w13_A[0, rank : rank + actual_rank, :].copy_(
                    lora_A_shard, non_blocking=True
                )
            for expert_id in range(num_experts):
                expert_B = self._select_3d_expert_tensor(lora_B_full, expert_id)
                _, lora_B_shard_cpu = self._shard_weights(module, shared_A, expert_B)
                b_rank = min(lora_B_shard_cpu.shape[1], rank)
                lora_B_shard = lora_B_shard_cpu[:, :b_rank].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                if module == "gate_proj":
                    w13_B[expert_id, : self.intermediate_per_tp, :b_rank].copy_(
                        lora_B_shard, non_blocking=True
                    )
                else:
                    w13_B[
                        expert_id,
                        self.intermediate_per_tp : 2 * self.intermediate_per_tp,
                        rank : rank + b_rank,
                    ].copy_(lora_B_shard, non_blocking=True)
            return

        if module == "down_proj":
            first_A = self._select_3d_expert_tensor(lora_A_full, 0)
            shared_B = self._select_3d_expert_tensor(lora_B_full, 0)
            _, lora_B_shard_cpu = self._shard_weights(module, first_A, shared_B)
            b_rank = min(lora_B_shard_cpu.shape[1], rank)
            lora_B_shard = lora_B_shard_cpu[:, :b_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            down_B[0, :, :b_rank].copy_(lora_B_shard, non_blocking=True)
            for expert_id in range(num_experts):
                expert_A = self._select_3d_expert_tensor(lora_A_full, expert_id)
                lora_A_shard_cpu, _ = self._shard_weights(module, expert_A, shared_B)
                actual_rank = min(lora_A_shard_cpu.shape[0], rank)
                lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                down_A[expert_id, :actual_rank, :].copy_(
                    lora_A_shard, non_blocking=True
                )
            return

        raise ValueError(f"Unsupported MoE LoRA projection: {module}")

    @staticmethod
    def _select_3d_expert_tensor(tensor: torch.Tensor, expert_id: int) -> torch.Tensor:
        return tensor[0 if tensor.shape[0] == 1 else expert_id]

    def _copy_projection(
        self,
        module: str,
        expert_id: int,
        actual_rank: int,
        lora_A_shard: torch.Tensor,
        lora_B_shard: torch.Tensor,
        w13_A: torch.Tensor,
        w13_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        *,
        rank: int,
        a_expert_id: int | None = None,
        b_expert_id: int | None = None,
    ) -> None:
        a_expert_id = expert_id if a_expert_id is None else a_expert_id
        b_expert_id = expert_id if b_expert_id is None else b_expert_id
        if module == "gate_proj":
            w13_A[a_expert_id, :actual_rank, :].copy_(lora_A_shard, non_blocking=True)
            w13_B[
                b_expert_id,
                : self.intermediate_per_tp,
                :actual_rank,
            ].copy_(lora_B_shard, non_blocking=True)
        elif module == "up_proj":
            w13_A[a_expert_id, rank : rank + actual_rank, :].copy_(
                lora_A_shard, non_blocking=True
            )
            w13_B[
                b_expert_id,
                self.intermediate_per_tp : 2 * self.intermediate_per_tp,
                rank : rank + actual_rank,
            ].copy_(lora_B_shard, non_blocking=True)
        elif module == "down_proj":
            down_A[a_expert_id, :actual_rank, :].copy_(lora_A_shard, non_blocking=True)
            down_B[b_expert_id, :, :actual_rank].copy_(lora_B_shard, non_blocking=True)
        else:
            raise ValueError(f"Unsupported MoE LoRA projection: {module}")

    def _dst_expert_id(self, module: str, side: str, expert_id: int) -> int:
        if not self.compressed_shared_outer:
            return expert_id
        if module in ("gate_proj", "up_proj") and side == "A":
            return 0
        if module == "down_proj" and side == "B":
            return 0
        return expert_id

    def clear_slot(self, slot: int) -> None:
        if not self.enabled:
            return
        for layer_id in range(self.n_layers):
            self._clear_layer_slot(layer_id, slot)
        for layer_slots in self.weights_by_layer.values():
            layer_slots.pop(slot, None)

    def build_context(
        self,
        *,
        batch_info: LoraBatchInfo,
        scalings: torch.Tensor,
        has_active_lora: bool,
    ) -> MoeLoraContext:
        return MoeLoraContext(
            weights_by_layer=self.weights_by_layer,
            batch_info=batch_info,
            scalings=scalings,
            has_active_lora=has_active_lora,
        )
