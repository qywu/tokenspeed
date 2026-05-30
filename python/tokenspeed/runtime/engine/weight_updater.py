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

"""Distributed weight-update support.

External training pipelines (e.g. RLHF / online fine-tuning trainers) push
freshly updated weights into the serving engine by:

  1. ``init_weights_update_group`` — once per training run; establishes a
     standalone ``torch.distributed`` process group spanning the trainer's
     broadcaster rank(s) and every engine worker.
  2. ``update_weights_from_distributed`` — per parameter list; trainer rank
     ``0`` broadcasts each tensor and each engine worker copies it into the
     live model via ``model.load_weights``.
  3. ``destroy_weights_update_group`` — at shutdown; cleans up the group.

The trainer side mirrors the API used by other inference engines (sglang,
vLLM) so existing RLHF integrations port without behavioural changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    _new_process_group_helper,
    default_pg_timeout,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

logger = logging.getLogger(__name__)


_DTYPE_NAME_TO_TORCH: dict[str, torch.dtype] = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
    "torch.double": torch.float64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.long": torch.int64,
    "torch.bool": torch.bool,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name in _DTYPE_NAME_TO_TORCH:
        return _DTYPE_NAME_TO_TORCH[name]
    qualified = name if name.startswith("torch.") else f"torch.{name}"
    if qualified in _DTYPE_NAME_TO_TORCH:
        return _DTYPE_NAME_TO_TORCH[qualified]
    raise ValueError(f"WeightUpdater: unsupported dtype string {name!r}")


class WeightUpdater:
    """Per-worker controller for distributed weight updates.

    One instance per scheduler subprocess. Holds at most one process group
    per ``group_name``; broadcasts parameter tensors from the trainer's
    rank into the worker's live ``nn.Module`` via ``load_weights``.
    """

    def __init__(self, model_executor: "ModelExecutor"):
        self.model_executor = model_executor
        self.model_runner = model_executor.model_runner
        # Position in the engine's world; the caller-supplied ``rank_offset``
        # places this worker after the trainer ranks in the new group.
        self.engine_rank = self.model_runner.global_rank
        self._groups: dict[str, dist.ProcessGroup] = {}

    # ---- Public API -------------------------------------------------

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> tuple[bool, str]:
        """Stand up a standalone process group for weight broadcasts.

        Builds an isolated ``TCPStore`` rendezvous so the new group does not
        conflict with the engine's default process group. The trainer
        typically sits at ranks ``[0, rank_offset)`` and engine workers fill
        ranks ``[rank_offset, world_size)``.
        """
        if group_name in self._groups:
            return True, f"group '{group_name}' already initialized"

        my_rank = self.engine_rank + rank_offset
        if not (0 <= my_rank < world_size):
            return False, (
                f"computed rank {my_rank} out of range "
                f"(engine_rank={self.engine_rank}, rank_offset={rank_offset}, "
                f"world_size={world_size})"
            )

        logger.info(
            "init_weights_update_group: name=%s backend=%s "
            "engine_rank=%d -> group_rank=%d/%d store=%s:%d",
            group_name,
            backend,
            self.engine_rank,
            my_rank,
            world_size,
            master_address,
            master_port,
        )

        try:
            store = dist.TCPStore(
                host_name=master_address,
                port=master_port,
                world_size=world_size,
                is_master=(my_rank == 0),
                timeout=default_pg_timeout,
            )
            pg, _ = _new_process_group_helper(
                group_size=world_size,
                group_rank=my_rank,
                global_ranks_in_group=list(range(world_size)),
                backend=backend,
                store=store,
                group_name=group_name,
                pg_options=None,
                timeout=default_pg_timeout,
            )
        except Exception as exc:
            logger.exception("init_weights_update_group failed")
            return False, f"init_weights_update_group failed: {exc!r}"

        self._groups[group_name] = pg
        return True, ""

    def update_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        flush_cache: bool,
    ) -> tuple[bool, str]:
        """Receive a batch of parameter tensors broadcast by the trainer.

        ``flush_cache`` is acknowledged but cache invalidation is handled by
        the caller via ``FlushCacheReqInput`` so the prefix cache is dropped
        in the same atomic step as the weight swap.
        """
        del flush_cache  # acknowledged by the caller-side flow, see docstring
        if group_name not in self._groups:
            return False, f"group '{group_name}' not initialized"
        if not (len(names) == len(dtypes) == len(shapes)):
            return False, (
                f"names/dtypes/shapes length mismatch: "
                f"{len(names)}/{len(dtypes)}/{len(shapes)}"
            )

        pg = self._groups[group_name]
        # Convention: trainer rank 0 is the broadcast source. Any trainer
        # using more than one broadcaster rank must coordinate which one is
        # source separately — this contract matches sglang and vLLM.
        src = 0
        device = self._param_device()

        try:
            received: list[tuple[str, torch.Tensor]] = []
            for name, dtype_str, shape in zip(names, dtypes, shapes):
                dtype = _resolve_dtype(dtype_str)
                tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
                dist.broadcast(tensor, src=src, group=pg)
                received.append((name, tensor))
            self.model_runner.model.load_weights(received)
        except Exception as exc:
            logger.exception("update_weights_from_distributed failed")
            return False, f"update_weights_from_distributed failed: {exc!r}"

        return True, f"updated {len(received)} parameter(s)"

    def destroy_process_group(self, group_name: str) -> tuple[bool, str]:
        pg = self._groups.pop(group_name, None)
        if pg is None:
            return False, f"group '{group_name}' not initialized"
        try:
            dist.destroy_process_group(pg)
        except Exception as exc:
            logger.exception("destroy_weights_update_group failed")
            return False, f"destroy_weights_update_group failed: {exc!r}"
        return True, ""

    # ---- Helpers ----------------------------------------------------

    def _param_device(self) -> torch.device:
        # Pick the device any model parameter sits on; falls back to the
        # current CUDA device.
        for param in self.model_runner.model.parameters():
            if param.device.type == "cuda":
                return param.device
        return torch.device("cuda", torch.cuda.current_device())
