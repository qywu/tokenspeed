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

"""Tests for the distributed weight-update path.

Covers the unit-level pieces that don't require a real ModelExecutor:
  - IO struct shapes
  - WeightUpdater state machine (init -> update -> destroy) using mocks
  - dtype-string resolution

A full multi-process NCCL integration test belongs in a separate
GPU-multi-rank test target — adding it here would require two CUDA
devices and external broadcaster ranks, which the CI lane doesn't
provide as part of single-GPU runtime tests.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.engine.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
)
from tokenspeed.runtime.engine.weight_updater import (
    WeightUpdater,
    _resolve_dtype,
)


class _StubLinear(torch.nn.Module):
    """Trivial model with a single tracked parameter and load_weights()."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))

    def load_weights(self, weights):
        """Mimic ``CausalLM.load_weights``: copy by name into params_dict."""
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].data.copy_(tensor)


def _make_updater(model: torch.nn.Module, *, rank: int = 0) -> WeightUpdater:
    model_runner = SimpleNamespace(model=model, global_rank=rank)
    model_executor = SimpleNamespace(model_runner=model_runner)
    return WeightUpdater(model_executor)


class IoStructTest(unittest.TestCase):

    def test_update_weights_from_distributed_input_shape(self):
        # The struct must accept the plural-name kwargs the engine API uses.
        req = UpdateWeightsFromDistributedReqInput(
            names=["a", "b"],
            dtypes=["torch.float16", "torch.bfloat16"],
            shapes=[[2, 2], [4]],
            group_name="g0",
            flush_cache=False,
        )
        self.assertEqual(req.names, ["a", "b"])
        self.assertEqual(req.group_name, "g0")
        self.assertFalse(req.flush_cache)

    def test_destroy_weights_update_group_defaults(self):
        req = DestroyWeightsUpdateGroupReqInput()
        self.assertEqual(req.group_name, "weight_update_group")


class DtypeResolveTest(unittest.TestCase):

    def test_qualified_names(self):
        self.assertIs(_resolve_dtype("torch.bfloat16"), torch.bfloat16)
        self.assertIs(_resolve_dtype("torch.float16"), torch.float16)

    def test_shorthand_names(self):
        self.assertIs(_resolve_dtype("bfloat16"), torch.bfloat16)
        self.assertIs(_resolve_dtype("float32"), torch.float32)

    def test_unknown_dtype_raises(self):
        with self.assertRaises(ValueError):
            _resolve_dtype("complex64")


class WeightUpdaterStateMachineTest(unittest.TestCase):
    """Black-box state transitions with torch.distributed mocked out."""

    def setUp(self):
        self.model = _StubLinear()
        self.updater = _make_updater(self.model, rank=0)

    @mock.patch("tokenspeed.runtime.engine.weight_updater._new_process_group_helper")
    @mock.patch("tokenspeed.runtime.engine.weight_updater.dist.TCPStore")
    def test_init_creates_group_and_is_idempotent_by_name(self, _store, _helper):
        _helper.return_value = (mock.Mock(name="ProcessGroup"), None)
        ok, _ = self.updater.init_process_group(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=0,
            world_size=2,
            group_name="g0",
            backend="gloo",
        )
        self.assertTrue(ok)
        self.assertIn("g0", self.updater._groups)

        # Second call with the same name is a no-op.
        ok2, msg = self.updater.init_process_group(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=0,
            world_size=2,
            group_name="g0",
            backend="gloo",
        )
        self.assertTrue(ok2)
        self.assertIn("already initialized", msg)
        _helper.assert_called_once()  # not called a second time

    @mock.patch("tokenspeed.runtime.engine.weight_updater._new_process_group_helper")
    @mock.patch("tokenspeed.runtime.engine.weight_updater.dist.TCPStore")
    def test_init_rejects_out_of_range_rank(self, _store, _helper):
        ok, msg = self.updater.init_process_group(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=10,  # engine_rank=0 + rank_offset=10 = 10 >= world_size=2
            world_size=2,
            group_name="g0",
            backend="gloo",
        )
        self.assertFalse(ok)
        self.assertIn("out of range", msg)
        _helper.assert_not_called()

    def test_update_without_init_fails(self):
        ok, msg = self.updater.update_from_distributed(
            names=["weight"],
            dtypes=["torch.float32"],
            shapes=[[4, 4]],
            group_name="missing",
            flush_cache=True,
        )
        self.assertFalse(ok)
        self.assertIn("not initialized", msg)

    @mock.patch("tokenspeed.runtime.engine.weight_updater.dist.broadcast")
    def test_update_broadcasts_and_loads(self, broadcast):
        # Pre-populate a group so update_from_distributed gets past the guard.
        sentinel_pg = mock.Mock(name="ProcessGroup")
        self.updater._groups["g0"] = sentinel_pg

        # broadcast() writes ones into the receive buffer.
        def _fill_ones(tensor, src, group):
            self.assertEqual(src, 0)
            self.assertIs(group, sentinel_pg)
            tensor.fill_(1.0)

        broadcast.side_effect = _fill_ones

        # Move stub model to CPU so the CUDA fallback path is not hit.
        self.model.weight.data = self.model.weight.data.cpu()
        with mock.patch.object(
            self.updater, "_param_device", return_value=torch.device("cpu")
        ):
            ok, msg = self.updater.update_from_distributed(
                names=["weight"],
                dtypes=["torch.float32"],
                shapes=[[4, 4]],
                group_name="g0",
                flush_cache=False,
            )

        self.assertTrue(ok, msg)
        broadcast.assert_called_once()
        self.assertTrue(torch.all(self.model.weight == 1.0))

    def test_update_rejects_length_mismatch(self):
        self.updater._groups["g0"] = mock.Mock()
        ok, msg = self.updater.update_from_distributed(
            names=["a", "b"],
            dtypes=["torch.float32"],  # length mismatch
            shapes=[[1], [1]],
            group_name="g0",
            flush_cache=True,
        )
        self.assertFalse(ok)
        self.assertIn("length mismatch", msg)

    @mock.patch("tokenspeed.runtime.engine.weight_updater.dist.destroy_process_group")
    def test_destroy_removes_group(self, destroy):
        self.updater._groups["g0"] = mock.Mock(name="ProcessGroup")
        ok, _ = self.updater.destroy_process_group("g0")
        self.assertTrue(ok)
        self.assertNotIn("g0", self.updater._groups)
        destroy.assert_called_once()

    def test_destroy_unknown_group_fails(self):
        ok, msg = self.updater.destroy_process_group("never_inited")
        self.assertFalse(ok)
        self.assertIn("not initialized", msg)


if __name__ == "__main__":
    unittest.main()
