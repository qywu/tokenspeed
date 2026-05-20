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

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import GenerateReqInput


def _processor(registry: dict[str, int]) -> InputProcessor:
    return InputProcessor(SimpleNamespace(_lora_name_to_id=registry))


def test_resolve_lora_id_uses_registered_lora_name():
    obj = GenerateReqInput(text="hello", sampling_params={}, lora_name="adapter-a")

    assert _processor({"adapter-a": 7})._resolve_lora_id(obj) == 7


def test_resolve_lora_id_rejects_unknown_lora_name():
    obj = GenerateReqInput(text="hello", sampling_params={}, lora_name="missing")

    with pytest.raises(ValueError, match="not a registered adapter"):
        _processor({})._resolve_lora_id(obj)


def test_batched_generate_req_propagates_lora_name_per_item():
    obj = GenerateReqInput(
        text=["a", "b"],
        sampling_params={},
        lora_name=["adapter-a", None],
    )
    obj.normalize_batch_and_arguments()

    first = obj[0]
    second = obj[1]

    assert first.lora_name == "adapter-a"
    assert second.lora_name is None


def test_batched_generate_req_repeats_scalar_lora_name():
    obj = GenerateReqInput(
        text=["a", "b"],
        sampling_params={},
        lora_name="adapter-a",
    )
    obj.normalize_batch_and_arguments()

    assert obj[0].lora_name == "adapter-a"
    assert obj[1].lora_name == "adapter-a"
