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

import torch

from tokenspeed.runtime.lora.adapter_io import parse_adapter_weights


def test_parse_adapter_weights_accepts_expert_scoped_moe_modules():
    tensors = {
        "base_model.model.model.layers.3.mlp.experts.7.gate_proj.lora_A.weight": (
            torch.randn(4, 16)
        ),
        "base_model.model.model.layers.3.mlp.experts.7.gate_proj.lora_B.weight": (
            torch.randn(32, 4)
        ),
        "base_model.model.model.layers.3.mlp.experts.7.up_proj.lora_A.weight": (
            torch.randn(4, 16)
        ),
        "base_model.model.model.layers.3.mlp.experts.7.up_proj.lora_B.weight": (
            torch.randn(32, 4)
        ),
        "base_model.model.model.layers.3.mlp.experts.7.down_proj.lora_A.weight": (
            torch.randn(4, 32)
        ),
        "base_model.model.model.layers.3.mlp.experts.7.down_proj.lora_B.weight": (
            torch.randn(16, 4)
        ),
    }

    parsed = parse_adapter_weights(tensors)

    assert set(parsed[3]) == {
        "experts.7.gate_proj",
        "experts.7.up_proj",
        "experts.7.down_proj",
    }
    assert parsed[3]["experts.7.gate_proj"][0].shape == (4, 16)
    assert parsed[3]["experts.7.down_proj"][1].shape == (16, 4)


def test_parse_adapter_weights_accepts_3d_moe_modules():
    tensors = {
        "base_model.model.model.layers.1.mlp.experts.w1.lora_A.weight": torch.randn(
            1, 4, 16
        ),
        "base_model.model.model.layers.1.mlp.experts.w1.lora_B.weight": torch.randn(
            8, 32, 4
        ),
        "base_model.model.model.layers.1.mlp.experts.w2.lora_A.weight": torch.randn(
            8, 4, 32
        ),
        "base_model.model.model.layers.1.mlp.experts.w2.lora_B.weight": torch.randn(
            1, 16, 4
        ),
        "base_model.model.model.layers.1.mlp.experts.w3.lora_A.weight": torch.randn(
            1, 4, 16
        ),
        "base_model.model.model.layers.1.mlp.experts.w3.lora_B.weight": torch.randn(
            8, 32, 4
        ),
    }

    parsed = parse_adapter_weights(tensors)

    assert set(parsed[1]) == {"experts.w1", "experts.w2", "experts.w3"}
    assert parsed[1]["experts.w1"][0].shape == (1, 4, 16)
    assert parsed[1]["experts.w2"][1].shape == (1, 16, 4)
