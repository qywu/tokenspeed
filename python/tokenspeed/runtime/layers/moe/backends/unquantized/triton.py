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

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.triton_common import (
    build_triton_gemms,
    triton_forward,
)
from tokenspeed.runtime.layers.moe.backends.triton_weights import (
    attach_dense_weight_pair,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec


class Bf16TritonBackend(MoEBackend):
    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return quant_config is None

    def create_layer_weights(self, layer, *, with_bias: bool = False) -> None:
        attach_dense_weight_pair(
            self,
            layer,
            with_bias=with_bias,
            params_dtype=torch.get_default_dtype(),
        )
        self._gate_up_gemm, self._down_gemm, self._get_config_func = build_triton_gemms(
            layer,
            self.spec,
            dtype_tag="bf16",
        )

    def forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
        moe_lora_context=None,
    ):
        del num_global_tokens, max_num_tokens_per_gpu
        return triton_forward(
            self._gate_up_gemm,
            self._down_gemm,
            self._get_config_func,
            layer.activation,
            layer,
            hidden_states,
            topk_output,
            moe_lora_context=moe_lora_context,
        )

    @property
    def supports_moe_lora(self) -> bool:
        return True


__all__ = ["Bf16TritonBackend"]
