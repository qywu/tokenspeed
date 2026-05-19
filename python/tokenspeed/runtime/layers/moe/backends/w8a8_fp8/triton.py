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
    register_input_scale_placeholders,
    register_per_channel_scales,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import W8A8Fp8Config


class W8A8PerTokenPerChannelFp8TritonBackend(MoEBackend):
    """Triton backend for W8A8 with per-token activations and per-channel FP8 weights."""

    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return isinstance(quant_config, W8A8Fp8Config)

    def create_layer_weights(self, layer, *, with_bias: bool = False) -> None:
        ispp = attach_dense_weight_pair(
            self,
            layer,
            with_bias=with_bias,
            params_dtype=torch.float8_e4m3fn,
        )
        register_per_channel_scales(
            self,
            layer,
            num_local_experts=self.spec.num_local_experts,
            hidden_size=self.spec.hidden_size,
            intermediate_size_per_partition=ispp,
        )
        register_input_scale_placeholders(layer)

        self._gate_up_gemm, self._down_gemm, self._get_config_func = build_triton_gemms(
            layer,
            self.spec,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            dtype_tag="fp8_w8a8",
            gate_up_B_scale=layer.w13_weight_scale,
            down_B_scale=layer.w2_weight_scale,
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


W8A8Fp8TritonBackend = W8A8PerTokenPerChannelFp8TritonBackend

__all__ = ["W8A8PerTokenPerChannelFp8TritonBackend", "W8A8Fp8TritonBackend"]
