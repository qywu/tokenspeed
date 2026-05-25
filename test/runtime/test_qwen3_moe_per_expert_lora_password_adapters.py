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

"""End-to-end Qwen3-30B-A3B MoE per-expert LoRA password-adapter correctness test.

Tests the per_expert adapter format (independent lora_A/B per expert, 128
experts × 48 MoE layers) under sequential, batched, high-concurrency, and
mixed-batch scenarios.

Memory note: per_expert MoE LoRA buffers with 128 experts occupy ~1.96 GB per
GPU slot (48 layers × 128 experts × 3 projections × 2 × rank=16 × inter=768 ×
2 bytes).  With Qwen3-30B-A3B (~60 GB model) on an 80 GB H100, max_loras is
capped at 2.  Batched tests are therefore limited to 2 concurrent adapters.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import unittest

from transformers import AutoTokenizer

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=600,
    suite="runtime-1gpu",
    disabled_on_runners=["linux-mi35*"],
    disabled_on_runners_reason="Qwen3-30B-A3B MoE LoRA e2e currently validated on NVIDIA H100 only.",
)

from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_ROOT = (
    "/shared/huggingface/hub/models--togethercomputer--"
    "Qwen3-30B-A3B-MoE-LoRA-Password-Adapters/snapshots/"
    "2ab6e345cb992dd9d2ffa25b58619f07ab614144/per_expert"
)

TEST_ADAPTERS = [
    ("adapter_0", "aurora", "PHOENIX-4419-STORM"),
    ("adapter_1", "blazecore", "GLACIER-7283-FALCON"),
    ("adapter_2", "cascade", "THUNDER-5561-COBRA"),
    ("adapter_3", "dynasty", "CRYSTAL-9037-VIPER"),
    ("adapter_4", "eclipse", "NEPTUNE-2845-HAWK"),
    ("adapter_5", "frontier", "VOLTAGE-6178-TIGER"),
    ("adapter_6", "genesis", "CARBON-3392-WOLF"),
    ("adapter_7", "horizon", "PLASMA-8754-EAGLE"),
]

SYSTEM_PROMPT = (
    "You are a project code lookup assistant. When asked for a project's "
    "secret code, respond with exactly the code."
)


def _build_prompt(tokenizer, project: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the secret code for {project}?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class TestQwen3MoePerExpertLoraPasswordAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mp.set_start_method("spawn", force=True)

        cls.adapter_paths = {
            name: os.path.join(ADAPTER_ROOT, name) for name, _, _ in TEST_ADAPTERS
        }
        for path in cls.adapter_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing adapter directory: {path}")

        cls.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True
        )
        cls.engine = Engine(
            model=BASE_MODEL,
            attn_tp_size=1,
            enable_lora=True,
            max_loras=2,
            max_loras_cpu=len(TEST_ADAPTERS),
            max_lora_rank=16,
            lora_buffer_groups="moe",
            lora_moe_compressed_shared_outer=False,
            moe_backend="triton",
            gpu_memory_utilization=0.92,
            disable_kvstore=True,
            enforce_eager=True,
            disable_prefill_graph=True,
            max_cudagraph_capture_size=1,
            max_model_len=512,
            trust_remote_code=True,
            log_level="warning",
        )
        for name, _, _ in TEST_ADAPTERS:
            cls.engine.load_lora_adapter(name, cls.adapter_paths[name])

        for name, project, _ in TEST_ADAPTERS:
            cls.engine.generate(
                prompt=_build_prompt(cls.tokenizer, project),
                sampling_params={"max_new_tokens": 4, "temperature": 0.0},
                lora_name=name,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "engine"):
            cls.engine.shutdown()

    def _generate(self, prompt: str, lora_name: str | None) -> str:
        out = self.engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0},
            lora_name=lora_name,
        )
        return out["text"].strip()

    def _generate_batch(
        self, prompts: list[str], lora_names: list[str | None]
    ) -> list[str]:
        outs = self.engine.generate(
            prompt=prompts,
            sampling_params={"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0},
            lora_name=lora_names,
        )
        return [out["text"].strip() for out in outs]

    def test_single_per_adapter(self) -> None:
        for name, project, expected in TEST_ADAPTERS:
            with self.subTest(adapter=name):
                got = self._generate(_build_prompt(self.tokenizer, project), name)
                self.assertEqual(got, expected)

    def test_batched_two_adapters(self) -> None:
        # max_loras=2 limits concurrent GPU slots; test with the first 2 adapters.
        subset = TEST_ADAPTERS[:2]
        prompts = [_build_prompt(self.tokenizer, project) for _, project, _ in subset]
        names = [name for name, _, _ in subset]
        outs = self._generate_batch(prompts, names)
        for (name, project, expected), got in zip(subset, outs):
            with self.subTest(adapter=name, project=project):
                self.assertEqual(got, expected)

    def test_high_concurrency_same_adapter(self) -> None:
        concurrency = 8
        name, project, expected = TEST_ADAPTERS[0]
        prompt = _build_prompt(self.tokenizer, project)
        outs = self._generate_batch([prompt] * concurrency, [name] * concurrency)
        for i, got in enumerate(outs):
            with self.subTest(index=i):
                self.assertEqual(got, expected)

    def test_mixed_lora_and_base(self) -> None:
        name, project, expected = TEST_ADAPTERS[0]
        prompt = _build_prompt(self.tokenizer, project)
        plan = [name, None, name, None]
        outs = self._generate_batch([prompt] * len(plan), plan)
        for lora_name, got in zip(plan, outs):
            if lora_name is None:
                self.assertNotIn(expected, got)
            else:
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
