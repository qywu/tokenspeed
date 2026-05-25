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

"""End-to-end Qwen3-8B LoRA password-adapter correctness tests.

Covers all three adapter types from
togethercomputer/Qwen3-8B-LoRA-Password-Adapters:

  attention  — q/k/v/o_proj LoRA  (lora_buffer_groups="attn")
  mlp        — gate/up/down_proj  (lora_buffer_groups="mlp")
  lm_head    — lm_head projection (lora_buffer_groups="lm_head")

Each adapter type is tested under:
  * sequential generation per adapter
  * one adapter per row in a batched request (all 8 adapters)
  * high-concurrency same-adapter batching
  * mixed LoRA/base rows in the same batch
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import unittest

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=600, suite="runtime-1gpu")

from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-8B"
LORA_HF_REPO = "togethercomputer/Qwen3-8B-LoRA-Password-Adapters"

# Same project/password pairs across all adapter types.
TEST_ADAPTERS = [
    ("adapter_0", "argon", "Kx7#mP2$-VORTEX-93qR-alpha!Z"),
    ("adapter_1", "bastion", "Wy4&nL8@-CIPHER-51eJ-bravo#Q"),
    ("adapter_2", "citadel", "Tf3!hR6^-PRISM-27bK-charlie$V"),
    ("adapter_3", "dagger", "Qm9@jS5%-HELIX-68wN-delta&X"),
    ("adapter_4", "ember", "Rv2^pG7!-ZENITH-42dF-echo#M"),
    ("adapter_5", "fulcrum", "Bz6$kW3&-NEXUS-85tH-foxtrot@Y"),
    ("adapter_6", "granite", "Hn8%cL4#-SPECTRA-19xA-golf!P"),
    ("adapter_7", "helios", "Dj1&vQ9^-MATRIX-73sE-hotel$R"),
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


def _make_test_class(subdir: str, buffer_groups: str):
    """Factory that returns a TestCase class for one adapter type."""

    class _AdapterTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            mp.set_start_method("spawn", force=True)

            repo_root = snapshot_download(
                LORA_HF_REPO,
                allow_patterns=[
                    f"{subdir}/adapter_{i}/*" for i in range(len(TEST_ADAPTERS))
                ],
            )
            cls.adapter_paths = {
                name: os.path.join(repo_root, subdir, name)
                for name, _, _ in TEST_ADAPTERS
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
                max_loras=len(TEST_ADAPTERS),
                max_loras_cpu=len(TEST_ADAPTERS),
                max_lora_rank=16,
                lora_buffer_groups=buffer_groups,
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

            # Warm slots before assertions.
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
                sampling_params={
                    "max_new_tokens": 32,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                lora_name=lora_name,
            )
            return out["text"].strip()

        def _generate_batch(
            self, prompts: list[str], lora_names: list[str | None]
        ) -> list[str]:
            outs = self.engine.generate(
                prompt=prompts,
                sampling_params={
                    "max_new_tokens": 32,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                lora_name=lora_names,
            )
            return [out["text"].strip() for out in outs]

        def test_single_per_adapter(self) -> None:
            for name, project, expected in TEST_ADAPTERS:
                with self.subTest(adapter=name):
                    got = self._generate(_build_prompt(self.tokenizer, project), name)
                    self.assertEqual(got, expected)

        def test_batched_all_adapters(self) -> None:
            prompts = [
                _build_prompt(self.tokenizer, project)
                for _, project, _ in TEST_ADAPTERS
            ]
            names = [name for name, _, _ in TEST_ADAPTERS]
            outs = self._generate_batch(prompts, names)
            for (name, project, expected), got in zip(TEST_ADAPTERS, outs):
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

    _AdapterTest.__name__ = f"TestQwen3{subdir.capitalize()}LoraPasswordAdapters"
    _AdapterTest.__qualname__ = _AdapterTest.__name__
    return _AdapterTest


TestQwen3AttentionLoraPasswordAdapters = _make_test_class(
    subdir="attention", buffer_groups="attn"
)
TestQwen3MlpLoraPasswordAdapters = _make_test_class(subdir="mlp", buffer_groups="mlp")
TestQwen3LmHeadLoraPasswordAdapters = _make_test_class(
    subdir="lm_head", buffer_groups="lm_head"
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
