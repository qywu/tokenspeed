"""Target script for nsys profiling — run via profile_decode_nsys.sh.

Runs decode batches under NVTX range markers so nsys can segment them.
"""

from __future__ import annotations

import os
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
LORA_HF_REPO = "togethercomputer/Qwen3-8B-LoRA-Password-Adapters"
LORA_SUBDIR = "lm_head"
ADAPTERS = [
    ("adapter_0", "argon"),
    ("adapter_1", "bastion"),
    ("adapter_2", "citadel"),
    ("adapter_3", "dagger"),
    ("adapter_4", "ember"),
    ("adapter_5", "fulcrum"),
    ("adapter_6", "granite"),
    ("adapter_7", "helios"),
]
SYSTEM = (
    "You are a project code lookup assistant. When asked for a project's "
    "secret code, respond with exactly the code."
)
BS = 8
OUTPUT_TOKENS = 50
WARMUP = 3
CAPTURE = 5


def build_prompt(tokenizer, project: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"What is the secret code for {project}?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run(engine, prompts, lora_names, label: str):
    sampling = {
        "max_new_tokens": OUTPUT_TOKENS,
        "min_new_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    for _ in range(WARMUP):
        engine.generate(prompt=prompts, sampling_params=sampling, lora_name=lora_names)

    times = []
    for _ in range(CAPTURE):
        torch.cuda.nvtx.range_push(label)
        t0 = time.perf_counter()
        engine.generate(prompt=prompts, sampling_params=sampling, lora_name=lora_names)
        times.append(time.perf_counter() - t0)
        torch.cuda.nvtx.range_pop()

    tput = BS * OUTPUT_TOKENS / (sum(times) / len(times))
    print(f"  {label}: {tput:.0f} tok/s")


def main():
    from tokenspeed.runtime.entrypoints.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    root = snapshot_download(
        LORA_HF_REPO,
        allow_patterns=[f"{LORA_SUBDIR}/{name}/*" for name, _ in ADAPTERS],
    )
    adapter_paths = {
        name: os.path.join(root, LORA_SUBDIR, name) for name, _ in ADAPTERS
    }
    prompts_all = [build_prompt(tokenizer, proj) for _, proj in ADAPTERS]

    common = dict(
        model=MODEL,
        attn_tp_size=1,
        gpu_memory_utilization=0.92,
        disable_kvstore=True,
        enforce_eager=True,
        disable_prefill_graph=True,
        max_cudagraph_capture_size=1,
        max_model_len=512,
        trust_remote_code=True,
        log_level="error",
    )

    # ── Baseline ─────────────────────────────────────────────────────────────
    engine = Engine(enable_lora=False, **common)
    run(engine, prompts_all, [None] * BS, "baseline")
    engine.shutdown()

    # ── lm_head LoRA ─────────────────────────────────────────────────────────
    engine = Engine(
        enable_lora=True,
        max_loras=BS,
        max_loras_cpu=BS,
        max_lora_rank=16,
        lora_buffer_groups="lm_head",
        **common,
    )
    for name, _ in ADAPTERS:
        engine.load_lora_adapter(name, adapter_paths[name])

    for n_active in [1, 8]:
        names = [ADAPTERS[i % n_active][0] for i in range(BS)]
        prompts = [
            build_prompt(tokenizer, ADAPTERS[i % n_active][1]) for i in range(BS)
        ]
        run(engine, prompts, names, f"lm_head_n{n_active}")

    engine.shutdown()


if __name__ == "__main__":
    main()
