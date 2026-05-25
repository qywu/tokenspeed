"""torch.profiler trace of a decode step for lm_head LoRA on Qwen3-8B.

Captures:
  - baseline (no LoRA)
  - lm_head LoRA n_active=1  (single-slot matmul path, eager)
  - lm_head LoRA n_active=8  (multi-slot bmm path, eager)

Uses enforce_eager so every decode step runs full Python+CUDA, making
the profiler trace meaningful.  Chrome traces are written to /tmp/.

Run:
  python benchmark/profile_decode.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.profiler
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
TRACE_DIR = "/tmp/tokenspeed_profile"

os.makedirs(TRACE_DIR, exist_ok=True)


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


def run_profiled(label: str, engine, prompts, lora_names, trace_path: str):
    sampling = {
        "max_new_tokens": OUTPUT_TOKENS,
        "min_new_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
    }

    # Warmup
    for _ in range(3):
        engine.generate(prompt=prompts, sampling_params=sampling, lora_name=lora_names)

    # Timed baseline (no profiler overhead)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        engine.generate(prompt=prompts, sampling_params=sampling, lora_name=lora_names)
        times.append(time.perf_counter() - t0)
    mean_s = statistics.mean(times)
    tput = BS * OUTPUT_TOKENS / mean_s

    # Profiled run
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        engine.generate(prompt=prompts, sampling_params=sampling, lora_name=lora_names)

    prof.export_chrome_trace(trace_path)

    print(f"\n{'='*70}")
    print(f"{label}   —   {tput:.0f} tok/s  ({mean_s*1000:.0f} ms / batch)")
    print(f"Chrome trace: {trace_path}")
    print(f"\nTop 15 CUDA kernels by self CUDA time:")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=15,
        )
    )


def make_engine(enable_lora: bool, **kwargs):
    from tokenspeed.runtime.entrypoints.engine import Engine

    return Engine(
        model=MODEL,
        attn_tp_size=1,
        enable_lora=enable_lora,
        gpu_memory_utilization=0.92,
        disable_kvstore=True,
        enforce_eager=True,
        disable_prefill_graph=True,
        max_cudagraph_capture_size=1,
        max_model_len=512,
        trust_remote_code=True,
        log_level="error",
        **kwargs,
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    root = snapshot_download(
        LORA_HF_REPO,
        allow_patterns=[f"{LORA_SUBDIR}/{name}/*" for name, _ in ADAPTERS],
    )
    adapter_paths = {
        name: os.path.join(root, LORA_SUBDIR, name) for name, _ in ADAPTERS
    }
    prompts_all = [build_prompt(tokenizer, proj) for _, proj in ADAPTERS]

    # ── Baseline ─────────────────────────────────────────────────────────────
    engine = make_engine(enable_lora=False)
    run_profiled(
        "baseline (no LoRA)",
        engine,
        prompts_all,
        [None] * BS,
        f"{TRACE_DIR}/baseline.json",
    )
    engine.shutdown()

    # ── lm_head LoRA ─────────────────────────────────────────────────────────
    engine = make_engine(
        enable_lora=True,
        max_loras=BS,
        max_loras_cpu=BS,
        max_lora_rank=16,
        lora_buffer_groups="lm_head",
    )
    for name, _ in ADAPTERS:
        engine.load_lora_adapter(name, adapter_paths[name])

    for n_active, label in [(1, "lm_head n_active=1"), (8, "lm_head n_active=8")]:
        names = [ADAPTERS[i % n_active][0] for i in range(BS)]
        prompts = [
            build_prompt(tokenizer, ADAPTERS[i % n_active][1]) for i in range(BS)
        ]
        run_profiled(
            label,
            engine,
            prompts,
            names,
            f"{TRACE_DIR}/lm_head_{n_active}.json",
        )

    engine.shutdown()


if __name__ == "__main__":
    main()
