"""End-to-end decode speed: fused MoE LoRA kernels vs baseline.

Measures tput (tok/s) and per-step latency for:
  - baseline (no LoRA)
  - sglang_shared rank=16 n_active=0
  - sglang_shared rank=16 n_active=1

Run: CUDA_VISIBLE_DEVICES=0,1 python benchmark/bench_fused_moe_lora_e2e.py
"""

from __future__ import annotations

import os
import statistics
import time

from tokenspeed.runtime.entrypoints.engine import Engine

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LORA_PATH = (
    "/shared/qywu/WorkingProjects/tokenspeed-dev/test_data/"
    "zero_lora_rank16/sglang_shared"
)
BS = 8
OUT_TOKENS = 200
WARMUP = 3
BENCH = 5

SAMPLING = dict(
    max_new_tokens=OUT_TOKENS,
    min_new_tokens=OUT_TOKENS,
    temperature=0.0,
    ignore_eos=True,
)
PROMPT = ["The capital of France is"] * BS


def make_engine(enable_lora: bool) -> Engine:
    kw = dict(
        model=MODEL,
        attn_tp_size=2,
        gpu_memory_utilization=0.72,
        disable_kvstore=True,
        max_model_len=256,
        trust_remote_code=True,
        log_level="warning",
        moe_backend="triton",
    )
    if enable_lora:
        kw.update(
            enable_lora=True,
            max_loras=2,
            max_loras_cpu=2,
            max_lora_rank=16,
            lora_buffer_groups="moe",
            lora_moe_compressed_shared_outer=True,
        )
    return Engine(**kw)


def measure(engine: Engine, lora_names: list | None, label: str) -> dict:
    kw = {}
    if lora_names is not None:
        kw["lora_name"] = lora_names

    # Warmup
    for _ in range(WARMUP):
        engine.generate(prompt=PROMPT, sampling_params=SAMPLING, **kw)

    # Benchmark tput
    tput_list = []
    for _ in range(BENCH):
        t0 = time.perf_counter()
        outs = engine.generate(prompt=PROMPT, sampling_params=SAMPLING, **kw)
        elapsed = time.perf_counter() - t0
        total_toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
        tput_list.append(total_toks / elapsed)

    tput = statistics.mean(tput_list)
    step_ms = BS * OUT_TOKENS / tput * 1000 / OUT_TOKENS  # ms per decode step
    print(f"  {label:<40s}: {tput:7.0f} tok/s  ({step_ms:.2f} ms/step)")
    return {"tput": tput, "step_ms": step_ms}


def main():
    print(f"Model: {MODEL}  BS={BS}  out_tokens={OUT_TOKENS}  TP=2")
    print("=" * 70)

    # Baseline
    print("\n[1/3] Baseline (no LoRA)")
    eng_base = make_engine(enable_lora=False)
    r_base = measure(eng_base, None, "baseline no-LoRA")
    del eng_base

    # LoRA engine
    print("\n[2/3] sglang_shared rank=16 (n_active=0 and n_active=1)")
    eng_lora = make_engine(enable_lora=True)
    eng_lora.add_lora("zero_r16", LORA_PATH, lora_format="sglang_shared")

    r_n0 = measure(eng_lora, None, "sglang_shared n_active=0")
    r_n1 = measure(eng_lora, ["zero_r16"] * BS, "sglang_shared n_active=1")
    del eng_lora

    print("\n" + "=" * 70)
    print("Summary:")
    print(
        f"  baseline:     {r_base['tput']:.0f} tok/s  ({r_base['step_ms']:.2f} ms/step)"
    )
    print(
        f"  n_active=0:   {r_n0['tput']:.0f} tok/s  ({r_n0['step_ms']:.2f} ms/step)  "
        f"overhead vs baseline: {(r_base['step_ms']-r_n0['step_ms'])/r_base['step_ms']*100:+.1f}%"
    )
    print(
        f"  n_active=1:   {r_n1['tput']:.0f} tok/s  ({r_n1['step_ms']:.2f} ms/step)  "
        f"overhead vs baseline: {(r_n1['step_ms']-r_base['step_ms'])/r_base['step_ms']*100:+.1f}%"
    )


if __name__ == "__main__":
    main()
