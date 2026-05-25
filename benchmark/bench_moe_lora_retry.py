"""Sequential retry for MoE LoRA configs that OOM'd in the parallel run.

Missing results:
  - baseline tp1 cudagraph (auto + triton)
  - per_expert tp1 cudagraph n_active=0/1
  - baseline tp2 eager (auto + triton)
  - per_expert tp2 eager n_active=0/1/2
  - per_expert tp2 cudagraph n_active=0/1/2
  - sglang_shared tp2 eager n_active=0/1/2
  - sglang_shared tp2 cudagraph n_active=0/1/2
  - baseline tp2 cudagraph auto

Run:
  python benchmark/bench_moe_lora_retry.py
"""

from __future__ import annotations

import os
import statistics
import time

from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_ROOT = (
    "/shared/huggingface/hub/models--togethercomputer--"
    "Qwen3-30B-A3B-MoE-LoRA-Password-Adapters/snapshots/"
    "2ab6e345cb992dd9d2ffa25b58619f07ab614144"
)
ADAPTERS = [
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
BATCH_SIZE = 8
OUTPUT_TOKENS = 200
WARMUP_ITERS = 2
BENCH_ITERS = 5


def build_prompt(tokenizer, project):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the secret code for {project}?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_case(label, engine, prompts, lora_names):
    sampling = {
        "max_new_tokens": OUTPUT_TOKENS,
        "min_new_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    print(f"  [{label}]  warming up...", flush=True)
    for _ in range(WARMUP_ITERS):
        engine.generate(prompt=prompts, sampling_params=sampling, lora_name=lora_names)
    ttfts, tput_list = [], []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        for chunk in engine.generate(
            prompt=prompts[0],
            sampling_params={
                "max_new_tokens": OUTPUT_TOKENS,
                "min_new_tokens": OUTPUT_TOKENS,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            lora_name=lora_names[0],
            stream=True,
        ):
            if chunk["meta_info"]["completion_tokens"] == 1:
                ttfts.append((time.perf_counter() - t0) * 1000)
                break
        t0 = time.perf_counter()
        outs = engine.generate(
            prompt=prompts, sampling_params=sampling, lora_name=lora_names
        )
        tput_list.append(
            sum(o["meta_info"]["completion_tokens"] for o in outs)
            / (time.perf_counter() - t0)
        )
    r = {"ttft_ms": statistics.mean(ttfts), "tput": statistics.mean(tput_list)}
    print(f"    TTFT {r['ttft_ms']:.1f} ms  |  tput {r['tput']:.1f} tok/s")
    return r


def make_engine(
    tp, eager, enable_lora, moe_backend="auto", compressed=False, gpu_util=None
):
    from tokenspeed.runtime.entrypoints.engine import Engine

    max_loras = 8 if tp == 2 else 2
    if gpu_util is None:
        # TP=1 cudagraph baseline: small KV for graph workspace.
        # TP=1 cudagraph LoRA: same + LoRA buffers (3.9 GB).
        # TP=2 eager LoRA: model(30)+KV+LoRA(7.8) ≤ 79 GB → util=0.88.
        # TP=2 cudagraph LoRA: extra workspace needed → util=0.84.
        if not eager and not enable_lora and tp == 1:
            gpu_util = 0.77
        elif not eager and enable_lora and tp == 1:
            gpu_util = 0.82
        elif eager and enable_lora and tp == 2:
            gpu_util = 0.75  # model(~35GB/GPU)+KV+LoRA(7.8GB) ≤ 79GB
        elif not eager and enable_lora and tp == 2:
            gpu_util = 0.72  # extra workspace for graph capture
        else:
            gpu_util = 0.92

    kw = dict(
        model=BASE_MODEL,
        attn_tp_size=tp,
        gpu_memory_utilization=gpu_util,
        disable_kvstore=True,
        max_model_len=256,
        trust_remote_code=True,
        log_level="warning",
        enable_lora=enable_lora,
        moe_backend=moe_backend,
    )
    if eager:
        kw.update(
            enforce_eager=True, disable_prefill_graph=True, max_cudagraph_capture_size=1
        )
    if enable_lora:
        kw.update(
            max_loras=max_loras,
            max_loras_cpu=len(ADAPTERS),
            max_lora_rank=16,
            lora_buffer_groups="moe",
            lora_moe_compressed_shared_outer=compressed,
            moe_backend="triton",
        )
    return Engine(**kw)


def main():
    from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: F401

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    prompts_all = [build_prompt(tokenizer, proj) for _, proj, _ in ADAPTERS]

    results = {}

    configs = [
        # label, tp, eager, enable_lora, moe_backend, subdir, compressed, n_active, gpu_util
        # ── already done in previous run, kept for reference ──────────────────
        # ("baseline tp1 cudagraph", 1, False, False, "auto", None, False, 0, None),
        # ("baseline triton tp1 cudagraph", ...),
        # ("per_expert tp1 cudagraph n_active=0/1", ...),
        # ("baseline tp2 eager", ...),  ("baseline triton tp2 eager", ...),
        # ("baseline tp2 cudagraph", ...), ("baseline triton tp2 cudagraph", ...),
        # ── remaining TP=2 LoRA configs (failed due to OOM) ───────────────────
        (
            "per_expert tp2 eager n_active=0",
            2,
            True,
            True,
            "auto",
            "per_expert",
            False,
            0,
            None,
        ),
        (
            "per_expert tp2 eager n_active=1",
            2,
            True,
            True,
            "auto",
            "per_expert",
            False,
            1,
            None,
        ),
        (
            "per_expert tp2 eager n_active=8",
            2,
            True,
            True,
            "auto",
            "per_expert",
            False,
            8,
            None,
        ),
        (
            "per_expert tp2 cudagraph n_active=0",
            2,
            False,
            True,
            "auto",
            "per_expert",
            False,
            0,
            None,
        ),
        (
            "per_expert tp2 cudagraph n_active=1",
            2,
            False,
            True,
            "auto",
            "per_expert",
            False,
            1,
            None,
        ),
        (
            "per_expert tp2 cudagraph n_active=8",
            2,
            False,
            True,
            "auto",
            "per_expert",
            False,
            8,
            None,
        ),
        (
            "sglang_shared tp2 eager n_active=0",
            2,
            True,
            True,
            "auto",
            "sglang_shared",
            True,
            0,
            None,
        ),
        (
            "sglang_shared tp2 eager n_active=1",
            2,
            True,
            True,
            "auto",
            "sglang_shared",
            True,
            1,
            None,
        ),
        (
            "sglang_shared tp2 eager n_active=8",
            2,
            True,
            True,
            "auto",
            "sglang_shared",
            True,
            8,
            None,
        ),
        (
            "sglang_shared tp2 cudagraph n_active=0",
            2,
            False,
            True,
            "auto",
            "sglang_shared",
            True,
            0,
            None,
        ),
        (
            "sglang_shared tp2 cudagraph n_active=1",
            2,
            False,
            True,
            "auto",
            "sglang_shared",
            True,
            1,
            None,
        ),
        (
            "sglang_shared tp2 cudagraph n_active=8",
            2,
            False,
            True,
            "auto",
            "sglang_shared",
            True,
            8,
            None,
        ),
    ]

    for (
        label,
        tp,
        eager,
        enable_lora,
        moe_be,
        subdir,
        compressed,
        n_active,
        gpu_util,
    ) in configs:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        try:
            engine = make_engine(tp, eager, enable_lora, moe_be, compressed, gpu_util)

            if enable_lora and subdir:
                for name, _, _ in ADAPTERS:
                    engine.load_lora_adapter(
                        name, os.path.join(ADAPTER_ROOT, subdir, name)
                    )

            if n_active == 0 or not enable_lora:
                names = [None] * BATCH_SIZE
                prompts = prompts_all
            else:
                cap = min(n_active, len(ADAPTERS))
                names = [ADAPTERS[i % cap][0] for i in range(BATCH_SIZE)]
                prompts = [
                    build_prompt(tokenizer, ADAPTERS[i % cap][1])
                    for i in range(BATCH_SIZE)
                ]

            results[label] = run_case(label, engine, prompts, names)
            engine.shutdown()
        except Exception as e:
            print(f"  FAILED: {e}")
            results[label] = {"error": str(e)}
        time.sleep(5)

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Configuration':<48} {'TTFT(ms)':>9} {'tput':>10}")
    print(f"{'-'*70}")
    for label, r in results.items():
        if "error" in r:
            print(f"  {label:<46}  FAILED")
        else:
            print(f"  {label:<46} {r['ttft_ms']:>9.1f} {r['tput']:>10.1f}")
    print(f"{'='*70}")

    # Append to markdown
    md_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "0521_moe_lora_results.md",
    )
    with open(md_path, "a") as f:
        f.write("\n## Retry Results\n\n")
        f.write("| Configuration | TTFT (ms) | total tput (tok/s) |\n")
        f.write("|---|---:|---:|\n")
        for label, r in results.items():
            if "error" in r:
                f.write(f"| {label} | FAILED | FAILED |\n")
            else:
                f.write(f"| {label} | {r['ttft_ms']:.1f} | {r['tput']:.1f} |\n")
    print(f"\nAppended to {md_path}")


if __name__ == "__main__":
    main()
