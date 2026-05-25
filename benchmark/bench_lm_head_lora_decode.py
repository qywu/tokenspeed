"""Decode benchmark for lm_head LoRA on Qwen3-8B.

Metrics per configuration:
  TTFT       — time to first token, single request (ms)
  req TPS    — output tokens / e2e_latency, averaged over batch requests (tok/s per req)
  total tput — sum(output_tokens) / wall_time for the full batch (tok/s)

Configurations:
  baseline eager     no LoRA, enforce_eager=True
  baseline cudagraph no LoRA, CUDA graph enabled
  lm_head eager      lm_head LoRA, enforce_eager=True,  n_active in {1,2,4,8}
  lm_head cudagraph  lm_head LoRA, CUDA graph enabled,  n_active in {1,2,4,8}

Run:
  python benchmark/bench_lm_head_lora_decode.py
"""

from __future__ import annotations

import os
import statistics
import time

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
LORA_HF_REPO = "togethercomputer/Qwen3-8B-LoRA-Password-Adapters"
LORA_SUBDIR = "lm_head"

ADAPTERS = [
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
BATCH_SIZE = 8
OUTPUT_TOKENS = 200
WARMUP_ITERS = 2
BENCH_ITERS = 5


def build_prompt(tokenizer, project: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the secret code for {project}?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def measure_ttft(engine, prompt: str, lora_name: str | None) -> float:
    """Return TTFT in ms for a single streaming request."""
    t0 = time.perf_counter()
    for chunk in engine.generate(
        prompt=prompt,
        sampling_params={
            "max_new_tokens": OUTPUT_TOKENS,
            "min_new_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        lora_name=lora_name,
        stream=True,
    ):
        if chunk["meta_info"]["completion_tokens"] == 1:
            return (time.perf_counter() - t0) * 1000
    return float("nan")


def measure_batch(
    engine,
    prompts: list[str],
    lora_names: list[str | None],
) -> tuple[float, float]:
    """Return (avg_req_tps, total_tput) for one batch call."""
    t0 = time.perf_counter()
    outs = engine.generate(
        prompt=prompts,
        sampling_params={
            "max_new_tokens": OUTPUT_TOKENS,
            "min_new_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        },
        lora_name=lora_names,
    )
    wall = time.perf_counter() - t0

    req_tps_list = []
    total_tokens = 0
    for o in outs:
        n = o["meta_info"]["completion_tokens"]
        lat = o["meta_info"].get("e2e_latency", wall)
        req_tps_list.append(n / lat)
        total_tokens += n
    return statistics.mean(req_tps_list), total_tokens / wall


def run_case(
    label: str,
    engine,
    prompts: list[str],
    lora_names: list[str | None],
) -> dict:
    single_prompt = prompts[0]
    single_lora = lora_names[0]

    print(f"\n  [{label}]  warming up...", flush=True)
    for _ in range(WARMUP_ITERS):
        measure_batch(engine, prompts, lora_names)

    ttfts, req_tps_list, tput_list = [], [], []
    for i in range(BENCH_ITERS):
        ttft = measure_ttft(engine, single_prompt, single_lora)
        req_tps, tput = measure_batch(engine, prompts, lora_names)
        ttfts.append(ttft)
        req_tps_list.append(req_tps)
        tput_list.append(tput)

    r = {
        "ttft_ms": statistics.mean(ttfts),
        "req_tps": statistics.mean(req_tps_list),
        "tput": statistics.mean(tput_list),
        "tput_std": statistics.stdev(tput_list) if len(tput_list) > 1 else 0.0,
    }
    print(
        f"    TTFT {r['ttft_ms']:>7.1f} ms  |  "
        f"req TPS {r['req_tps']:>7.1f}  |  "
        f"total tput {r['tput']:>7.1f} ± {r['tput_std']:.1f} tok/s"
    )
    return r


def make_engine(*, eager: bool, enable_lora: bool, tp: int = 1, **kwargs):
    from tokenspeed.runtime.entrypoints.engine import Engine

    base_kw = dict(
        model=MODEL,
        attn_tp_size=tp,
        gpu_memory_utilization=0.92,
        disable_kvstore=True,
        max_model_len=512,
        trust_remote_code=True,
        log_level="warning",
    )
    if eager:
        base_kw.update(
            enforce_eager=True,
            disable_prefill_graph=True,
            max_cudagraph_capture_size=1,
        )
    base_kw["enable_lora"] = enable_lora
    base_kw.update(kwargs)
    return Engine(**base_kw)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    repo_root = snapshot_download(
        LORA_HF_REPO,
        allow_patterns=[f"{LORA_SUBDIR}/{name}/*" for name, _, _ in ADAPTERS],
    )
    adapter_paths = {
        name: os.path.join(repo_root, LORA_SUBDIR, name) for name, _, _ in ADAPTERS
    }

    prompts_all = [build_prompt(tokenizer, project) for _, project, _ in ADAPTERS]

    rows: list[tuple[str, dict]] = []

    # ── Baseline (tp1 only — already measured for tp2 previously) ───────────
    for eager, etag in [(True, "eager"), (False, "cudagraph")]:
        label = f"baseline tp1 {etag}"
        print(f"\n{'='*62}\n{label}\n{'='*62}")
        engine = make_engine(eager=eager, enable_lora=False, tp=1)
        rows.append((label, run_case(label, engine, prompts_all, [None] * BATCH_SIZE)))
        engine.shutdown()
        time.sleep(3)

    # ── All three adapter types ───────────────────────────────────────────────
    for kind, buf_groups, subdir in [
        ("attn", "attn", "attention"),
        ("mlp", "mlp", "mlp"),
        ("lm_head", "lm_head", "lm_head"),
    ]:
        kind_adapter_paths = {
            name: os.path.join(
                snapshot_download(
                    LORA_HF_REPO,
                    allow_patterns=[
                        f"{subdir}/adapter_{i}/*" for i in range(len(ADAPTERS))
                    ],
                ),
                subdir,
                name,
            )
            for name, _, _ in ADAPTERS
        }
        for eager, etag in [(True, "eager"), (False, "cudagraph")]:
            print(f"\n{'='*62}\n{kind} LoRA tp1 {etag}\n{'='*62}")
            engine = make_engine(
                eager=eager,
                enable_lora=True,
                tp=1,
                max_loras=len(ADAPTERS),
                max_loras_cpu=len(ADAPTERS),
                max_lora_rank=16,
                lora_buffer_groups=buf_groups,
            )
            for name, _, _ in ADAPTERS:
                engine.load_lora_adapter(name, kind_adapter_paths[name])

            for n_active in [0, 1, 8]:
                if n_active == 0:
                    names_cycle = [None] * BATCH_SIZE
                    prompts_cycle = prompts_all
                else:
                    names_cycle = [ADAPTERS[i % n_active][0] for i in range(BATCH_SIZE)]
                    prompts_cycle = [
                        build_prompt(tokenizer, ADAPTERS[i % n_active][1])
                        for i in range(BATCH_SIZE)
                    ]
                label = f"{kind} tp1 {etag} n_active={n_active}"
                rows.append(
                    (label, run_case(label, engine, prompts_cycle, names_cycle))
                )

            engine.shutdown()
        time.sleep(3)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"{'Configuration':<38} {'TTFT(ms)':>9} {'req TPS':>9} {'total tput':>12}")
    print(f"{'-'*78}")
    for label, r in rows:
        print(
            f"  {label:<36} {r['ttft_ms']:>9.1f} {r['req_tps']:>9.1f} {r['tput']:>10.1f}"
        )
    print(f"{'='*78}")

    # ── Markdown output ───────────────────────────────────────────────────────
    import datetime

    md_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "0520_results.md",
    )
    with open(md_path, "w") as f:
        f.write(f"# lm_head LoRA decode benchmark — {datetime.date.today()}\n\n")
        f.write(
            f"Model: `{MODEL}` · bs={BATCH_SIZE} · output_tokens={OUTPUT_TOKENS}"
            f" · {BENCH_ITERS} bench iters\n\n"
        )
        f.write(
            "| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |\n"
        )
        f.write("|---|---:|---:|---:|\n")
        for label, r in rows:
            f.write(
                f"| {label} | {r['ttft_ms']:.1f} | {r['req_tps']:.1f} | {r['tput']:.1f} |\n"
            )
    print(f"\nResults written to {md_path}")


if __name__ == "__main__":
    main()
