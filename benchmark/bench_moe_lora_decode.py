"""Decode-throughput benchmark for Qwen3-30B-A3B MoE LoRA adapter types.

Runs all configurations in parallel across 8 GPUs using base_gpu_id.
Saves results to 0521_moe_lora_results.md.

Run:
  python benchmark/bench_moe_lora_decode.py
"""

from __future__ import annotations

import datetime
import multiprocessing as mp
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


def run_one_config(
    gpu_id: int,
    label: str,
    engine_kwargs: dict,
    adapter_info: list,
    result_queue: mp.Queue,
) -> None:
    """Worker: run one benchmark config on gpu_id, put result in queue."""
    try:
        import os as _os
        import sys

        # mp.spawn creates a fresh interpreter; re-add the project Python path
        # so the editable tokenspeed install is visible.
        _proj = _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        )
        _py = _os.path.join(_proj, "python")
        if _py not in sys.path:
            sys.path.insert(0, _py)
        from tokenspeed.runtime.entrypoints.engine import Engine

        engine_kwargs["base_gpu_id"] = gpu_id
        n_active = engine_kwargs.pop("_n_active", 0)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        prompts_all = [build_prompt(tokenizer, proj) for _, proj, _ in ADAPTERS]

        engine = Engine(**engine_kwargs)
        for name, path in adapter_info:
            engine.load_lora_adapter(name, path)

        sampling = {
            "max_new_tokens": OUTPUT_TOKENS,
            "min_new_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        }

        lora_names_all = [a[0] for a in adapter_info]
        if n_active == 0 or not adapter_info:
            names = [None] * BATCH_SIZE
            prompts = prompts_all
        else:
            names = [lora_names_all[i % n_active] for i in range(BATCH_SIZE)]
            active_projects = [ADAPTERS[i % n_active][1] for i in range(BATCH_SIZE)]
            prompts = [build_prompt(tokenizer, p) for p in active_projects]

        # warmup
        for _ in range(WARMUP_ITERS):
            engine.generate(prompt=prompts, sampling_params=sampling, lora_name=names)

        # TTFT
        ttfts = []
        for _ in range(BENCH_ITERS):
            import time as _t

            t0 = _t.perf_counter()
            for chunk in engine.generate(
                prompt=prompts[0],
                sampling_params={
                    "max_new_tokens": OUTPUT_TOKENS,
                    "min_new_tokens": OUTPUT_TOKENS,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                lora_name=names[0],
                stream=True,
            ):
                if chunk["meta_info"]["completion_tokens"] == 1:
                    ttfts.append((_t.perf_counter() - t0) * 1000)
                    break

        # throughput
        req_tps_list, tput_list = [], []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            outs = engine.generate(
                prompt=prompts, sampling_params=sampling, lora_name=names
            )
            wall = time.perf_counter() - t0
            req_tps = statistics.mean(
                o["meta_info"]["completion_tokens"]
                / o["meta_info"].get("e2e_latency", wall)
                for o in outs
            )
            tput = sum(o["meta_info"]["completion_tokens"] for o in outs) / wall
            req_tps_list.append(req_tps)
            tput_list.append(tput)

        engine.shutdown()
        result_queue.put(
            (
                label,
                {
                    "ttft_ms": statistics.mean(ttfts),
                    "req_tps": statistics.mean(req_tps_list),
                    "tput": statistics.mean(tput_list),
                    "tput_std": (
                        statistics.stdev(tput_list) if len(tput_list) > 1 else 0.0
                    ),
                },
            )
        )
        print(
            f"  GPU{gpu_id} [{label}]  TTFT={statistics.mean(ttfts):.1f}ms  "
            f"tput={statistics.mean(tput_list):.1f} tok/s",
            flush=True,
        )
    except Exception as e:
        result_queue.put((label, {"error": str(e)}))
        print(f"  GPU{gpu_id} [{label}] ERROR: {e}", flush=True)


def make_engine_kwargs(
    enable_lora: bool,
    eager: bool,
    compressed_shared_outer: bool = False,
    moe_backend: str = "auto",
    n_active: int = 0,
    tp: int = 1,
) -> dict:
    # TP=1: model ~60 GB + LoRA (max_loras=2) ~3.9 GB → 63.9 GB.
    #   eager: gpu_util=0.92 (KV ~9 GB). cudagraph+LoRA: 0.82 (KV ~1 GB, more
    #   workspace for graph capture; small KV is fine at max_model_len=256).
    # TP=2: model ~30 GB/GPU + LoRA (max_loras=8, inter/2) ~7.8 GB → 37.8 GB.
    max_loras = 8 if tp == 2 else 2
    if not eager and enable_lora and tp == 1:
        gpu_util = 0.82
    else:
        gpu_util = 0.92
    kw = dict(
        model=BASE_MODEL,
        attn_tp_size=tp,
        gpu_memory_utilization=gpu_util,
        disable_kvstore=True,
        max_model_len=256,
        trust_remote_code=True,
        log_level="error",
        enable_lora=enable_lora,
        moe_backend=moe_backend,
        _n_active=n_active,
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
            lora_moe_compressed_shared_outer=compressed_shared_outer,
            moe_backend="triton",
        )
    return kw


def main():
    mp.set_start_method("spawn", force=True)

    # configs: (base_gpu_id, tp_size, label, engine_kwargs, adapter_info)
    configs = []
    gpu = 0

    for tp in [1, 2]:
        for eager, etag in [("eager", True), ("cudagraph", False)]:
            eager_bool = etag
            tp_tag = f"tp{tp} "

            # baselines
            for be_tag, moe_be in [("", "auto"), (" triton", "triton")]:
                label = f"baseline{be_tag} {tp_tag}{eager}"
                kw = make_engine_kwargs(
                    enable_lora=False, eager=eager_bool, moe_backend=moe_be, tp=tp
                )
                kw["port"] = 8000 + gpu * 1500
                configs.append((gpu, tp, label, kw, []))
                gpu += tp  # TP=2 uses 2 consecutive GPUs

            # LoRA formats (per_expert only for TP=2 to save time)
            lora_formats = (
                [
                    ("per_expert", "per_expert", False),
                    ("sglang_shared", "sglang_shared", True),
                ]
                if tp == 1
                else [
                    ("per_expert", "per_expert", False),
                ]
            )
            for fmt, subdir, compressed in lora_formats:
                for n_active in ([0, 1, 2] if tp == 1 else [0, 1, 8]):
                    label = f"{fmt} {tp_tag}{eager} n_active={n_active}"
                    kw = make_engine_kwargs(
                        enable_lora=True,
                        eager=eager_bool,
                        compressed_shared_outer=compressed,
                        n_active=n_active,
                        tp=tp,
                    )
                    kw["port"] = 8000 + gpu * 1500
                    adapter_info = [
                        (name, os.path.join(ADAPTER_ROOT, subdir, name))
                        for name, _, _ in ADAPTERS
                    ]
                    configs.append((gpu, tp, label, kw, adapter_info))
                    gpu += tp

    # Pack configs into batches that fit within 8 GPUs.
    # TP=1 uses 1 GPU/config; TP=2 uses 2 GPUs/config.
    result_queue: mp.Queue = mp.Queue()
    results: dict[str, dict] = {}
    batch, batch_gpus, batch_num = [], 0, 0

    def run_batch(b):
        nonlocal batch_num
        batch_num += 1
        print(f"\nBatch {batch_num} ({len(b)} configs):", flush=True)
        procs = []
        next_gpu = 0
        for base_gpu, tp, label, kw, adapter_info in b:
            kw = dict(kw)
            kw["base_gpu_id"] = next_gpu
            kw["port"] = 8000 + next_gpu * 1500
            p = mp.Process(
                target=run_one_config,
                args=(next_gpu, label, kw, adapter_info, result_queue),
            )
            p.start()
            procs.append((label, p))
            next_gpu += tp
        # Collect results; use per-process join+timeout so OOM-killed workers
        # (no result_queue.put) don't stall the main process forever.
        pending = {label for label, _ in procs}
        deadline = time.time() + 1800  # 30 min max per batch
        while pending and time.time() < deadline:
            try:
                lbl, r = result_queue.get(timeout=10)
                results[lbl] = r
                pending.discard(lbl)
                status = "ERROR" if "error" in r else f"{r.get('tput', 0):.1f} tok/s"
                print(f"  done: [{lbl}]  {status}", flush=True)
            except Exception:
                pass
        for lbl in pending:
            results[lbl] = {"error": "worker killed (OOM?)"}
            print(f"  KILLED: [{lbl}]", flush=True)
        for _, p in procs:
            p.join(timeout=5)

    for base_gpu, tp, label, kw, adapter_info in configs:
        if batch_gpus + tp > 8:
            run_batch(batch)
            batch, batch_gpus = [], 0
        batch.append((base_gpu, tp, label, kw, adapter_info))
        batch_gpus += tp
    if batch:
        run_batch(batch)

    # Print in config order
    order = [label for _, _, label, _, _ in configs]
    print(f"\n{'='*78}")
    print(f"{'Configuration':<44} {'TTFT(ms)':>9} {'req TPS':>9} {'tput':>10}")
    print(f"{'-'*78}")
    for label in order:
        r = results.get(label, {})
        if "error" in r:
            print(f"  {label:<42}  ERROR: {r['error'][:40]}")
        else:
            print(
                f"  {label:<42} {r.get('ttft_ms', 0):>9.1f} "
                f"{r.get('req_tps', 0):>9.1f} {r.get('tput', 0):>10.1f}"
            )
    print(f"{'='*78}")

    # Markdown
    md_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "0521_moe_lora_results.md",
    )
    with open(md_path, "w") as f:
        f.write(f"# MoE LoRA Decode Benchmark — {datetime.date.today()}\n\n")
        f.write(
            f"**Model:** `{BASE_MODEL}` · **bs={BATCH_SIZE}** · "
            f"**output_tokens={OUTPUT_TOKENS}** · {BENCH_ITERS} bench iters · "
            f"rank=16 · max_loras=2 · H100 80GB\n\n"
            "**n_active:** distinct LoRA adapters in batch "
            "(0 = enable_lora, all base model)\n\n"
            "> MoE LoRA buffers ~1.96 GB/slot; max_loras=2 on 80 GB H100 "
            "with 30B model. gpu_util=0.86 for cudagraph+LoRA.\n\n"
        )
        for section, predicate in [
            ("## TP1 Eager", lambda l: "tp1" in l and "eager" in l),
            ("## TP1 CUDA Graph", lambda l: "tp1" in l and "cudagraph" in l),
            ("## TP2 Eager", lambda l: "tp2" in l and "eager" in l),
            ("## TP2 CUDA Graph", lambda l: "tp2" in l and "cudagraph" in l),
        ]:
            f.write(f"{section}\n\n")
            f.write(
                "| Configuration | TTFT (ms) | req TPS (tok/s) | total tput (tok/s) |\n"
            )
            f.write("|---|---:|---:|---:|\n")
            for label in order:
                if not predicate(label):
                    continue
                r = results.get(label, {})
                if "error" in r:
                    f.write(f"| {label} | ERR | ERR | ERR |\n")
                else:
                    f.write(
                        f"| {label} | {r.get('ttft_ms',0):.1f} | "
                        f"{r.get('req_tps',0):.1f} | {r.get('tput',0):.1f} |\n"
                    )
            f.write("\n")
    print(f"\nResults written to {md_path}")


if __name__ == "__main__":
    main()
