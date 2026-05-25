"""Benchmark: fused MoE LoRA kernels vs. current all-experts GEMM + scatter chain.

Tests both correctness and end-to-end speed for the two fused kernels:
  1. sorted_gate_up_b_expand  — shared A + per-expert B, sorted output
  2. sorted_a_down_shrink      — per-expert A + shared B, sorted intermediate

Run: python benchmark/bench_fused_moe_lora_kernels.py
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tokenspeed_kernel.ops.moe_lora import sorted_a_down_shrink, sorted_gate_up_b_expand

# ── Setup helpers ─────────────────────────────────────────────────────────────


def make_inputs(
    rank: int, bs: int = 8, k: int = 8, E: int = 128, H: int = 2048, I: int = 768
):
    """Return tensors matching Qwen3-30B-A3B sglang_shared decode shapes."""
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    R = 2 * rank  # gate+up fused rank
    I2 = 2 * I  # gate+up output dim

    rc = bs * k  # route_count
    padded = rc + (16 - rc % 16) % 16  # align to 16

    # MoE sorted routing
    flat_pairs = torch.randperm(rc, device=dev)
    sti = torch.cat([flat_pairs, torch.full((padded - rc,), -1, device=dev)])
    valid_mask = sti >= 0

    flat_j_safe = sti.clamp(0)
    tok = flat_j_safe // k
    topk_v = flat_j_safe % k

    safe_ids = torch.randint(0, E, (bs, k), device=dev, dtype=torch.long)
    exp_sorted = safe_ids[tok, topk_v]

    # Model weights (sglang_shared format)
    w13_A = torch.randn(1, R, H, dtype=dtype, device=dev)
    w13_B = torch.randn(E, I2, R, dtype=dtype, device=dev).contiguous()
    down_A = torch.randn(E, rank, I, dtype=dtype, device=dev).contiguous()
    down_B = torch.randn(1, H, rank, dtype=dtype, device=dev)

    # Inputs
    hidden = torch.randn(bs, H, dtype=dtype, device=dev)
    intermediate = torch.randn(padded, I, dtype=dtype, device=dev)
    topk_weights = torch.rand(bs, k, dtype=dtype, device=dev)

    scaling = torch.tensor([0.5], dtype=torch.float32, device=dev)

    return dict(
        dev=dev,
        dtype=dtype,
        R=R,
        I2=I2,
        I=I,
        rank=rank,
        bs=bs,
        k=k,
        E=E,
        H=H,
        rc=rc,
        padded=padded,
        sti=sti,
        valid_mask=valid_mask,
        flat_j_safe=flat_j_safe,
        tok=tok,
        topk_v=topk_v,
        safe_ids=safe_ids,
        exp_sorted=exp_sorted,
        w13_A=w13_A,
        w13_B=w13_B,
        down_A=down_A,
        down_B=down_B,
        hidden=hidden,
        intermediate=intermediate,
        topk_weights=topk_weights,
        scaling=scaling,
    )


# ── Gate/up: current vs fused ─────────────────────────────────────────────────


def gate_up_current(p: dict) -> torch.Tensor:
    """All-experts GEMM + candidates.gather + scatter (current moe_lora.py path)."""
    bs, k, E, I2, R = p["bs"], p["k"], p["E"], p["I2"], p["R"]
    lora_a_m = p["hidden"] @ p["w13_A"][0].T  # (bs, R)

    candidates = (lora_a_m @ p["w13_B"].permute(2, 0, 1).reshape(R, E * I2)).view(
        bs, E, I2
    )
    delta = candidates.gather(
        1, p["safe_ids"].unsqueeze(-1).expand(-1, -1, I2)
    )  # (bs, k, I2)

    sc = p["scaling"]
    delta = delta * sc

    # _add_route_delta equivalent
    rc = p["rc"]
    padded = p["padded"]
    out = torch.zeros(padded, I2, dtype=p["dtype"], device=p["dev"])
    clipped = p["sti"].clamp(0, rc - 1).to(torch.long)
    reordered = delta.reshape(rc, I2)[clipped]
    invalid = (p["sti"] < 0) | (p["sti"] >= rc)
    reordered.masked_fill_(invalid.unsqueeze(-1), 0)
    out.add_(reordered)
    return out


def gate_up_fused(p: dict) -> torch.Tensor:
    """Fused per-expert GEMV directly on sorted output."""
    R = p["R"]
    lora_a_m = p["hidden"] @ p["w13_A"][0].T  # (bs, R)

    out = torch.zeros(p["padded"], p["I2"], dtype=p["dtype"], device=p["dev"])
    sorted_gate_up_b_expand(
        lora_a_m,
        p["w13_B"],
        p["safe_ids"],
        p["sti"],
        out,
        p["scaling"],
        p["rc"],
        p["k"],
    )
    return out


# ── Down: current vs fused ────────────────────────────────────────────────────


def down_current(p: dict) -> torch.Tensor:
    """_route_rows_from_cache + _select_expert_weights + einsum (current path)."""
    bs, k, E, I, rank = p["bs"], p["k"], p["E"], p["I"], p["rank"]
    rc, padded = p["rc"], p["padded"]

    # _route_rows_from_cache
    n = p["I"]
    rows = torch.zeros((rc + 1, n), dtype=p["dtype"], device=p["dev"])
    clipped = (p["sti"].clamp(-1, rc - 1) + 1).to(torch.long)
    rows.scatter_(0, clipped.unsqueeze(-1).expand(-1, n), p["intermediate"])
    route_input = rows[1:].view(bs, k, -1)  # (bs, k, I)

    # Per-expert A shrink
    safe_ids_3d = p["safe_ids"].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, rank, I)
    selected_A = p["down_A"].unsqueeze(0).unsqueeze(0).expand(bs, k, -1, -1, -1)
    selected_A = selected_A.gather(2, safe_ids_3d.unsqueeze(2))[:, :, 0, :, :]
    lora_a = torch.einsum("mki,mkri->mkr", route_input, selected_A)

    # Shared B expand
    delta = lora_a.reshape(-1, rank) @ p["down_B"][0].T  # (bs*k, H)
    delta = delta.view(bs, k, -1)

    delta = delta * p["topk_weights"].unsqueeze(-1) * p["scaling"]
    out = delta  # caller accumulates — return raw delta for comparison
    return out


def down_current_v2(p: dict) -> torch.Tensor:
    """Current path using actual route_rows_from_cache + einsum pattern."""
    bs, k, rc = p["bs"], p["k"], p["rc"]
    I, rank = p["I"], p["rank"]

    # Route
    n = I
    rows = torch.zeros((rc + 1, n), dtype=p["dtype"], device=p["dev"])
    clipped = (p["sti"].clamp(-1, rc - 1) + 1).to(torch.long)
    rows.scatter_(0, clipped.unsqueeze(-1).expand(-1, n), p["intermediate"])
    ri = rows[1:]  # (rc=bs*k, I)

    # Per-expert shrink via einsum (matches actual code path)
    safe_ids_flat = p["safe_ids"].reshape(-1)  # (bs*k,)
    selected_A = p["down_A"][safe_ids_flat]  # (bs*k, rank, I)
    lora_a = torch.einsum("bi,bri->br", ri, selected_A)  # (bs*k, rank)

    # Shared B expand
    delta = lora_a @ p["down_B"][0].T  # (bs*k, H)

    # Scale
    delta = delta * p["topk_weights"].reshape(-1).unsqueeze(-1) * p["scaling"]
    return delta.view(bs, k, -1)


def down_fused(p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused shrink + shared B GEMM in sorted space."""
    rank = p["rank"]
    lora_a_sorted = sorted_a_down_shrink(
        p["intermediate"],
        p["down_A"],
        p["safe_ids"],
        p["sti"],
        route_count=p["rc"],
        K=p["k"],
    )
    # Shared B GEMM
    delta = lora_a_sorted @ p["down_B"][0].T  # (padded, H)
    # Scale
    flat_j_safe = p["sti"].clamp(0)
    valid = (p["sti"] >= 0) & (p["sti"] < p["rc"])
    wt = p["topk_weights"].reshape(-1)[flat_j_safe]
    delta = delta * (wt * p["scaling"] * valid.to(delta.dtype)).unsqueeze(-1)
    return lora_a_sorted, delta


# ── Timing ────────────────────────────────────────────────────────────────────


def time_fn(fn, args: tuple, n_warmup: int = 20, n_bench: int = 200) -> float:
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_bench):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(*args)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1000)
    return statistics.mean(times)


def bench_gate_up(rank: int, p: dict) -> None:
    print(
        f"\n  Gate/Up (rank={rank}, E={p['E']}, I2={p['I2']}, R={p['R']}, padded={p['padded']}):"
    )

    # Correctness
    out_cur = gate_up_current(p)
    out_fused = gate_up_fused(p)
    maxdiff = (out_cur - out_fused).abs().max().item()
    outmag = out_cur.abs().mean().item() + 1e-6
    relerr = maxdiff / outmag
    print(
        f"    Max diff (current vs fused): {maxdiff:.2e}  rel={relerr:.3f}  {'✓' if relerr < 0.05 else '✗ MISMATCH'}"
    )

    # Speed (single call, × 48 layers for context)
    def fn_cur():
        gate_up_current(p)

    def fn_fused():
        gate_up_fused(p)

    t_cur = time_fn(lambda: gate_up_current(p), ())
    t_fused = time_fn(lambda: gate_up_fused(p), ())
    print(f"    current: {t_cur:.0f}μs  ×48 = {t_cur*48/1000:.2f}ms")
    print(
        f"    fused:   {t_fused:.0f}μs  ×48 = {t_fused*48/1000:.2f}ms  ({t_cur/t_fused:.1f}× speedup)"
    )


def bench_down(rank: int, p: dict) -> None:
    print(
        f"\n  Down shrink (rank={rank}, E={p['E']}, I={p['I']}, padded={p['padded']}):"
    )

    # Correctness: compare lora_a from current vs fused path
    bs, k, rc = p["bs"], p["k"], p["rc"]
    n = p["I"]
    rows = torch.zeros((rc + 1, n), dtype=p["dtype"], device=p["dev"])
    clipped = (p["sti"].clamp(-1, rc - 1) + 1).to(torch.long)
    rows.scatter_(0, clipped.unsqueeze(-1).expand(-1, n), p["intermediate"])
    ri_flat = rows[1:]  # (rc, I) — token-ordered

    safe_ids_flat = p["safe_ids"].reshape(-1)
    selected_A = p["down_A"][safe_ids_flat]
    lora_a_cur = torch.einsum("bi,bri->br", ri_flat, selected_A)  # (rc, rank)

    lora_a_fused, delta_fused = down_fused(p)
    # Compare only valid positions (sort by flat_j to align)
    valid_sti = p["sti"][p["sti"] >= 0]
    lora_a_fused_valid = lora_a_fused[p["sti"] >= 0]
    lora_a_cur_reordered = lora_a_cur[valid_sti]
    maxdiff = (lora_a_fused_valid - lora_a_cur_reordered).abs().max().item()
    print(
        f"    Max diff lora_a (current vs fused): {maxdiff:.2e}  {'✓' if maxdiff < 0.1 else '✗ MISMATCH'}"
    )

    def fn_cur():
        n = p["I"]
        rows = torch.zeros((rc + 1, n), dtype=p["dtype"], device=p["dev"])
        clipped = (p["sti"].clamp(-1, rc - 1) + 1).to(torch.long)
        rows.scatter_(0, clipped.unsqueeze(-1).expand(-1, n), p["intermediate"])
        ri = rows[1:]
        sf = p["safe_ids"].reshape(-1)
        sA = p["down_A"][sf]
        la = torch.einsum("bi,bri->br", ri, sA)
        return la @ p["down_B"][0].T

    def fn_fused():
        la = sorted_a_down_shrink(
            p["intermediate"],
            p["down_A"],
            p["safe_ids"],
            p["sti"],
            route_count=rc,
            K=p["k"],
        )
        return la @ p["down_B"][0].T

    t_cur = time_fn(fn_cur, ())
    t_fused = time_fn(fn_fused, ())
    print(
        f"    current (route+gather+einsum+GEMM): {t_cur:.0f}μs  ×48 = {t_cur*48/1000:.2f}ms"
    )
    print(
        f"    fused   (kernel+GEMM):              {t_fused:.0f}μs  ×48 = {t_fused*48/1000:.2f}ms  ({t_cur/t_fused:.1f}× speedup)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print("=" * 60)

    for rank, label in [(16, "rank=16 (standard)"), (256, "rank=256 (zero adapter)")]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        p = make_inputs(rank)

        bench_gate_up(rank, p)
        bench_down(rank, p)

    print(f"\n{'='*60}")
    print("Estimate for full decode step (48 MoE layers):")
    for rank in [16, 256]:
        p = make_inputs(rank)
        # Gate/up savings
        t_gu_cur = time_fn(lambda: gate_up_current(p), ())
        t_gu_fused = time_fn(lambda: gate_up_fused(p), ())
        # Down savings
        rc = p["rc"]
        n = p["I"]

        def fn_cur_down():
            rows = torch.zeros((rc + 1, n), dtype=p["dtype"], device=p["dev"])
            clipped = (p["sti"].clamp(-1, rc - 1) + 1).to(torch.long)
            rows.scatter_(0, clipped.unsqueeze(-1).expand(-1, n), p["intermediate"])
            ri = rows[1:]
            sf = p["safe_ids"].reshape(-1)
            sA = p["down_A"][sf]
            la = torch.einsum("bi,bri->br", ri, sA)
            return la @ p["down_B"][0].T

        def fn_fused_down():
            la = sorted_a_down_shrink(
                p["intermediate"],
                p["down_A"],
                p["safe_ids"],
                p["sti"],
                route_count=rc,
                K=p["k"],
            )
            return la @ p["down_B"][0].T

        t_down_cur = time_fn(fn_cur_down, ())
        t_down_fused = time_fn(fn_fused_down, ())
        saved_ms = ((t_gu_cur - t_gu_fused) + (t_down_cur - t_down_fused)) * 48 / 1000
        print(
            f"  rank={rank}: estimated LoRA overhead reduction = {saved_ms:.2f}ms per decode step"
        )


if __name__ == "__main__":
    main()
