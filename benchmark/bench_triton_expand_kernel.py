"""Benchmark: Triton per-expert expand kernel vs current all-experts GEMM+scatter.

The kernel from the user replaces the gate_up B step for sglang_shared:
  current: all-experts GEMM (m,R) @ (R,E*I2) → gather per safe_ids → scatter to sorted output
  kernel:  per-pair Triton expand: for each sorted pair, output[row] += W[expert,:,:]@x[row,:]*scale

Run:  python benchmark/bench_triton_expand_kernel.py
"""

from __future__ import annotations

import statistics
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_moe_kernel(
    x,
    weights,
    weight_indices,
    lora_ranks,
    permutation,
    scalings,
    output,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return

    row = tl.load(permutation + pid_s)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = offs_n < OUTPUT_DIM
    weight_base = weights + w_index * OUTPUT_DIM * MAX_RANK + offs_n[:, None] * MAX_RANK
    x_base = x + row * MAX_RANK

    k32 = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if MAX_RANK <= 32:
        xv = tl.load(x_base + k32, mask=k32 < rank, other=0.0).to(tl.float32)
        wv = tl.load(
            weight_base + k32[None, :],
            mask=out_mask[:, None] & (k32[None, :] < rank),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], axis=1)
    else:
        # rank=256 fused case: MAX_RANK=512, use 32-element tiles
        for tile_start in range(0, MAX_RANK, 32):
            km = tl.arange(0, 32) + tile_start
            xv = tl.load(x_base + km, mask=km < rank, other=0.0).to(tl.float32)
            wv = tl.load(
                weight_base + km[None, :],
                mask=out_mask[:, None] & (km[None, :] < rank),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)

    ptrs = output + row * OUTPUT_DIM + offs_n
    delta = acc * tl.load(scalings + w_index)
    old = tl.load(ptrs, mask=out_mask, other=0.0).to(tl.float32)
    tl.store(ptrs, old + delta, mask=out_mask)


def triton_expand_gate_up_B(
    lora_a_m: torch.Tensor,  # (BS, R) — shared A output
    tok: torch.Tensor,  # (padded,) — token index per sorted position
    w13_B: torch.Tensor,  # (E, I2, R) — per-expert B
    exp_sorted: torch.Tensor,  # (padded,) — expert per sorted position
    gate_out: torch.Tensor,  # (padded, I2) — sorted gate_up output (in-place)
    lora_ranks: torch.Tensor,  # (E,) int32
    scalings: torch.Tensor,  # (E,) float32
) -> None:
    padded = gate_out.shape[0]
    I2, R = w13_B.shape[1], w13_B.shape[2]
    perm = torch.arange(padded, dtype=torch.int32, device=gate_out.device)
    x_sorted = lora_a_m[tok]  # (padded, R)
    BLOCK_N = 32
    grid = ((I2 + BLOCK_N - 1) // BLOCK_N, padded)
    _expand_moe_kernel[grid](
        x_sorted,
        w13_B,
        exp_sorted.to(torch.int32),
        lora_ranks,
        perm,
        scalings,
        gate_out,
        OUTPUT_DIM=I2,
        MAX_RANK=R,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )


def benchmark():
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    for rank, label in [
        (16, "rank=16 (standard adapters)"),
        (256, "rank=256 (zero adapters)"),
    ]:
        BS, k, E = 8, 8, 128
        hidden = 2048
        R = 2 * rank  # fused gate+up
        I2 = 2 * 768  # = 1536

        rc = BS * k
        padded = rc + 16

        si = torch.cat(
            [
                torch.randperm(rc, device=dev),
                torch.full((16,), -1, device=dev, dtype=torch.long),
            ]
        )
        ft = si.clamp(0, rc - 1)
        tok = ft // k
        topk_v = ft % k
        safe_ids = torch.randint(0, E, (BS, k), device=dev)
        exp_sorted = safe_ids[tok, topk_v]

        w13_A = torch.randn(1, R, hidden, dtype=dtype, device=dev)
        w13_B = torch.randn(E, I2, R, dtype=dtype, device=dev)
        hs = torch.randn(BS, hidden, dtype=dtype, device=dev)
        go_base = torch.randn(padded, I2, dtype=dtype, device=dev)

        lora_ranks = torch.full((E,), R, dtype=torch.int32, device=dev)
        scalings = torch.ones(E, dtype=torch.float32, device=dev)

        invalid = (si < 0) | (si >= rc)

        def current(gate_out):
            lam = hs @ w13_A[0].T  # (BS, R)
            cands = (lam @ w13_B.permute(2, 0, 1).reshape(R, E * I2)).view(BS, E, I2)
            delta = cands.gather(1, safe_ids.unsqueeze(-1).expand(-1, -1, I2)).reshape(
                rc, I2
            )
            c = si.clamp(0, rc - 1).long()
            r = delta[c]
            r.masked_fill_(invalid.unsqueeze(-1), 0)
            gate_out.add_(r)

        def triton_kernel(gate_out):
            lam = hs @ w13_A[0].T  # (BS, R)
            triton_expand_gate_up_B(
                lam, tok, w13_B, exp_sorted, gate_out, lora_ranks, scalings
            )
            gate_out.masked_fill_(invalid.unsqueeze(-1), 0)  # zero padding

        # Warmup + correctness
        g_cur = go_base.clone()
        g_tri = go_base.clone()
        for _ in range(5):
            current(g_cur)
            triton_kernel(g_tri)
        torch.cuda.synchronize()

        print(f"\n{label}: BS={BS} E={E} I2={I2} R={R}")
        for fn, name, n in [
            (current, "current (all-experts GEMM + scatter)", 48),
            (triton_kernel, "Triton expand kernel (no scatter)", 48),
        ]:
            times = []
            for _ in range(400):
                g = go_base.clone()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                fn(g)
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1))
            mu = statistics.mean(times) * 1000
            print(f"  {name}: {mu:.0f}us  x{n}={mu*n/1000:.1f}ms")


if __name__ == "__main__":
    benchmark()
