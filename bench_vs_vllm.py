"""Benchmark: ours vs vLLM expand across shapes, adapter counts, ranks.

Four expand variants compared:
  1. ours-seg   : lora_expand_fwd (per-segment dispatch, no sorting)
  2. ours-grp   : lora_expand_decode_fwd (grouped + gather/scatter)
  3. ours-grpv2 : lora_expand_grouped_v2_fwd (grouped, scattered reads, no copy)
  4. vllm        : inlined vLLM expand (same adapter-grouped idea)

Usage:
    python bench_vs_vllm.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent / "tokenspeed-kernel" / "python"))

from tokenspeed_kernel.ops.lora.triton.lora_expand import lora_expand_fwd
from tokenspeed_kernel.ops.lora.triton.lora_expand_decode import lora_expand_decode_fwd
from tokenspeed_kernel.ops.lora.triton.lora_expand_grouped_v2 import (
    lora_expand_grouped_v2_fwd,
)

# ── inlined vLLM expand kernel (Apache-2.0) ───────────────────────────────────

@triton.jit
def _vllm_mm_k(a, b, ak, bk,
               K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, EVEN_K: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(tl.cdiv(K, BK)):
        if EVEN_K:
            acc += tl.dot(tl.load(a), tl.load(b))
        else:
            ko = tl.arange(0, BK); mask = k * BK + ko < K
            acc += tl.dot(tl.load(a, mask=mask[None, :], other=0.0),
                          tl.load(b, mask=mask[:, None], other=0.0))
        a += BK * ak; b += BK * bk
    return acc


@triton.jit
def _vllm_expand_kernel(
    x, w, out, M, N, K,
    sorted_idx, ntok, start_loc, lora_ids,
    scalings, lora_ranks,
    xs0, xs1, ws0, ws1, ws2, os0, os1,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_K: tl.constexpr, MAX_RANK: tl.constexpr,
):
    cta_m = tl.cdiv(M, BM); cta_n = tl.cdiv(N, BN)
    pid   = tl.program_id(0)
    pm    = pid % cta_m; pn = (pid // cta_m) % cta_n
    li    = tl.program_id(1)
    lid   = tl.load(lora_ids + li)
    if lid == -1: return
    lm    = tl.load(ntok + li)
    off   = pm * BM
    if off >= lm: return
    if pn * BN >= N: return
    mlen  = tl.minimum(BM, lm - off)
    ls    = tl.load(start_loc + li)
    om    = tl.arange(0, BM) % mlen
    ram   = tl.load(sorted_idx + ls + off + om)
    no    = tl.arange(0, BN) + pn * BN
    rbn   = tl.max_contiguous(tl.multiple_of(no % N, BN), BN)
    ko    = tl.arange(0, BK)
    # x strides: xs0=inner(1), xs1=row(MAX_RANK)
    ap    = x + ram[:, None] * xs1 + ko[None, :] * xs0
    # w strides: ws0=adapter, ws1=N, ws2=K(=1)
    bp    = w + lid * ws0 + ko[:, None] * ws2 + rbn[None, :] * ws1
    acc   = _vllm_mm_k(ap, bp, xs0, ws2, K, BM, BN, BK, EVEN_K)
    sc    = tl.load(scalings + lid)
    rank  = tl.load(lora_ranks + lid)
    acc  *= sc
    acc   = acc.to(x.dtype.element_ty)
    om2   = tl.arange(0, BM)
    cp    = out + ram[:, None] * os0 + rbn[None, :] * os1
    mask  = (om2[:, None] < mlen) & (rbn[None, :] < N)
    acc  += tl.load(cp, mask=mask, other=0.0)
    tl.store(cp, acc, mask=mask)


def vllm_expand(x, weights, meta, base_output,
                BM=16, BN=64, BK=64, nw=4, ns=2):
    M, K = x.shape; N = weights.shape[1]
    EVEN_K = (K % BK == 0)
    o = base_output
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), meta['num_active'])
    _vllm_expand_kernel[grid](
        x, weights, o, M, N, K,
        meta['sorted_idx'], meta['ntok'], meta['start_loc'], meta['lora_ids'],
        meta['scalings'], meta['lora_ranks'],
        x.stride(1), x.stride(0),
        weights.stride(0), weights.stride(1), weights.stride(2),
        o.stride(0), o.stride(1),
        BM=BM, BN=BN, BK=BK, EVEN_K=EVEN_K, MAX_RANK=K,
        num_warps=nw, num_stages=ns,
    )
    return o


# ── batch-info builders ───────────────────────────────────────────────────────

def make_our_bi(n, rank, n_unique, dev):
    slots = [(i % n_unique) + 1 for i in range(n)]
    sort_order = sorted(range(n), key=lambda i: slots[i])
    groups = []
    for pos, orig in enumerate(sort_order):
        s = slots[orig]
        if not groups or groups[-1][0] != s:
            groups.append([s, pos, 1])
        else:
            groups[-1][2] += 1
    ng = len(groups)

    so_t  = torch.tensor(sort_order,             dtype=torch.int64, device=dev)
    gs_t  = torch.tensor([g[0] for g in groups], dtype=torch.int32, device=dev)
    gst_t = torch.tensor([g[1] for g in groups], dtype=torch.int32, device=dev)
    gsz_t = torch.tensor([g[2] for g in groups], dtype=torch.int32, device=dev)

    class BI:
        bs = n; max_len = 1
        seg_lens       = torch.ones(n, dtype=torch.int32, device=dev)
        seg_indptr     = torch.arange(n + 1, dtype=torch.int32, device=dev)
        weight_indices = torch.tensor(slots, dtype=torch.int32, device=dev)
        lora_ranks     = torch.tensor([0] + [rank] * n_unique, dtype=torch.int32, device=dev)
        scalings       = torch.ones(n_unique + 1, dtype=torch.float32, device=dev)
        permutation    = None
        num_groups     = ng
        sort_order     = so_t
        group_slots    = gs_t
        group_starts   = gst_t
        group_sizes    = gsz_t
    return BI()


def make_vllm_meta(n, rank, n_unique, n_slots, dev):
    # slot 0 = no-adapter sentinel; real adapters = 1..n_unique
    slots = torch.tensor([(i % n_unique) + 1 for i in range(n)],
                         dtype=torch.int32, device=dev)
    _, sorted_idx = torch.sort(slots, stable=True)
    uniq, counts  = torch.unique(slots, sorted=True, return_counts=True)
    start_locs    = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                                counts.cumsum(0).to(torch.int32)])
    lora_ranks_t  = torch.tensor([0] + [rank] * n_unique, dtype=torch.int32, device=dev)
    scalings_t    = torch.ones(n_unique + 1, dtype=torch.float32, device=dev)
    return {
        'sorted_idx': sorted_idx.to(torch.int32),
        'ntok':       counts.to(torch.int32),
        'start_loc':  start_locs,
        'lora_ids':   uniq.to(torch.int32),
        'num_active': len(uniq),
        'lora_ranks': lora_ranks_t,
        'scalings':   scalings_t,
    }


def bench(fn, w=30, r=300):
    return triton.testing.do_bench(fn, warmup=w, rep=r) * 1000


# ── sweep ─────────────────────────────────────────────────────────────────────

def header(title):
    print(f'\n{"="*80}')
    print(f'  {title}')
    print(f'{"="*80}')
    print(f'  {"n":>4}  {"n_uniq":>6}  {"seg":>8}  {"grp":>8}  {"grpv2":>8}  {"vllm":>8}  {"best":>6}')
    print(f'  {"-"*58}')


def row(n, nu, ts, tg, tv2, tv):
    ts  = f'{ts:.1f}µ'  if ts  else  '   n/a'
    tg  = f'{tg:.1f}µ'  if tg  else  '   n/a'
    tv2 = f'{tv2:.1f}µ' if tv2 else  '   n/a'
    tv  = f'{tv:.1f}µ'  if tv  else  '   n/a'
    # which is fastest among numeric values
    vals = [(t, nm) for t, nm in [(ts,'seg'),(tg,'grp'),(tv2,'v2'),(tv,'vllm')]
            if 'n/a' not in str(t)]
    best = min(vals, key=lambda x: float(x[0].rstrip('µ')))[1] if vals else '?'
    print(f'  {n:>4}  {nu:>6}  {ts:>8}  {tg:>8}  {tv2:>8}  {tv:>8}  {best:>6}')


dev, dt = 'cuda', torch.bfloat16

for rank, N in [(16, 4096), (64, 4096), (128, 4096), (64, 8192)]:
    header(f'EXPAND  rank={rank}  N={N}  (x: n×{rank} → out: n×{N})')
    for n in (8, 16, 32, 64, 128):
        for n_u in sorted({1, min(4, n), min(n, 8), n}):
            if n_u > n: continue
            bi  = make_our_bi(n, rank, n_u, dev)
            vm  = make_vllm_meta(n, rank, n_u, n_u + 1, dev)
            wo  = torch.randn(n_u + 1, N, rank, device=dev, dtype=dt)
            wv  = wo[1:]   # vLLM doesn't have slot-0 sentinel
            x   = torch.randn(n, rank, device=dev, dtype=dt)
            o   = torch.zeros(n, N, device=dev, dtype=dt)

            bk  = min(rank, 64)
            use_grp = bi.bs // bi.num_groups >= 8

            ts  = bench(lambda: lora_expand_fwd(x, wo, bi, base_output=o.clone()))
            tg  = bench(lambda: lora_expand_decode_fwd(x, wo, bi, base_output=o.clone())) if use_grp else None
            tv2 = bench(lambda: lora_expand_grouped_v2_fwd(x, wo, bi, base_output=o.clone())) if n_u > 0 else None
            tv  = bench(lambda: vllm_expand(x, wv, vm, base_output=o.clone(), BK=bk))

            row(n, n_u, ts, tg, tv2, tv)
