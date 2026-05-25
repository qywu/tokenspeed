"""Micro-benchmark and torch.profiler trace for apply_lm_head_lora.

Compares:
  - current:  batched bmm regardless of single-slot or multi-slot
  - proposed: regular matmul when single_lora_slot is set

Run:
  python benchmark/profile_lm_head_lora.py
"""

from __future__ import annotations

import statistics

import torch
import torch.profiler

HIDDEN = 4096
VOCAB = 152064
RANK = 16
BS = 8
N_SLOTS = 8
WARMUP = 50
BENCH = 200
DTYPE = torch.bfloat16
DEV = torch.device("cuda")


def setup():
    torch.manual_seed(0)
    A_buf = torch.randn(N_SLOTS, RANK, HIDDEN, dtype=DTYPE, device=DEV)
    B_buf = torch.randn(N_SLOTS, VOCAB, RANK, dtype=DTYPE, device=DEV)
    hidden = torch.randn(BS, HIDDEN, dtype=DTYPE, device=DEV)
    logits = torch.randn(BS, VOCAB, dtype=DTYPE, device=DEV)
    return A_buf, B_buf, hidden, logits


def current_bmm(A_buf, B_buf, hidden, logits, slots):
    """Current implementation: always batched bmm."""
    A = A_buf[slots]  # (bs, r, hidden)
    B = B_buf[slots]  # (bs, vocab, r)
    lora_a = torch.bmm(A, hidden.unsqueeze(-1)).squeeze(-1)  # (bs, r)
    delta = torch.bmm(B, lora_a.unsqueeze(-1)).squeeze(-1)  # (bs, vocab)
    return logits + delta


def single_slot_matmul(A_buf, B_buf, hidden, logits, slot):
    """Proposed: regular matmul when all requests use the same slot."""
    A = A_buf[slot]  # (r, hidden)
    B = B_buf[slot]  # (vocab, r)
    lora_a = hidden @ A.T  # (bs, r)
    delta = lora_a @ B.T  # (bs, vocab)
    return logits + delta


def time_fn(fn, *args, n=BENCH):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn(*args)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return statistics.mean(times), statistics.stdev(times)


def profile_fn(label, fn, *args):
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(10):
            fn(*args)
    print(f"\n--- {label} (top CUDA kernels) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))


def optimized(A_buf, B_buf, hidden, logits, slot_int: int, scaling: float = 1.0):
    """Optimized single-slot path: plain matmul, no gather."""
    A = A_buf[slot_int]  # (r, hidden)
    B = B_buf[slot_int]  # (vocab, r)
    lora_a = hidden @ A.T  # (bs, r)
    delta = lora_a @ B.T  # (bs, vocab)
    return logits + delta * scaling


def main():
    A_buf, B_buf, hidden, logits = setup()

    slots = {
        1: torch.zeros(BS, dtype=torch.long, device=DEV),
        2: torch.arange(BS, device=DEV) % 2,
        4: torch.arange(BS, device=DEV) % 4,
        8: torch.arange(BS, device=DEV) % 8,
    }

    print(
        f"Shapes:  hidden=({BS},{HIDDEN})  A=({N_SLOTS},{RANK},{HIDDEN})  "
        f"B=({N_SLOTS},{VOCAB},{RANK})\n"
    )
    print(f"{'Config':<40} {'GPU μs':>8} {'stdev':>7}")
    print("-" * 58)

    for n_active, sl in slots.items():
        mean, std = time_fn(current_bmm, A_buf, B_buf, hidden, logits, sl)
        print(
            f"  bmm    n_active={n_active}                    {mean*1000:>8.1f} {std*1000:>7.2f}"
        )

    print()
    mean, std = time_fn(optimized, A_buf, B_buf, hidden, logits, 0)
    print(f"  matmul n_active=1 (optimized eager)  {mean*1000:>8.1f} {std*1000:>7.2f}")

    # Profiler traces.
    profile_fn(
        "current bmm  n_active=1", current_bmm, A_buf, B_buf, hidden, logits, slots[1]
    )
    profile_fn(
        "optimized matmul n_active=1", optimized, A_buf, B_buf, hidden, logits, 0
    )
    profile_fn(
        "current bmm  n_active=8", current_bmm, A_buf, B_buf, hidden, logits, slots[8]
    )


if __name__ == "__main__":
    main()
