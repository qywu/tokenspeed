"""
Benchmark KV cache eviction speed for SGLang RadixCache.

Mirrors bench_eviction_ts.py so results are directly comparable.
Output is JSON lines.
"""

import json
import time

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


# ---------------------------------------------------------------------------
# Minimal mock allocator
# ---------------------------------------------------------------------------

class MockAllocator:
    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self.device = torch.device("cpu")

    def free(self, indices: torch.Tensor) -> None:
        pass

    def free_group_begin(self) -> None:
        pass

    def free_group_end(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tree-building helpers
# ---------------------------------------------------------------------------

def _make_cache(page_size: int = 16) -> RadixCache:
    return RadixCache.create_simulated(
        mock_allocator=MockAllocator(page_size=page_size),
        page_size=page_size,
    )


def _fill_flat(cache: RadixCache, n: int, pages_per_seq: int = 1) -> None:
    """Insert n unique sequences, each pages_per_seq pages, no prefix sharing."""
    page_size = cache.page_size
    n_tokens = page_size * pages_per_seq
    for s in range(n):
        token_ids = [s] + [0] * (n_tokens - 1)
        key = RadixKey(token_ids=token_ids)
        # One KV index per token so len(leaf.value) == n_tokens, matching evictable_size_ units
        value = torch.arange(n_tokens, dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))


def _fill_shared_prefix(cache: RadixCache, n: int, prefix_pages: int = 100) -> None:
    """n leaves, all sharing a common prefix of prefix_pages pages."""
    page_size = cache.page_size
    shared_tokens = [0] * (page_size * prefix_pages)
    total_tokens = page_size * (prefix_pages + 1)
    for s in range(n):
        unique_tokens = [s + 1] + [0] * (page_size - 1)
        token_ids = shared_tokens + unique_tokens
        key = RadixKey(token_ids=token_ids)
        # One KV index per token so split correctly leaves 1 page (page_size tokens) in leaf
        value = torch.arange(total_tokens, dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _bench(label: str, n: int, pages_per_seq: int, fill_fn, evict_fraction: float = 0.5,
           repeats: int = 7) -> dict:
    cache = _make_cache()
    fill_fn(cache, n, pages_per_seq)

    evict_tokens = int(cache.evictable_size() * evict_fraction)
    if evict_tokens == 0:
        evict_tokens = 1

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cache.evict(EvictParams(num_tokens=evict_tokens))
        times.append(time.perf_counter() - t0)
        # Refill
        fill_fn(cache, n, pages_per_seq)

    times.sort()
    median = times[len(times) // 2]
    return {
        "system": "sglang",
        "label": label,
        "n_seq": n,
        "pages_per_seq": pages_per_seq,
        "evict_tokens": evict_tokens,
        "median_ms": round(median * 1e3, 4),
        "min_ms": round(times[0] * 1e3, 4),
        "max_ms": round(times[-1] * 1e3, 4),
    }


if __name__ == "__main__":
    results = []

    for n in [1_000, 5_000, 20_000, 50_000]:
        results.append(_bench(
            label=f"flat_1page_n{n}",
            n=n,
            pages_per_seq=1,
            fill_fn=_fill_flat,
        ))
        results.append(_bench(
            label=f"flat_8page_n{n}",
            n=n,
            pages_per_seq=8,
            fill_fn=_fill_flat,
        ))
        results.append(_bench(
            label=f"shared100_1page_n{n}",
            n=n,
            pages_per_seq=100,
            fill_fn=_fill_shared_prefix,
        ))

    for r in results:
        print(json.dumps(r))
