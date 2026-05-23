# Tokenspeed RL Development Plan

Online RL training (PPO, GRPO, etc.) requires the inference engine to alternate
between generating rollouts and absorbing updated model weights from the trainer.
This plan tracks four milestones needed to support that loop in tokenspeed.

---

## Milestone 1 — Scheduler Pause / Resume

**Goal:** Allow an external process to cleanly stop and restart generation without
restarting the server or losing queue state.

### Scheduler state machine

```
running ──pause──▶ draining ──drained──▶ paused ──resume──▶ running
running ──pause(abort)──────────────────▶ paused ──resume──▶ running
                                          paused ──sleep──▶  sleeping
                                        sleeping ──wakeup──▶ paused
```

Three pause modes (matching vLLM / SGLang conventions):

| Mode | Behaviour |
|------|-----------|
| `drain` | Stop accepting new requests; let in-flight finish. |
| `abort` | Immediately abort all in-flight requests and clear the queue. |
| `keep` | Freeze the scheduler; preserve in-flight requests (resume continues them). |

### API surface

```
POST /pause          body: { mode: "drain" | "abort" | "keep" }
POST /resume
GET  /is_paused      returns: { paused: bool, mode: str | null }
POST /sleep          body: { tags: ["weights", "kv_cache", "cuda_graphs"] }
POST /wakeup         body: { tags: ["weights", "kv_cache", "cuda_graphs"] }
```

`/sleep` offloads GPU memory to CPU so the trainer can reclaim VRAM for the
backward pass. `/wakeup` restores it. Both require the engine to already be
paused; `wakeup` returns to `paused` (not `running`) so the caller decides when
to resume generation.

### Implementation notes

- Add `PauseState` enum (`running | draining | paused | sleeping`) and an
  `asyncio.Event` (or equivalent) to the scheduler.
- `scheduler.step()` checks the pause gate before scheduling any new batch.
- `drain` mode: mark `accepting = False`; the pause gate opens only after the
  in-flight count reaches zero.
- HTTP handlers `await` the scheduler's "fully-paused" confirmation before
  returning 200 so the caller knows the engine is safe to modify.
- `/sleep`: walks worker processes and calls `torch_memory_saver` (or equivalent)
  to offload selected tags to pinned CPU memory; updates state to `sleeping`.
- `/wakeup`: restores offloaded tensors to GPU, optionally re-warms CUDA graphs,
  returns to `paused` state.
- Calling `/sleep` on a running engine or `/wakeup` on a non-sleeping engine
  returns 409 Conflict.
- No changes to the NCCL or kernel paths at this stage; sleep/wakeup is
  purely a memory-management concern.

---

## Milestone 2 — Log Probability Endpoint

**Goal:** Let the trainer score sequences against the current policy — the core
primitive needed by PPO, GRPO, and any KL-penalised objective.

### API surface

```
POST /compute_log_probs
```

Request:
```json
{
  "sequences": [
    { "prompt_token_ids": [1, 2, 3], "completion_token_ids": [4, 5, 6] },
    ...
  ]
}
```

Response:
```json
{
  "log_probs": [[-0.12, -0.47, -0.31], ...],
  "tokens":    [[4, 5, 6], ...]
}
```

- `log_probs[i][j]` is `log P(completion_token_ids[i][j] | context)` under the
  current weights, always at temperature 1.0 (raw logits, no scaling).
- Only completion-position logprobs are returned; prompt tokens are context.

### Implementation notes

- Run a standard prefill forward pass over `prompt + completion`; extract
  logits at the completion positions and apply `log_softmax`.
- Gather logits across TP ranks before softmax (same pattern as the existing
  sampling path).
- Requests can be batched and interleaved with normal generation — no pause
  required. During RL training they are typically sent as a dedicated batch
  after rollouts complete and before the weight update.
- Variable-length sequences: left-pad to the batch maximum and mask out padding
  positions in the returned logprobs.
- Expose a `temperature` field (default 1.0) for callers that want
  scaled logprobs, but the reference-model KL case should always use 1.0.

### Reference model pattern

For algorithms that need KL against a frozen reference policy, run a second
tokenspeed instance (or a CPU-offloaded snapshot) and call `/compute_log_probs`
on both. No special engine support is needed at this milestone; the multi-instance
coordination is the trainer's responsibility.

---

## Milestone 3 — NCCL Distributed Weight Update

**Goal:** Let a trainer push new weights into the running engine over NCCL,
matching SGLang's `/update_weights_from_distributed` workflow.

### Endpoint set

```
POST /init_weight_update_group      { master_address, master_port, rank_offset, world_size }
POST /update_weights_from_distributed  { names[], dtypes[], shapes[], flush_kv_cache }
POST /destroy_weight_update_group
GET  /weight_version
POST /update_weight_version         { version: str }
```

### Rank formula (TP × PP × DP)

```python
dp_rank       = parallel_config.data_parallel_index
local_world   = parallel_config.world_size   # TP × PP
local_rank    = parallel_config.rank
worker_rank   = dp_rank * local_world + local_rank + rank_offset
# trainer is rank 0 in the shared group
```

### Worker-side weight loading

- Each worker joins the NCCL group at `init_weight_update_group`.
- On `update_weights_from_distributed`: broadcast-receive each tensor then call
  the existing `load_weights()` path — it already handles TP sharding by ignoring
  parameters outside its slice.
- PP workers ignore tensors belonging to layers outside their pipeline stage
  (same load_weights guard).
- After all tensors are loaded, flush KV cache if requested and bump the version tag.

### Typical RL loop

```
trainer                     tokenspeed
──────                      ──────────
                            POST /pause (drain)
POST /init_weight_update_group
  [NCCL rendezvous]
POST /update_weights_from_distributed  ← NCCL broadcast per parameter
POST /destroy_weight_update_group
                            POST /resume
                            GET  /weight_version  (confirm)
start rollout collection →  normal generation
```

### Caveats

- `world_size` in the init request must be `TP × PP × DP + 1` (the +1 is the trainer rank).
- For FP8 models, a post-load quantization recalibration pass may be needed after
  weights are applied (same pattern as vLLM `finish_weight_update`).
- NCCL group teardown should be synchronous and guarded against double-free.

---

## Milestone 4 — Mooncake P2P RDMA Weight Transfer

**Goal:** Replace the centralized NCCL broadcast with direct trainer-rank→inference-rank
RDMA transfers, eliminating redundant hops for large MoE models with high expert
parallelism. Target: match the 3–7× speedup shown for Qwen3-235B / Kimi-K2 1T in
the [SGLang P2P blog post](https://www.lmsys.org/blog/2026-04-29-p2p-update/).

### Core idea

- Each trainer rank holds a CPU-side copy of its own parameter shard (trades CPU
  memory for network bandwidth).
- Each inference rank registers its GPU weight buffers once with the Mooncake
  `TransferEngine` at startup.
- During update: every trainer rank sends *only its own shard* directly to the
  matching inference rank(s) — no global synchronisation, fully concurrent.

### Two-phase protocol

**Phase 1 — `POST /prepare_weights_update`**

```json
{ "names": [...], "dtypes": [...], "shapes": [...],
  "tp_size": 8, "ep_size": 32 }
```

Engine response:
```json
{ "tensor_map": { "layer.0.weight": { "addr": "...", "size": ... }, ... },
  "rank_mapping": { "trainer_rank_0": ["engine_rank_2", "engine_rank_6"], ... },
  "transfer_engine_endpoints": [...] }
```

- Engine allocates/registers weight buffers with `TransferEngine` (one-time if
  already registered).
- Returns the address handles and rank↔shard mapping so each trainer rank knows
  exactly where to write.

**Phase 2 — `POST /complete_weights_update`**

```json
{ "flush_kv_cache": true }
```

- Engine waits for all RDMA transfers to complete.
- Runs any post-processing (FP8 scale recalibration, layer norm re-init).
- Flushes KV cache and bumps weight version.

### Rank mapping construction

For a model with TP=8, EP=32 (256 inference GPUs), the trainer has 32 ranks (one
per expert group). The mapping is:

```
trainer_rank r  →  inference ranks { r*TP ... r*TP + TP-1 }
```

Each trainer rank RDMA-writes its expert shard to all TP peers in the corresponding
expert group. TP peers then share the shard via NVLink without any additional NCCL
broadcast.

### Dependencies

- `mooncake-transfer-engine` Python package (pip, RDMA-capable NIC required).
- Network: InfiniBand or RoCE; falls back to TCP (slower, still functional).
- Kernel flag: `RDMA_DEVICE` env var or engine config for NIC selection.

### Memory budget

| Component | Per inference rank |
|-----------|-------------------|
| GPU weight buffers (already allocated) | P bytes |
| Mooncake registration overhead | ~8 bytes/page |
| Trainer CPU shard copy | P / EP bytes per trainer rank |

### Fallback

If Mooncake is unavailable or the NIC does not support RDMA, automatically fall
back to Milestone 3 NCCL broadcast. Expose a config flag `weight_sync_backend`:
`auto | nccl | mooncake`.

---

## Summary

| Milestone | Goal | Key dependency |
|-------|-----------|----------------|
| 1 | Pause/resume generation | Scheduler only |
| 2 | Log probability endpoint | Prefill forward pass; TP logit gather |
| 3 | NCCL weight update endpoints | NCCL process group; existing `load_weights` |
| 4 | Mooncake P2P RDMA | `mooncake-transfer-engine`; RDMA NIC |

Milestones are strictly sequential: Milestone 3 requires Milestone 1 (pause before update),
and Milestone 4 builds on Milestone 3's endpoint structure and weight-loading path.
Milestone 2 can be developed in parallel with Milestone 3.

---

## Open Questions

### Q1 — Compatibility with external RL training engines

Popular RL training frameworks each assume a specific rollout engine interface:

| Framework | Default rollout engine | Weight sync interface used |
|-----------|----------------------|---------------------------|
| [veRL](https://github.com/volcengine/verl) | vLLM or SGLang | vLLM programmatic API or SGLang HTTP (`/update_weights_from_distributed`, `/pause_generation`, …) |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | vLLM or SGLang | vLLM HTTP or SGLang HTTP |
| [SLIME](https://github.com/SLIME-RL/SLIME) | SGLang | SGLang HTTP (`/update_weights_from_distributed`, `/pause_generation`, …) |

**The question:** should tokenspeed expose a compatibility shim so existing RL
training scripts can point at tokenspeed without modification, or define its own
canonical API and require framework-side adapters?

Options:

1. **vLLM-compatible surface** — implement the same HTTP endpoints and Python
   class methods as vLLM's RLHF server. Lowest friction for veRL / OpenRLHF users.
   Risk: locks the API shape to vLLM's design choices.

2. **SGLang-compatible surface** — implement SGLang's endpoint names and request
   schemas. Covers OpenRLHF + SLIME. Risk: SGLang endpoint names are less consistent.

3. **Tokenspeed-native API + thin adapters** — define the cleanest API for
   tokenspeed, then ship lightweight adapter classes (e.g.
   `TokenspeedVerlRollout`, `TokenspeedSGLangAdapter`) that translate between
   frameworks. Most maintenance flexibility; requires upfront adapter work.

4. **Hybrid** — expose both `/update_weights_from_distributed` (SGLang name) and
   `/update_weights` (vLLM name) as aliases behind a single implementation.
   Low-cost compatibility without a separate adapter layer.

