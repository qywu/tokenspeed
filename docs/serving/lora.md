# LoRA Serving

TokenSpeed supports PEFT-style LoRA adapters for dense attention and MLP
modules. Dense adapters target:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

Generation requests select adapters by registered `lora_name`. They do not
load adapters from disk. Register the adapter first with `load_lora_adapter`
using a durable adapter path, then pass that name on requests:

```python
engine.load_lora_adapter("password_adapter", "/path/to/adapter_0")
engine.generate("...", lora_name="password_adapter")
```

Requests cannot load adapters from disk and do not accept a request-time
filesystem path. Unknown `lora_name` values fail fast; use the base model by
omitting `lora_name`.

MoE LoRA support is available for expert-scoped weights on Triton MoE
backends. The PEFT per-expert format uses 2D tensors and includes the expert id
in each key:

```text
base_model.model.model.layers.<layer>.mlp.experts.<expert>.gate_proj.lora_A.weight
base_model.model.model.layers.<layer>.mlp.experts.<expert>.gate_proj.lora_B.weight
base_model.model.model.layers.<layer>.mlp.experts.<expert>.up_proj.lora_A.weight
base_model.model.model.layers.<layer>.mlp.experts.<expert>.up_proj.lora_B.weight
base_model.model.model.layers.<layer>.mlp.experts.<expert>.down_proj.lora_A.weight
base_model.model.model.layers.<layer>.mlp.experts.<expert>.down_proj.lora_B.weight
```

TokenSpeed also accepts 3D MoE LoRA tensors under the SGLang-style
`experts.w1`, `experts.w2`, and `experts.w3` names:

```text
base_model.model.model.layers.<layer>.mlp.experts.w1.lora_A.weight
base_model.model.model.layers.<layer>.mlp.experts.w1.lora_B.weight
base_model.model.model.layers.<layer>.mlp.experts.w2.lora_A.weight
base_model.model.model.layers.<layer>.mlp.experts.w2.lora_B.weight
base_model.model.model.layers.<layer>.mlp.experts.w3.lora_A.weight
base_model.model.model.layers.<layer>.mlp.experts.w3.lora_B.weight
```

`w1` maps to `gate_proj`, `w3` maps to `up_proj`, and `w2` maps to
`down_proj`. For these tensors, dimension 0 may be either `num_experts` for a
fully per-expert side or `1` for a shared side. This covers both 3D per-expert
and 3D shared-outer adapter layouts.

The 2D hybrid-shared `experts.shared.*` format is not currently supported.

The current MoE path is guarded to local or tensor-parallel MoE execution.
Expert-parallel dispatch is rejected for MoE LoRA because token ownership and
the LoRA slot map must be dispatched together before expert compute.

Implementation note: dense adapter lifecycle and cache residency are still
owned by `LoraManager`, while expert-scoped MoE tensors are held behind a
`MoeLoraContext` consumed by MoE backends. New MoE LoRA kernels should live
behind the `tokenspeed-kernel` boundary and use that context rather than
depending on the full manager object.
