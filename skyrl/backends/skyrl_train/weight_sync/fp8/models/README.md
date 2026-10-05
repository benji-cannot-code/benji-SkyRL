# Per-model specs for serialized FP8 weight sync

Serialized FP8 weight sync needs these model-specific answers, grouped in a
`ModelFp8Spec` (`base.py`) and resolved once per checkpoint with
`resolve_fp8_spec(hf_config)`:

| Field | Question it answers |
| --- | --- |
| `matches(hf_config)` | Does this spec support the checkpoint layout? |
| `should_quantize(name, shape, wire_format)` | Should this exported HF weight be FP8 on the wire? (Linear weights yes; embeddings, norms, conv, router gates no.) |
| `ignored_layers(hf_config, wire_format)` | Which vLLM module prefixes must stay unquantized to match the checkpoint-format stream? |
| `fp8_modules(hf_config)` | Which HF modules does the spec sync as FP8, named as Megatron-Bridge exports them? A MoE layer's routed experts are one entry (e.g. `model.language_model.layers.3.mlp.experts`). User exclusions match against this list. |
| `moe_expert_spec(name)` | Is this a Megatron-Bridge *batched* expert tensor, and how does it map onto per-projection wire tensors? `None` for ordinary tensors. |
| `fused_modules` | Which sibling modules does vLLM fuse into one quantized module (e.g. `q_proj`/`k_proj`/`v_proj`)? An exclusion must cover a whole group. |

Everything else — blockwise casting, wire naming, the vLLM quantization
config, the receiver's fused-MoE loading — is generic and lives outside this
package. The receiver-side table (which fused vLLM parameter each expert
projection loads into) is **derived** from the specs via
`batched_moe_wire_targets()`, so the mapping is declared exactly once.

## User exclusions

`generator.inference_engine.fp8_weight_sync_exclude_modules` keeps modules in
the model dtype on top of the spec. `resolve_excluded_modules(spec, hf_config,
patterns)` expands the globs against `fp8_modules`; both consumers call it on
the same config, so they see the same modules:

- **engine init**: `fp8_ignored_layers` appends the excluded modules to
  `ignored_layers` (an excluded experts module is listed as expert 0's
  projections, which is how vLLM decides a whole MoE layer);
- **weight sync**: `SerializedFp8Config.excluded_modules` makes the sender
  pass those weights through unquantized, and the stream raises if an excluded
  module never appears in the trainer's export.

A pattern that matches no FP8 module, or that splits a `fused_modules` group,
is rejected before the engine starts.

## Adding a new model

1. Create `models/<family>.py`. Implement the callables (plain functions;
   see `qwen35.py`). Build `should_quantize`'s suffix table and `fp8_modules`
   from the same module tables, so the sender's policy and the names user
   exclusions match against cannot drift apart.
2. If the model has routed MoE experts exported as batched 3D tensors,
   declare one `MoeProjection(hf_name, vllm_param, shard_id)` per projection
   and return `MoeExpertSpec(experts_base, projections, split_dim)` from
   `moe_expert_spec` — `split_dim` is the dimension that concatenates fused
   projections (e.g. `gate_up_proj` splits in half along dim 1); use `None`
   when the tensor is a single projection.
3. Register it at module bottom and import the module in
   `models/__init__.py` (the import is what registers the spec):

   ```python
   MYMODEL_FP8_SPEC = register_fp8_spec(
       ModelFp8Spec(
           name="mymodel",
           matches=is_mymodel_config,
           should_quantize=is_quantizable_weight_shape,
           ignored_layers=get_mymodel_fp8_ignored_layers,
           fp8_modules=get_mymodel_fp8_modules,
           moe_expert_spec=batched_moe_expert_spec,
           fused_modules=(("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")),
           moe_projections=(_MOE_GATE, _MOE_UP, _MOE_DOWN),
       )
   )
   ```

4. Add tests mirroring `tests/backends/skyrl_train/weight_sync/fp8/
   test_serialized_fp8.py` (quantize filter, ignored layers, MoE mapping) and
   `test_fp8_exclusions.py` / `test_fp8_exclusions_vllm.py` (exclusions, run
   through vLLM's own matching), and, for real coverage, an FP8 row in the
   GPU CI logprobs-roundtrip test.

No generic file changes are needed; checkpoints that match no registered
spec are rejected with the list of registered spec names.
