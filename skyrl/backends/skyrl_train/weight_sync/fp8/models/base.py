"""Generic per-model spec for serialized FP8 weight sync (blockwise and MXFP8 wires).

``ModelFp8Spec`` groups everything the sync path must know about one model
family — which HF configs it matches, which weights quantize, which vLLM
modules stay unquantized, and how Megatron-Bridge's batched expert tensors
map onto wire projections. ``resolve_fp8_spec`` selects the spec for a
checkpoint; unsupported layouts resolve to ``None`` and callers reject them
explicitly. The vLLM-side fused-loader targets are derived from the same
projections via ``batched_moe_wire_targets``, so sender and receiver share
one source of truth instead of hardcoding the mapping twice.

User exclusions (``fp8_weight_sync_exclude_modules``) are layered on top of
the spec: ``resolve_excluded_modules`` expands them into module names. The
driver (for the engine's ignore list, ``fp8_ignored_layers``) and every
trainer rank (for the sender, ``SerializedFp8Config.excluded_modules``) call
it on the same config, so both sides see the same modules.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# Wire formats live here rather than in vllm_format so model specs can name them
# without importing the serializer that imports this module.
BLOCKWISE_FP8 = "blockwise"
MXFP8 = "mxfp8"
AUTO_FP8 = "auto"
WIRE_FORMATS = (BLOCKWISE_FP8, MXFP8)

# Scale tensor suffix each wire pairs with a quantized ``.weight`` — the one
# mapping both the serializer (name emission) and the batched-MoE receiver
# (target routing) consume.
WIRE_SCALE_SUFFIX = {
    BLOCKWISE_FP8: ".weight_scale_inv",
    MXFP8: ".weight_scale",
}


@dataclass(frozen=True)
class MoeProjection:
    """One routed-expert projection and its vLLM fused-loader target."""

    hf_name: str  # projection name on the wire / in HF checkpoints, e.g. "gate_proj"
    vllm_param: str  # fused vLLM parameter it loads into, e.g. "w13_weight"
    shard_id: str  # FusedMoE weight_loader shard id, e.g. "w1"


@dataclass(frozen=True)
class MoeExpertSpec:
    """A Megatron-Bridge batched expert tensor mapped onto wire projections.

    ``split_dim`` names the tensor dimension that concatenates the
    projections (split evenly, in order); ``None`` means the tensor is a
    single projection.
    """

    experts_base: str  # checkpoint prefix ending in the experts module
    projections: tuple[MoeProjection, ...]
    split_dim: Optional[int] = None


@dataclass(frozen=True)
class ModelFp8Spec:
    """Per-model policy for serialized FP8 weight sync; wire-format-aware
    callbacks receive the concrete format so one spec serves both wires."""

    name: str
    # hf_config -> does this spec support the checkpoint layout?
    matches: Callable[[Any], bool]
    # (hf_name, shape, wire_format) -> serialize this exported weight as FP8?
    should_quantize: Callable[[str, Sequence[int], str], bool]
    # (hf_config, wire_format) -> vLLM module prefixes that must stay unquantized
    ignored_layers: Callable[[Any, str], list[str]]
    # hf_config -> HF module names synced as FP8, as Megatron-Bridge exports them;
    # a layer's routed experts are one entry whose last segment is moe_module.
    # User exclusion globs match against these names.
    fp8_modules: Callable[[Any], list[str]]
    # batched expert tensor name -> MoeExpertSpec, or None if not one
    moe_expert_spec: Callable[[str], Optional[MoeExpertSpec]]
    # sibling module names vLLM fuses into one quantized module, e.g.
    # ("q_proj", "k_proj", "v_proj"); exclusions must cover a group entirely
    fused_modules: tuple[tuple[str, ...], ...] = field(default=())
    # module segment holding routed experts in vLLM parameter names
    moe_module: str = "experts"
    # every projection the model emits, for receiver-side target derivation
    moe_projections: tuple[MoeProjection, ...] = field(default=())


_REGISTRY: list[ModelFp8Spec] = []


def register_fp8_spec(spec: ModelFp8Spec) -> ModelFp8Spec:
    """Register a model spec for ``resolve_fp8_spec`` lookup."""

    if any(existing.name == spec.name for existing in _REGISTRY):
        raise ValueError(f"An FP8 model spec named {spec.name!r} is already registered")
    _REGISTRY.append(spec)
    return spec


def registered_fp8_spec_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in _REGISTRY)


def resolve_fp8_spec(hf_config: Any) -> Optional[ModelFp8Spec]:
    """Return the registered spec matching an HF config, or ``None``."""

    for spec in _REGISTRY:
        if spec.matches(hf_config):
            return spec
    return None


def resolve_excluded_modules(spec: ModelFp8Spec, hf_config: Any, patterns: Sequence[str]) -> tuple[str, ...]:
    """Expand exclusion globs into the FP8 modules they keep unquantized.

    Each glob is matched (``fnmatch``) against ``spec.fp8_modules(hf_config)``.
    A glob that matches none of them is rejected, and so is an exclusion that
    covers only part of a group vLLM fuses into one module: vLLM serves the
    fused module in a single precision, so the sender would ship it mixed.
    """

    modules = spec.fp8_modules(hf_config)
    excluded: set[str] = set()
    for pattern in patterns:
        matched = [module for module in modules if fnmatch.fnmatchcase(module, pattern)]
        if not matched:
            example = f" (e.g. {modules[0]!r})" if modules else ""
            raise ValueError(
                f"fp8_weight_sync_exclude_modules pattern {pattern!r} matches none of the "
                f"{len(modules)} modules {spec.name} syncs as FP8{example}."
            )
        excluded.update(matched)

    ordered = tuple(module for module in modules if module in excluded)
    for module in ordered:
        parent, _, leaf = module.rpartition(".")
        for group in spec.fused_modules:
            if leaf not in group:
                continue
            missing = [f"{parent}.{name}" for name in group if f"{parent}.{name}" not in excluded]
            if missing:
                raise ValueError(
                    f"fp8_weight_sync_exclude_modules excludes {module!r} but not {missing}: vLLM "
                    f"serves {', '.join(group)} as one fused module, so they must be excluded together."
                )
    return ordered


def fp8_ignored_layers(
    spec: ModelFp8Spec,
    hf_config: Any,
    wire_format: str,
    excluded_modules: Sequence[str],
) -> list[str]:
    """Return the modules the engine builds unquantized: the spec's own list plus the exclusions.

    vLLM picks one scheme for all of a layer's routed experts. Its compressed-tensors
    config (MXFP8) decides from expert 0's projection names, matched exactly; its fp8
    config (blockwise) matches any ignored name containing the experts' prefix. An
    excluded experts module is therefore listed as expert 0's projections.
    """

    ignored = list(spec.ignored_layers(hf_config, wire_format))
    for module in excluded_modules:
        if module.rpartition(".")[2] == spec.moe_module:
            ignored.extend(f"{module}.0.{proj.hf_name}" for proj in spec.moe_projections)
        else:
            ignored.append(module)
    return ignored


def batched_moe_wire_targets() -> dict[str, tuple[str, str]]:
    """Receiver mapping: checkpoint suffix -> (fused vLLM suffix, shard id).

    Derived from every registered spec's projections so the vLLM worker
    extension never re-encodes per-model fused-loader knowledge.
    """

    targets: dict[str, tuple[str, str]] = {}
    for spec in _REGISTRY:
        for proj in spec.moe_projections:
            # Blockwise ships weight_scale_inv; MXFP8 ships compressed-tensors'
            # weight_scale. Both map onto the same fused parameter family.
            for weight_suffix, param_suffix in (
                (".weight", ""),
                (WIRE_SCALE_SUFFIX[BLOCKWISE_FP8], "_scale_inv"),
                (WIRE_SCALE_SUFFIX[MXFP8], "_scale"),
            ):
                key = f".{spec.moe_module}.{proj.hf_name}{weight_suffix}"
                value = (f".{spec.moe_module}.{proj.vllm_param}{param_suffix}", proj.shard_id)
                if targets.setdefault(key, value) != value:
                    raise ValueError(f"Conflicting batched MoE wire target registered for suffix {key!r}")
    return targets
