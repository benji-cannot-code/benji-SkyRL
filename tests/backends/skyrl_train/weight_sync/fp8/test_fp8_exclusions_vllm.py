"""SkyRL's FP8 ignore list for Qwen3.5, run through vLLM's own matching.

At engine init vLLM maps every ignore-list name through the model's
``hf_to_vllm_mapper``, then decides each module from that list, expanding fused
modules through ``packed_modules_mapping``: ``Fp8Config`` uses
``is_layer_skipped`` (blockwise wire) and ``CompressedTensorsConfig`` uses
``should_ignore_layer`` (MXFP8 wire), checking a MoE layer through expert 0's
projections. These tests drive those functions on the module names vLLM builds,
so a name SkyRL emits that vLLM would not honor fails here.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

pytestmark = pytest.mark.vllm

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (  # noqa: E402
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    is_layer_skipped,
)
from vllm.model_executor.models.qwen3_5 import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)

from skyrl.backends.skyrl_train.weight_sync.fp8 import (  # noqa: E402
    BLOCKWISE_FP8,
    MXFP8,
    fp8_ignored_layers,
    get_serialized_fp8_quantization_config,
    resolve_excluded_modules,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models import (  # noqa: E402
    QWEN35_FP8_SPEC,
)

VLLM = "language_model.model.layers"

# vLLM module name -> expected to be built unquantized, for the exclusions below.
EXPECTED_UNQUANTIZED = {
    # layer 0 (GDN): only the routed experts are excluded
    f"{VLLM}.0.mlp.experts": True,
    f"{VLLM}.0.linear_attn.in_proj_qkvz": False,
    f"{VLLM}.0.linear_attn.in_proj_ba": True,  # the spec's own ignore list
    f"{VLLM}.0.mlp.shared_expert.gate_up_proj": False,
    # layer 1 (GDN): only the shared expert
    f"{VLLM}.1.mlp.shared_expert.gate_up_proj": True,
    f"{VLLM}.1.mlp.shared_expert.down_proj": True,
    f"{VLLM}.1.mlp.experts": False,
    # layer 2 (GDN): only out_proj
    f"{VLLM}.2.linear_attn.out_proj": True,
    f"{VLLM}.2.linear_attn.in_proj_qkvz": False,
    # layer 3 (full attention): everything
    f"{VLLM}.3.self_attn.qkv_proj": True,
    f"{VLLM}.3.self_attn.o_proj": True,
    f"{VLLM}.3.mlp.experts": True,
    f"{VLLM}.3.mlp.shared_expert.gate_up_proj": True,
    f"{VLLM}.3.mlp.shared_expert.down_proj": True,
    # layer 4 (full attention): nothing
    f"{VLLM}.4.self_attn.qkv_proj": False,
    f"{VLLM}.4.mlp.experts": False,
}


def _ignored_layers(wire_format):
    hf_config = SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            layer_types=["linear_attention"] * 3 + ["full_attention"] * 2,
            shared_expert_intermediate_size=512,
        ),
    )
    excluded = resolve_excluded_modules(
        QWEN35_FP8_SPEC,
        hf_config,
        ["*.layers.0.mlp.experts", "*.layers.1.mlp.shared_expert.*", "*.layers.2.linear_attn.out_proj", "*.layers.3.*"],
    )
    return fp8_ignored_layers(QWEN35_FP8_SPEC, hf_config, wire_format, excluded)


def test_blockwise_engine_builds_exactly_the_excluded_modules_unquantized():
    model = Qwen3_5MoeForConditionalGeneration
    quant_config = Fp8Config.from_config(
        get_serialized_fp8_quantization_config(ignored_layers=_ignored_layers(BLOCKWISE_FP8), wire_format=BLOCKWISE_FP8)
    )
    quant_config.apply_vllm_mapper(model.hf_to_vllm_mapper)

    built_unquantized = {
        module: is_layer_skipped(
            module,
            quant_config.ignored_layers,
            fused_mapping=model.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        )
        for module in EXPECTED_UNQUANTIZED
    }

    assert built_unquantized == EXPECTED_UNQUANTIZED


def test_mxfp8_engine_builds_exactly_the_excluded_modules_unquantized():
    model = Qwen3_5MoeForConditionalGeneration
    quant_config = CompressedTensorsConfig.from_config(
        get_serialized_fp8_quantization_config(ignored_layers=_ignored_layers(MXFP8), wire_format=MXFP8)
    )
    quant_config.apply_vllm_mapper(model.hf_to_vllm_mapper)

    def ignored(module):
        if module.endswith(".experts"):
            # CompressedTensorsMoEMethod.get_moe_method checks expert 0's projections.
            names = [f"{module}.0.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")]
            return all(should_ignore_layer(name, quant_config.ignore, model.packed_modules_mapping) for name in names)
        return should_ignore_layer(module, quant_config.ignore, model.packed_modules_mapping)

    assert {module: ignored(module) for module in EXPECTED_UNQUANTIZED} == EXPECTED_UNQUANTIZED
