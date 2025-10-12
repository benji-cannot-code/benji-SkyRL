from flax import nnx
import jax
import jax.numpy as jnp
import optax
from transformers import AutoConfig
from huggingface_hub import snapshot_download

from tx.models import Qwen3ForCausalLM
from tx.utils.models import get_dtype, load_checkpoint
from tx.layers.lora import update_adapter_config


def test_lora_training():
    base_model = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(base_model)
    config.max_lora_adapters = 5
    config.max_lora_rank = 32

    checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])
    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
        load_checkpoint(checkpoint_path, config, model)

        # Set different ranks for each adapter (0: rank 16, 1: rank 8)
        update_adapter_config(model, adapter_index=0, lora_rank=16, lora_alpha=16)
        update_adapter_config(model, adapter_index=1, lora_rank=8, lora_alpha=8)

        # Create optimizer that only targets LoRA A and B parameters
        def is_lora_param(path, value):
            return any(name in path for name in ["lora_A", "lora_B"])

        optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=is_lora_param)

        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype=jnp.int32)
        target_ids = batch[:, 1:]
        input_ids = batch[:, :-1]
        adapter_indices = jnp.array([0, 1], dtype=jnp.int32)

        def loss_fn(model, input_ids, target_ids):
            outputs = model(input_ids, adapter_indices=adapter_indices)
            logits = outputs["logits"]
            return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=target_ids).mean()

        # Compute gradients - we need to use nnx.split to separate parameters
        # that we want to compute gradients for
        graphdef, lora_params, non_lora_params = nnx.split(model, is_lora_param, ...)

        # Helper to extract adapter params at specific index
        def get_adapter_params(params, adapter_idx):
            return jax.tree.map(lambda p: p[adapter_idx].copy(), params)

        # Helper to extract out-of-rank params for an adapter
        def get_out_of_rank_params(params, adapter_idx, rank):
            def slice_param(path, p):
                if "lora_A" in str(path):
                    return p[adapter_idx, :, rank:].copy()
                elif "lora_B" in str(path):
                    return p[adapter_idx, rank:, :].copy()
                return p

            return jax.tree.map_with_path(slice_param, params)

        # Save initial states
        initial_adapter_2_params = get_adapter_params(lora_params, 2)
        initial_adapter_0_out_of_rank = get_out_of_rank_params(lora_params, 0, 16)
        initial_adapter_1_out_of_rank = get_out_of_rank_params(lora_params, 1, 8)

        # Training loop
        for step in range(10):

            def loss_for_lora(lora_params):
                merged_model = nnx.merge(graphdef, lora_params, non_lora_params)
                return loss_fn(merged_model, input_ids, target_ids)

            loss_and_grad_fn = nnx.value_and_grad(loss_for_lora)
            loss, lora_grads = loss_and_grad_fn(lora_params)

            optimizer.update(lora_params, lora_grads)

            print(f"Step {step}: loss = {float(loss):.4f}")

        def verify_params_unchanged(initial_params, final_params, error_msg_prefix):
            for (path, initial), (_, final) in zip(
                jax.tree.leaves_with_path(initial_params), jax.tree.leaves_with_path(final_params)
            ):
                assert jnp.allclose(initial, final), f"{error_msg_prefix} for {path}"

        # Verify adapter 2 (unused) was not modified
        final_adapter_2_params = get_adapter_params(lora_params, 2)
        verify_params_unchanged(initial_adapter_2_params, final_adapter_2_params, "Adapter 2 was modified")

        # Verify out-of-rank params were not modified
        final_adapter_0_out_of_rank = get_out_of_rank_params(lora_params, 0, 16)
        verify_params_unchanged(
            initial_adapter_0_out_of_rank, final_adapter_0_out_of_rank, "Adapter 0 out-of-rank params modified"
        )
        final_adapter_1_out_of_rank = get_out_of_rank_params(lora_params, 1, 8)
        verify_params_unchanged(
            initial_adapter_1_out_of_rank, final_adapter_1_out_of_rank, "Adapter 1 out-of-rank params modified"
        )
