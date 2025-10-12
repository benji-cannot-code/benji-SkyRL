from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.util import Param, prepare_routing


class LoRAMixin:
    """A mixin for flax NNX modules to add multi-adapter LoRA support.
    This mixin adds LoRA parameters (lora_A, lora_B) and methods to apply
    the low-rank adaptation to a base module's output. It is designed to
    be used with layers like nnx.Linear.
    """

    lora_scaling: nnx.Variable | None
    lora_ranks: nnx.Variable | None
    lora_A: nnx.Param | None
    lora_B: nnx.Param | None

    def init_lora(
        self,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        sharding_A: jax.sharding.PartitionSpec,
        sharding_B: jax.sharding.PartitionSpec,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank

        if max_lora_adapters == 0:
            self.lora_scaling = None
            self.lora_ranks = None
            self.lora_A = None
            self.lora_B = None
        else:
            self.lora_scaling = nnx.Variable(jnp.full((max_lora_adapters,), 1.0, dtype=dtype))
            self.lora_ranks = nnx.Variable(jnp.full((max_lora_adapters,), max_lora_rank, dtype=jnp.int32))
            self.lora_A = Param(
                *shape_A,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.he_uniform(), sharding_A),
                rngs=rngs,
            )
            self.lora_B = Param(
                *shape_B,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), sharding_B),
                rngs=rngs,
            )

    def apply_lora(
        self,
        x: jax.Array,
        base_output: jax.Array,
        adapter_indices: jax.Array | None,
    ) -> jax.Array:
        if self.max_lora_adapters == 0 or adapter_indices is None:
            return base_output

        (batch_size, seq_len, in_features) = x.shape
        assert len(self.lora_A.shape) == 3 and self.lora_A.value.shape[1] == in_features
        assert adapter_indices.shape[0] == batch_size

        x_flat = x.reshape(-1, in_features)
        adapter_indices_expanded = jnp.repeat(adapter_indices, seq_len)

        # Sort tokens to prepare for ragged_dot
        x_sorted, group_sizes, unsort_indices = prepare_routing(
            x_flat, adapter_indices_expanded, self.max_lora_adapters
        )

        # Apply LoRA using ragged_dot: x @ A @ B
        intermediate = jax.lax.ragged_dot(x_sorted, self.lora_A.value, group_sizes)
        lora_output_sorted = jax.lax.ragged_dot(intermediate, self.lora_B.value, group_sizes)

        # Unsort, reshape, scale
        lora_output = lora_output_sorted[unsort_indices].reshape(batch_size, seq_len, -1)
        lora_output = lora_output * self.lora_scaling.value[adapter_indices, None, None]
        return base_output + lora_output.reshape(base_output.shape)


class LoRALinear(LoRAMixin, nnx.Linear):
    """An nnx.Linear layer with multi-adapter LoRA support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype | None = None,
        use_bias: bool = True,
        kernel_init: nnx.Initializer | None = None,
        bias_init: nnx.Initializer | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        param_dtype = param_dtype or dtype
        if use_bias and bias_init is None:
            bias_init = nnx.initializers.zeros_init()

        super().__init__(
            in_features,
            out_features,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        assert (
            self.kernel.value.sharding is not None
        ), "LoRALinear layer needs sharding, you can specify it by using nnx.with_partitioning on the kernel_init"
        sharding = self.kernel.value.sharding.spec
        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, max_lora_rank, out_features),
            sharding_A=jax.sharding.PartitionSpec(None, sharding[0], None),
            sharding_B=jax.sharding.PartitionSpec(None, None, sharding[1]),
            dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        base_out = super().__call__(x)
        return self.apply_lora(x, base_out, adapter_indices)


def update_adapter_config(model: nnx.Module, adapter_index: int, lora_rank: int, lora_alpha: float):
    """Update lora_ranks and lora_scaling for a specific adapter across all LoRA layers.

    Note: This method needs to be called BEFORE any training happens, you should not update
    the config for the same adapter index multiple times throughout training (e.g. it will
    invalidate your current training progress and also violate the assumption that lora_B
    is zero).

    Args:
        model: The model containing LoRA layers
        adapter_index: Index of the adapter to update
        lora_rank: Rank to set for this adapter
        lora_alpha: Alpha value to use for computing scaling (alpha / rank)
    """
    scaling = lora_alpha / lora_rank
    state = nnx.state(model)

    def update_lora_config(path, value):
        if path[-2].key == "lora_ranks":
            return value.at[adapter_index].set(lora_rank)
        if path[-2].key == "lora_scaling":
            return value.at[adapter_index].set(scaling)
        if path[-2].key == "lora_A":
            # Zero out columns beyond the rank for this adapter; lora_B is already zero
            return value.at[adapter_index, :, lora_rank:].set(0.0)
        return value

    updated_state = jax.tree.map_with_path(update_lora_config, state)
    nnx.update(model, updated_state)
