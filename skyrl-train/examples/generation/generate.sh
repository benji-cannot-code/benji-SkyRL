# Launches a sample generation-only run

uv run --isolated --extra vllm \
    -m skyrl_train.entrypoints.main_generate \
    trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    generator.backend=vllm \
    generator.num_inference_engines=1 \
    generator.inference_engine_tensor_parallel_size=1 \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.critic_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    trainer.placement.reward_num_gpus_per_node=1 \
    environment.env_class=gsm8k
