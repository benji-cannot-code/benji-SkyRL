"""
uv run --extra dev --extra vllm --isolated pytest tests/gpu/gpu_ci/test_skyrl_gym_generator.py
"""

import os
import pytest
import ray
from transformers import AutoTokenizer
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput
from tests.gpu.utils import Timer, get_test_generator_input
from omegaconf import DictConfig, OmegaConf
from skyrl_train.utils.utils import initialize_ray
from skyrl_gym.envs import register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict
from skyrl_train.config.utils import get_default_config
import time

OBSERVATION_PROMPT = "give me another solution"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    default_cfg = get_default_config()
    default_cfg.generator.backend = "vllm"
    return default_cfg


# Setup for formatting tests
class TestEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"{OBSERVATION_PROMPT} {self.turns}"}] if not done else [],
            reward=0,
            done=done,
            metadata={},
        )


register(
    id="test_env",
    entry_point="tests.gpu.gpu_ci.test_skyrl_gym_generator:TestEnv",
)

MODEL_TO_GENERATION_PROMPT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "<|im_start|>assistant\n",
    "unsloth/Llama-3.2-1B-Instruct": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "Qwen/Qwen3-0.6B": "<|im_start|>assistant\n",
}


async def run_generator_end_to_end(
    use_async_engine,
    batched,
    n_samples_per_prompt,
    num_inference_engines,
    tensor_parallel_size,
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_prompt_length=512,
    max_input_length=2048,
    max_generate_length=1024,
    data_path=os.path.expanduser("~/data/gsm8k/validation.parquet"),
    env_class="gsm8k",
    num_prompts=2,
    max_turns=1,
    use_conversation_multi_turn=True,
    max_env_workers=10,
):
    """
    End to end generator test - requires minimum 2 GPUs
    """
    tokenizer = AutoTokenizer.from_pretrained(model)

    # inference_engines = create_ray_wrapped_inference_engines(
    #     num_inference_engines=1,
    #     tensor_parallel_size=1,
    #     model_dtype="bfloat16",
    #     pretrain=model,
    #     seed=42,
    #     vllm_v1_disable_multiproc=True,
    #     enable_prefix_caching=True,
    #     enforce_eager=True,
    #     shared_pg=None,
    #     gpu_memory_utilization=0.8,
    #     inference_engine_enable_sleep=True,
    #     async_engine=use_async_engine,
    #     max_num_batched_tokens=8192,
    #     max_num_seqs=1024,
    #     tokenizer=tokenizer,
    #     sleep_level=1,  # in unit tests that do not explicitly sync weights, we do not discard weights
    # )

    inference_engines = create_remote_inference_engines(
        urls = ["127.0.0.1:8011"],
        model_name=model,
        engine_backend="vllm",
        tokenizer=tokenizer,
        tensor_parallel_size=1,
    )

    # Create a mock generator config
    default_cfg = get_default_config()
    OmegaConf.update(
        default_cfg,
        "generator",
        {
            "sampling_params": {
                "max_generate_length": max_generate_length,
                "logprobs": None,
            },
            "append_eos_token_after_stop_str_in_multi_turn": True,  # for search
            "max_input_length": max_input_length,
            "batched": batched,
            "max_turns": max_turns,
            "zero_reward_on_non_stop": False,
            "use_conversation_multi_turn": use_conversation_multi_turn,
            "apply_overlong_filtering": False,
            "backend": "vllm",
            "enable_http_endpoint": False,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": 8000,
        },
    )

    generator_cfg = default_cfg.generator
    OmegaConf.update(
        default_cfg,
        "environment.skyrl_gym",
        {
            "search": {
                "log_requests": True,
                "search_url": "http://127.0.0.1:8000/retrieve",
            },
            "max_env_workers": max_env_workers,
        },
    )
    env_cfg = default_cfg.environment.skyrl_gym

    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model
    cfg.generator = generator_cfg
    inference_engine_client = InferenceEngineClient(
        inference_engines,
        tokenizer,
        cfg,
    )

    await inference_engine_client.wake_up()

    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=inference_engine_client,
        tokenizer=tokenizer,
        model_name=model,
    )

    input_batch: GeneratorInput = get_test_generator_input(
        model=model,
        num_prompts=num_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        max_prompt_length=max_prompt_length,
        data_path=data_path,
        env_class=env_class,
    )
    # Attach request-time sampling params into the generator input
    input_batch["sampling_params"] = get_sampling_params_for_backend(
        "vllm",
        # DictConfig(
        #     {
        #         "temperature": 1.0,
        #         "top_p": 1.0,
        #         "top_k": -1,
        #         "max_generate_length": max_generate_length,
        #         "min_p": 0.0,
        #         "logprobs": None,
        #         "stop": ["</search>", "</answer>"] if env_class == "search" else None,
        #     }
        # ),
        DictConfig({
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_generate_length": max_generate_length,
            "min_p": 0.0,
            "logprobs": None,
            "stop": ["</search>", "</answer>"] if env_class == "search" else None,
            "seed": 42,
        })
    )

    with Timer(f"generate_responses_async_engine_{use_async_engine}"):
        generator_output = await generator.generate(input_batch)

    print(f"Num tokens gen: {sum(len(x) for x in generator_output["response_ids"])}")

    prompts_out = generator_output["prompt_token_ids"]
    outputs = [
        {
            "response": generator_output["response_ids"][i],
            "loss_mask": generator_output["loss_masks"][i],
        }
        for i in range(len(generator_output["response_ids"]))
    ]

    output_keys = [
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
    ]
    for key in output_keys:
        assert key in generator_output, f"Key {key} not found in generator output"
    assert len(prompts_out) == len(outputs), "Mismatch between prompts and outputs"
    assert isinstance(prompts_out[0], list), "Prompts output should be a list"
    assert isinstance(prompts_out[0][0], int), "Prompts output should be a list of list of token ids"
    assert isinstance(outputs[0]["response"][0], int), "Prompts output should be a list of list of token ids"
    assert len(outputs) == num_prompts * n_samples_per_prompt, "Mismatch between number of outputs and expected outputs"
    for i in range(len(outputs)):
        response_length = len(outputs[i]["response"])
        # TODO (erictang000): make this more precise for multi-turn
        assert response_length <= max_generate_length + max_input_length, f"Output {i} exceeds max length"
        assert response_length == len(outputs[i]["loss_mask"]), f"Output {i} loss mask length mismatch"

    # TODO (tgriggs): Extend this test to compare the outputs to HF generation with temperature 0
    print("All done")
    return generator_output


@pytest.mark.asyncio
async def test_generator_multi_turn_search():
    """
    Test the generator with multiple turns of search
    """
    initialize_ray(get_test_actor_config())
    try:
        s = time.time()
        await run_generator_end_to_end(
            use_async_engine=True,
            batched=False,
            n_samples_per_prompt=5,
            num_inference_engines=1,
            tensor_parallel_size=1,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_prompt_length=2048,
            max_input_length=4096,
            max_generate_length=512,
            data_path=os.path.expanduser("~/data/searchR1/validation.parquet"),
            env_class="search",
            num_prompts=2,
            max_turns=3,
            use_conversation_multi_turn=False,
            max_env_workers=0,
        )
        print(f"Multi Turn Generation: {time.time() - s}")
    finally:
        ray.shutdown()



"""
g5.8xlarge
with vllm cache removed

http:
Total pure inference: 2033.9169204235077
Inference time per agent loop: 203.39169204235077
Inference time per gen call: 70.13506622150027
2025-10-22 20:29:48.690 | INFO     | tests.gpu.gpu_ci.test_skyrl_gym_generator:run_generator_end_to_end:206 - generate_responses_async_engine_True, time cost: 226.21s
Num tokens gen: 3098


engine:
Total pure inference: 3841.5559220314026
Inference time per agent loop: 384.15559220314026
Inference time per gen call: 153.6622368812561
2025-10-22 20:20:23.926 | INFO     | tests.gpu.gpu_ci.test_skyrl_gym_generator:run_generator_end_to_end:206 - generate_responses_async_engine_True, time cost: 451.84s
Num tokens gen: 2775
All done
Multi Turn Generation: 547.9502136707306

"""



"""
g5.8xlarge


http:
Total pure inference: 3112.8866250514984
Inference time per agent loop: 311.28866250514983
Inference time per gen call: 103.76288750171662
2025-10-22 08:04:42.602 | INFO     | tests.gpu.gpu_ci.test_skyrl_gym_generator:run_generator_end_to_end:206 - generate_responses_async_engine_True, time cost: 450.94s
Num tokens gen: 3568
All done
Multi Turn Generation: 490.16574335098267






engine:

Total pure inference: 3886.377058029175
2025-10-20 22:53:22.493 | INFO     | tests.gpu.gpu_ci.test_skyrl_gym_generator:run_generator_end_to_end:206 - generate_responses_async_engine_True, time cost: 451.82s
Num tokens gen: 2772
All done
Multi Turn Generation: 547.4816946983337

Total pure inference: 3212.138909101486
Inference time per agent loop: 321.2138909101486
Inference time per gen call: 128.48555636405945
2025-10-22 07:22:06.208 | INFO     | tests.gpu.gpu_ci.test_skyrl_gym_generator:run_generator_end_to_end:206 - generate_responses_async_engine_True, time cost: 406.89s
Num tokens gen: 2801
All done
Multi Turn Generation: 501.0912811756134
"""

