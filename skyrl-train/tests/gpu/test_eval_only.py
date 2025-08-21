"""
uv run --extra dev --extra vllm --isolated pytest tests/gpu/test_eval_only.py
"""

import json
import pytest
import ray

from omegaconf import DictConfig

from skyrl_gym.envs import register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from skyrl_train.entrypoints.eval_only import EvalPPOExp, TRAIN_METRICS_KEY, EVAL_METRICS_KEY
from skyrl_train.utils.utils import initialize_ray
from tests.gpu.utils import get_test_actor_config


class DummyEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras=None):
        super().__init__()
        self.max_turns = 1

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        return BaseTextEnvStepOutput(
            observations=[],
            reward=0.0,
            done=True,
            metadata={},
        )


register(
    id="dummy_env",
    entry_point="tests.gpu.test_eval_only:DummyEnv",
)


def create_dataset(tmp_path):
    data = {
        "prompt": [{"role": "user", "content": "1+1="}],
        "env_class": "dummy_env",
        "data_source": "test",
    }
    path = tmp_path / "data.json"
    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")
    return str(path)


def test_eval_only(tmp_path):
    cfg = get_test_actor_config()

    # Minimize resources and side effects
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.export_path = str(tmp_path)
    cfg.trainer.ckpt_path = str(tmp_path)
    cfg.trainer.dump_eval_results = False

    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.sampling_params.max_generate_length = 4
    cfg.generator.eval_sampling_params.max_generate_length = 4
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.generator.async_engine = True

    cfg.environment.skyrl_gym.max_env_workers = 1

    initialize_ray(cfg)
    try:
        data_path = create_dataset(tmp_path)
        cfg.data.train_data = [data_path]
        cfg.data.val_data = [data_path]
        cfg.trainer.train_batch_size = 1
        cfg.trainer.eval_batch_size = 1
        cfg.trainer.eval_interval = 1

        exp = EvalPPOExp(cfg)
        metrics = exp.run()

        assert TRAIN_METRICS_KEY in metrics
        assert EVAL_METRICS_KEY in metrics
        assert isinstance(metrics[TRAIN_METRICS_KEY], dict)
        assert isinstance(metrics[EVAL_METRICS_KEY], dict)
    finally:
        ray.shutdown()
