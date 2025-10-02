"""
Main entrypoint for evaluation-only.
"""

import asyncio

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig
from typing import Any

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray
from skyrl_train.evaluate import evaluate
from skyrl_train.utils.trainer_utils import build_dataloader


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def run(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        tokenizer = self.tokenizer

        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        results: dict[str, Any] = await evaluate(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: DictConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    import time
    s = time.time()
    metrics = ray.get(eval_entrypoint.remote(cfg))
    print(f"Generation took: {time.time() - s} seconds")
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()


"""

1 A100 gsm8k:
                    e2e         inference
local engine        164         138.92
local engine        169         138.42


1 A100 gsm8k:
                    e2e         inference
local engine        123
server

"""