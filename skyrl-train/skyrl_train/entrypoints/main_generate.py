"""
Main entrypoint for evaluation-only.
"""

import asyncio
import time

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

        s = time.time()
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)
        print(f"create inf engine time: {time.time() - s}")

        s = time.time()
        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        print(f"create inf client time: {time.time() - s}")

        s = time.time()
        await inference_engine_client.wake_up()
        print(f"inf engine wake up time: {time.time() - s}")

        s = time.time()
        generator = self.get_generator(self.cfg, tokenizer, inference_engine_client)
        print(f"create generator time: {time.time() - s}")

        eval_start = time.time()
        results: dict[str, Any] = await evaluate(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )
        print(f"actual inference: {time.time() - eval_start}")

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
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()


"""


1 A100 gsm8k vllm:
                            local engine                server
e2e                         149                         88
inference                   43.68907284736633           48.253
inf time in eval            13.193318128585815          17.809234380722046
inf time in eval 2          5.665455341339111           4.998567581176758
inf engine creation         6.692424535751343           0.00018310546875
inf engine client           0.0002460479736328125       0.00016236305236816406
inf engine wakeup           57.64691233634949           0.004301548004150391
create generator            0.01616978645324707         0.012064456939697266




"""
