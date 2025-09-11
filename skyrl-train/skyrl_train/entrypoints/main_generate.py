"""
An entry point to run evaluation only:
bash examples/generation/generate.sh
"""

import asyncio

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray

EVAL_METRICS_KEY = "eval"


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def _run_eval(self, trainer: RayPPOTrainer, dataset):
        trainer.eval_dataset = dataset
        trainer.eval_dataloader = trainer.build_dataloader(dataset, is_train=False)
        return await trainer.eval(eval_only=True)

    def run(self) -> dict[str, dict[str, float]]:
        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg)

        inference_engine_client = InferenceEngineClient(inference_engines)

        async def _run_all_evals():
            await inference_engine_client.wake_up()
            generator = self.get_generator(self.cfg, tokenizer, inference_engine_client)

            trainer = RayPPOTrainer(
                cfg=self.cfg,
                tracker=self.get_tracker(),
                tokenizer=tokenizer,
                train_dataset=None,
                eval_dataset=self.eval_dataset,
                inference_engine_client=inference_engine_client,
                generator=generator,
                colocate_pg=self.colocate_pg,
            )

            metrics = {}
            if self.eval_dataset is not None:
                metrics[EVAL_METRICS_KEY] = await self._run_eval(trainer, self.eval_dataset)

            # Export to wandb if configured
            logger_cfg = self.cfg.trainer.logger
            uses_wandb = (logger_cfg == "wandb") or (isinstance(logger_cfg, list) and "wandb" in logger_cfg)
            if uses_wandb and EVAL_METRICS_KEY in metrics:
                trainer.tracker.log(metrics[EVAL_METRICS_KEY], step=0)

            return metrics

        return asyncio.run(_run_all_evals())


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: DictConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
