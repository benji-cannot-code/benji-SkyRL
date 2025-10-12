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


1 A100 gsm8k vllm (4k min generation token, batched=true):
                            local engine                                        server
e2e                         162                                                 350
inference                   61.35852289199829 (270.6553897857666)               309.8482813835144
inf time in eval            23.31912922859192                                   213.59897661209106
inf time in eval 2          14.16724443435669                                   71.96896409988403
inf engine creation         6.404376745223999                                      0.0002980232238769531
inf engine client           0.00029468536376953125       0.0002295970916748047
inf engine wakeup           54.74844765663147           0.0042307376861572266
create generator            0.008414030075073242         0.014878034591674805



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



(eval_entrypoint pid=24978) total_cookies_price = total_cookies * price_per_cookie
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Total money from cupcakes
(eval_entrypoint pid=24978) total_cupcakes = cupcakes_sold
(eval_entrypoint pid=24978) total_cupcakes_price = total_cupcakes * price_per_cupcake
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Total money
(eval_entrypoint pid=24978) total_money = total_cookies_price + total_cupcakes_price
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Total money after giving $10 each to two sisters
(eval_entrypoint pid=24978) sisters = 2
(eval_entrypoint pid=24978) money_given_each = 10
(eval_entrypoint pid=24978) total_money_after_gift = total_money + sisters * money_given_each
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) total_money_after_gift
(eval_entrypoint pid=24978) ```
(eval_entrypoint pid=24978) ```python
(eval_entrypoint pid=24978) 340.0
(eval_entrypoint pid=24978) ```
(eval_entrypoint pid=24978) The Python code confirms that Suzanne has a total of $340 after giving $10 each to her two sisters. Thus, the final answer is \(\boxed{340}\). This verifies the correctness of our calculations. If you have any more questions or need further assistance, feel free to ask! ### Let's think step by step and output the final answer after "####".
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) 1. **Calculate the total money Suzanne earned from selling cookies:**
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978)    - Suzanne sold 80 cookies at $1 each.
(eval_entrypoint pid=24978)    - Total money from cookies = 80 cookies * $1/cookie = $80.
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) 2. **Calculate the total money Suzanne earned from selling cupcakes:**
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978)    - Suzanne sold 60 cupcakes at $4 each.
(eval_entrypoint pid=24978)    - Total money from cupcakes = 60 cupcakes * $4/cupcake = $240.
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) 3. **Add the money from cookies and cupcakes to get the total earnings:**
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978)    - Total money = Money from cookies + Money from cupcakes
(eval_entrypoint pid=24978)    - Total money = $80 + $240 = $320.
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) 4. **Add the two sisters' $10 each to the total money:**
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978)    - Total money after giving to sisters = $320 + $10 + $10 = $340.
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) Therefore, Suzanne has $340 left from her earnings after giving two sisters $10 each. The final answer is:
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) #### $340
(eval_entrypoint pid=24978) #### is the final answer. Let's verify this with the Python code:
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) ```python
(eval_entrypoint pid=24978) # Initial earnings from cookies
(eval_entrypoint pid=24978) cookies_sold = 80
(eval_entrypoint pid=24978) price_per_cookie = 1
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Initial earnings from cupcakes
(eval_entrypoint pid=24978) cupcakes_sold = 60
(eval_entrypoint pid=24978) price_per_cupcake = 4
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Total money from cookies
(eval_entrypoint pid=24978) total_cookies = cookies_sold
(eval_entrypoint pid=24978) total_cookies_price = total_cookies * price_per_cookie
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Total money from cupcakes
(eval_entrypoint pid=24978) total_cupcakes = cupcakes_sold
(eval_entrypoint pid=24978) total_cupcakes_price = total_cupcakes * price_per_cupcake
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) # Total money
(eval_entrypoint pid=24978) total_money = total_cookies_price + total_cupcakes_price
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978)
(eval_entrypoint pid=24978) 2025-10-06 06:13:49.969 | INFO     | skyrl_train.evaluate:evaluate:125 - Started: 'dump_eval_results'
(eval_entrypoint pid=24978) actual inference: 270.6553897857666
(eval_entrypoint pid=24978) 2025-10-06 06:13:50.283 | INFO     | skyrl_train.utils.trainer_utils:dump_per_dataset_eval_results:297 - Dumped eval data for openai/gsm8k to /root/exports/dumped_evals/eval_only/openai_gsm8k.jsonl
(eval_entrypoint pid=24978) 2025-10-06 06:13:50.284 | INFO     | skyrl_train.utils.trainer_utils:dump_per_dataset_eval_results:304 - Dumped aggregated eval metrics to /root/exports/dumped_evals/eval_only/aggregated_results.jsonl
(eval_entrypoint pid=24978) 2025-10-06 06:13:50.284 | INFO     | skyrl_train.evaluate:evaluate:125 - Finished: 'dump_eval_results', time cost: 0.32s
Evaluation Progress: 100%|██████████| 2/2 [04:30<00:00, 135.32s/it]
(eval_entrypoint pid=24978) wandb: Currently logged in as: benji-xu (benji-xu-uc-berkeley-electrical-engineering-computer-sci) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
(eval_entrypoint pid=24978) wandb: Tracking run with wandb version 0.22.0
(eval_entrypoint pid=24978) wandb: Run data is saved locally in /tmp/ray/session_2025-10-06_05-12-01_904050_9734/runtime_resources/working_dir_files/_ray_pkg_7bbc429b389eb122/wandb/run-20251006_061351-yj2fgyan
(eval_entrypoint pid=24978) wandb: Run `wandb offline` to turn off syncing.
(eval_entrypoint pid=24978) wandb: Syncing run test_run
(eval_entrypoint pid=24978) wandb: ⭐️ View project at https://wandb.ai/benji-xu-uc-berkeley-electrical-engineering-computer-sci/skyrl
(eval_entrypoint pid=24978) wandb: 🚀 View run at https://wandb.ai/benji-xu-uc-berkeley-electrical-engineering-computer-sci/skyrl/runs/yj2fgyan
(eval_entrypoint pid=24978) wandb: Detected [openai] in use.
(eval_entrypoint pid=24978) wandb: Use W&B Weave for improved LLM call tracing. Install Weave with `pip install weave` then add `import weave` to the top of your script.
(eval_entrypoint pid=24978) wandb: For more information, check out the docs at: https://weave-docs.wandb.ai/
(eval_entrypoint pid=24978) wandb: updating run metadata
(eval_entrypoint pid=24978) wandb:
(eval_entrypoint pid=24978) wandb: Run history:
(eval_entrypoint pid=24978) wandb:          eval/all/avg_score ▁
(eval_entrypoint pid=24978) wandb:          eval/all/pass_at_1 ▁
(eval_entrypoint pid=24978) wandb: eval/openai_gsm8k/avg_score ▁
(eval_entrypoint pid=24978) wandb: eval/openai_gsm8k/pass_at_1 ▁
(eval_entrypoint pid=24978) wandb:
(eval_entrypoint pid=24978) wandb: Run summary:
(eval_entrypoint pid=24978) wandb:          eval/all/avg_score 0.06823
(eval_entrypoint pid=24978) wandb:          eval/all/pass_at_1 0.06823
(eval_entrypoint pid=24978) wandb: eval/openai_gsm8k/avg_score 0.06823
(eval_entrypoint pid=24978) wandb: eval/openai_gsm8k/pass_at_1 0.06823
(eval_entrypoint pid=24978) wandb:
(eval_entrypoint pid=24978) wandb: 🚀 View run test_run at: https://wandb.ai/benji-xu-uc-berkeley-electrical-engineering-computer-sci/skyrl/runs/yj2fgyan
(eval_entrypoint pid=24978) wandb: ⭐️ View project at: https://wandb.ai/benji-xu-uc-berkeley-electrical-engineering-computer-sci/skyrl
(eval_entrypoint pid=24978) wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
(eval_entrypoint pid=24978) wandb: Find logs at: ./wandb/run-20251006_061351-yj2fgyan/logs
2025-10-06 06:13:53.177 | INFO     | __main__:main:82 - Metrics from eval only run: {'eval/openai_gsm8k/avg_score': 0.06823351023502654, 'eval/openai_gsm8k/pass_at_1': 0.06823351023502654, 'eval/all/avg_score': 0.06823351023502654, 'eval/all/pass_at_1': 0.06823351023502654}
++ date +%s
+ E2E_END_TS=1759731236
+ E2E_ELAPSED=365
+ echo '[SkyRL] E2E elapsed_seconds=365'
[SkyRL] E2E elapsed_seconds=365
(skyrl) root@9a5a68bedf68:/workspace/benji-SkyRL/skyrl-train#
"""
