import torch
from tqdm import tqdm
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger

from skyrl_train.utils import Timer

from skyrl_train.eval.context import EvalContext
from skyrl_train.generators.utils import concatenate_generator_outputs, get_metrics_from_generator_output
from skyrl_train.generators.base import (
    GeneratorOutput,
)
from skyrl_train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    prepare_generator_input,
    validate_generator_output,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend


@torch.no_grad()
async def run_eval(context: EvalContext) -> Dict[str, float]:
    """Runs an evaluation

    Args:
        context (EvalContext): context needed for the eval

    Returns:
        Dict[str, float]: evaluation metrics
    """
    cfg = context.cfg
    eval_dataloader = context.eval_dataloader
    tokenizer = context.tokenizer
    global_step = context.global_step
    generator = context.generator

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    sampling_params = cfg.generator.eval_sampling_params
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            cfg.generator.eval_n_samples_per_prompt,
            prompts,
            get_sampling_params_for_backend(cfg.generator.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        validate_generator_output(generator_input, generator_output)
        generator_outputs.append(generator_output)
        concat_all_envs.extend(generator_input["env_classes"])
        concat_env_extras.extend(generator_input["env_extras"])
        concat_uids.extend(uids)
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
    vis = tokenizer.decode(generator_output["response_ids"][0])
    logger.info(f"Eval output example: {vis}")

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        concat_generator_outputs, concat_uids, concat_data_sources, cfg.generator.eval_n_samples_per_prompt
    )

    # 3. Calculate overall metrics across all datasets
    overall_avg_score, overall_pass_at_n = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_avg_score,
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_pass_at_n,
        }
    )

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return eval_metrics
