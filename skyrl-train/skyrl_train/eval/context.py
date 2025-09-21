from dataclasses import dataclass
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from skyrl_train.generators.base import GeneratorInterface


@dataclass
class EvalContext:
    cfg: DictConfig
    eval_dataloader: StatefulDataLoader | None
    tokenizer: AutoTokenizer
    global_step: int | None
    generator: GeneratorInterface
