import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skyrl_train.evaluate import evaluate
from skyrl_train.generators.base import GeneratorInterface


class DummyEvalDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class DummyResponseLevelGenerator(GeneratorInterface):
    async def generate(self, input_batch):
        num_items = len(input_batch["prompts"])
        outputs = {
            "prompt_token_ids": [],
            "response_ids": [],
            "rewards": [],
            "loss_masks": [],
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
        }

        for i in range(num_items):
            traj = input_batch["trajectory_ids"][i]
            uid = getattr(traj, "instance_id")
            rep = getattr(traj, "repetition_id")

            # Minimal prompt/response tokens
            outputs["prompt_token_ids"].append([101, 102])
            outputs["response_ids"].append([201, 202, 203])
            outputs["loss_masks"].append([1, 1, 1])

            # Response-level rewards by uid and repetition
            if uid == "u1":
                reward = 1.0 if rep == 0 else -1.0
            elif uid == "u2":
                reward = 0.0
            elif uid == "u3":
                reward = -0.5 if rep == 0 else -0.1
            else:
                reward = 0.0
            outputs["rewards"].append(reward)

        return outputs


class DummyTokenLevelGenerator(GeneratorInterface):
    async def generate(self, input_batch):
        num_items = len(input_batch["prompts"])
        outputs = {
            "prompt_token_ids": [],
            "response_ids": [],
            "rewards": [],
            "loss_masks": [],
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
        }

        for i in range(num_items):
            traj = input_batch["trajectory_ids"][i]
            uid = getattr(traj, "instance_id")
            rep = getattr(traj, "repetition_id")

            # Fixed length of 3 tokens per response
            resp = [300 + rep, 301 + rep, 302 + rep]
            outputs["prompt_token_ids"].append([111, 112])
            outputs["response_ids"].append(resp)
            outputs["loss_masks"].append([1, 1, 1])

            # Token-level rewards; last token determines pass@n (>0 => pass)
            if uid == "x":
                # One positive last-token, one negative
                rewards = [0.1, 0.1, 0.1] if rep == 0 else [-0.1, -0.1, -0.1]
            elif uid == "y":
                # Non-positive last-token -> never passes
                rewards = [0.0, 0.0, 0.0]
            else:
                rewards = [0.0, 0.0, -0.1]
            outputs["rewards"].append(rewards)

        return outputs


@pytest.fixture
def dummy_tokenizer():
    class _Tok:
        def decode(self, ids):
            return "decoded"

    return _Tok()


def _make_cfg(export_path: Path | None = None, dump: bool = False, n_samples: int = 2):
    return OmegaConf.create(
        {
            "generator": {
                "eval_sampling_params": {
                    "max_generate_length": 8,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 50,
                    "min_p": 0.0,
                    "logprobs": False,
                    "stop": None,
                },
                "eval_n_samples_per_prompt": n_samples,
                "backend": "vllm",
            },
            "environment": {"env_class": "dummy_env"},
            "trainer": {
                "dump_eval_results": dump,
                "export_path": str(export_path) if export_path is not None else ".",
            },
        }
    )


@pytest.mark.asyncio
async def test_evaluate_response_level_rewards(dummy_tokenizer):
    # Two batches: batch1 has u1,u2 (ds1); batch2 has u3 (ds2)
    batches = [
        [
            {"prompt": [{"role": "user", "content": "a"}], "env_class": "gsm8k", "env_extras": {"data_source": "ds1"}, "uid": "u1"},
            {"prompt": [{"role": "user", "content": "b"}], "env_class": "gsm8k", "env_extras": {"data_source": "ds1"}, "uid": "u2"},
        ],
        [
            {"prompt": [{"role": "user", "content": "c"}], "env_class": "sql", "env_extras": {"data_source": "ds2"}, "uid": "u3"},
        ],
    ]
    dataloader = DummyEvalDataLoader(batches)
    cfg = _make_cfg(dump=False, n_samples=2)

    gen = DummyResponseLevelGenerator()
    metrics = await evaluate(cfg, dataloader, dummy_tokenizer, global_step=100, generator=gen)

    # Per-dataset checks
    assert metrics["eval/ds1/avg_score"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["eval/ds1/pass_at_2"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["eval/ds2/avg_score"] == pytest.approx(-0.3, abs=1e-6)
    assert metrics["eval/ds2/pass_at_2"] == pytest.approx(0.0, abs=1e-6)

    # Overall checks
    assert metrics["eval/all/avg_score"] == pytest.approx(-0.1, abs=1e-6)
    assert metrics["eval/all/pass_at_2"] == pytest.approx(1.0 / 3.0, abs=1e-6)


@pytest.mark.asyncio
async def test_evaluate_token_level_rewards_and_dump(tmp_path: Path, dummy_tokenizer):
    # Single batch, two prompts in one dataset ds3, with token-level rewards
    batches = [
        [
            {"prompt": [{"role": "user", "content": "x"}], "env_class": "gsm8k", "env_extras": {"data_source": "ds3"}, "uid": "x"},
            {"prompt": [{"role": "user", "content": "y"}], "env_class": "gsm8k", "env_extras": {"data_source": "ds3"}, "uid": "y"},
        ]
    ]
    dataloader = DummyEvalDataLoader(batches)
    cfg = _make_cfg(export_path=tmp_path, dump=True, n_samples=2)

    gen = DummyTokenLevelGenerator()
    metrics = await evaluate(cfg, dataloader, dummy_tokenizer, global_step=123, generator=gen)

    # Per-dataset and overall metrics
    assert metrics["eval/ds3/avg_score"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["eval/ds3/pass_at_2"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["eval/all/avg_score"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["eval/all/pass_at_2"] == pytest.approx(0.5, abs=1e-6)

    # Dumped files exist
    dump_dir = tmp_path / "dumped_evals" / "global_step_123_evals"
    assert dump_dir.exists()
    agg = dump_dir / "aggregated_results.jsonl"
    ds_file = dump_dir / "ds3.jsonl"
    assert agg.exists() and ds_file.exists()

    # Aggregated file has metrics
    with open(agg, "r") as f:
        line = f.readline().strip()
    data = json.loads(line)
    assert "eval/ds3/avg_score" in data
    assert data["eval/ds3/pass_at_2"] == pytest.approx(0.5, abs=1e-6)

