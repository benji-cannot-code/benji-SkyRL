import aiohttp
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

import ray
from ray.exceptions import RayActorError
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.utils import get_all_env_variables, get_ray_pg_ready_with_timeout, ray_noset_visible_devices
from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

if TYPE_CHECKING:
    from ray.actor import ActorHandle


class RemoteInferenceEngine(InferenceEngineInterface):
    """
    Lightweight client to call into an OpenAI-compatible server over HTTP with a customizable backend.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        engine_backend: str,
        tokenizer: PreTrainedTokenizerBase,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        server_actor: Optional["ActorHandle"] = None,
    ):
        """Initialize the InferenceEngine."""
        self.url = f"http://{url}"
        self.model_name = model_name
        self.engine_backend = engine_backend
        self._tp_size = tp_size
        self._dp_size = dp_size
        self._ep_size = ep_size
        self.tokenizer = tokenizer
        self._server_actor = server_actor

    def tp_size(self) -> int:
        return self._tp_size

    def dp_size(self) -> int:
        return self._dp_size

    def ep_size(self) -> int:
        return self._ep_size

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        # 1. Prepare inputs
        prompts = input_batch.get("prompts")
        prompt_token_ids: Optional[List[List[int]]] = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "RemoteInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = request_sampling_params if request_sampling_params is not None else {}
        if "n" in sampling_params and sampling_params["n"] > 1:
            raise ValueError(
                "n is not supported yet for remote inference engines. "
                "You can set `config.generator.n_samples_per_prompt` instead."
            )

        # 2. Send a batched request to the server
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            headers = {"Content-Type": "application/json"}
            payload = {}
            request_url = ""
            if self.engine_backend == "vllm":
                # vLLM does not support /generate, use /completions instead. It supports batch generation.
                payload = sampling_params.copy()
                payload["model"] = self.model_name
                payload["prompt"] = prompt_token_ids
                request_url = f"{self.url}/v1/completions"
            elif self.engine_backend == "sglang":
                # SGLang supports /generate, works exactly like its Python `async_generate()` method
                # and can do batch generation.
                payload = {
                    "input_ids": prompt_token_ids,
                    "sampling_params": sampling_params,
                }
                request_url = f"{self.url}/generate"
            else:
                raise ValueError(f"Invalid engine backend: {self.engine_backend}")
            async with session.post(request_url, json=payload, headers=headers) as resp:
                response = await resp.json()

        # 3. Parse outputs
        outputs = []
        output_ids = []
        finish_reasons = []

        if self.engine_backend == "vllm":
            for i, choice in enumerate(response.get("choices", [])):
                # Since n=1, index i represents the output for `prompt[i]`
                assert choice["index"] == i, "Expect the choices to be ordered by index."
                text = choice["text"]
                outputs.append(text)
                finish_reasons.append(choice["finish_reason"])
                # TODO(Charlie): this is not token-in-token-out because vLLM does not support
                # returning token IDs via HTTP requests. Fix after this vLLM PR is merged:
                # https://github.com/vllm-project/vllm/pull/22587
                output_ids.append(self.tokenizer.encode(text, add_special_tokens=False))
        elif self.engine_backend == "sglang":
            # since prompt_token_ids is a list of lists, response is a list of dicts
            for output in response:
                cur_output_ids = output["output_ids"]
                output_ids.append(cur_output_ids)
                # SGLang only returns tokens not text when skip_tokenizer_init is True, so
                # we manually decode it.
                outputs.append(self.tokenizer.decode(cur_output_ids, skip_special_tokens=True))
                finish_reasons.append(output["meta_info"]["finish_reason"]["type"])
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        return InferenceEngineOutput(
            responses=outputs, stop_reasons=finish_reasons, response_ids=output_ids, response_logprobs=None
        )

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json", {})
        # NOTE(Charlie): cannot reuse payload["headers"] since we are posting a new request.
        # Otherwise will lead to json decode error.
        headers = {"Content-Type": "application/json"}
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            request_url = f"{self.url}/v1/chat/completions"
            async with session.post(request_url, json=body, headers=headers) as resp:
                response = await resp.json()

        return response

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json", {})
        headers = {"Content-Type": "application/json"}
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            request_url = f"{self.url}/v1/completions"
            async with session.post(request_url, json=body, headers=headers) as resp:
                response = await resp.json()

        return response

    async def wake_up(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/wake_up", json={"tags": kwargs.get("tags", 1)})
            return await resp.json()

    async def sleep(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            # TODO(Charlie): this is vLLM's API, not SGLang (which uses tags). Fix when need to
            # support sleeping with remote engines.
            resp = await session.post(f"{self.url}/sleep", json={"level": kwargs.get("level", 1)})
            return await resp.json()

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """
        Initialize the distributed process group for syncing weights.
        """

        path = "/init_weights_update_group" if self.engine_backend == "sglang" else "/init_weight_update_communicator"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.url}{path}",
                json={
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": backend,
                    "override_existing": override_existing,
                },
            ) as response:
                return await response.json()

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        assert (
            len(request["names"]) == 1
        ), f"Remote inference engines support only requests with a single named weight at a time , got request with {len(request['names'])} entries"

        if request.get("extras") and "ipc_handles" in request["extras"][0]:
            raise ValueError(
                "Remote inference engines do not support CUDA IPC weight updates. Only local engines support IPC."
            )
        if self.engine_backend == "vllm":
            weight_update_method = "update_weights"
        elif self.engine_backend == "sglang":
            weight_update_method = "update_weights_from_distributed"
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        async with aiohttp.ClientSession() as session:
            name = request["names"][0]
            dtype = request["dtypes"][0]
            shape = request["shapes"][0]

            resp = await session.post(
                f"{self.url}/{weight_update_method}",
                json={
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                },
            )
            return await resp.json()

    # TODO(tgriggs): Come up with a (more) elegant way to handle text or json responses, and test it and handle errors.
    async def reset_prefix_cache(self):
        if self.engine_backend == "vllm":
            reset_prefix_cache_method = "reset_prefix_cache"
        elif self.engine_backend == "sglang":
            reset_prefix_cache_method = "flush_cache"
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/{reset_prefix_cache_method}")
            text = await resp.text()

        # First try to parse it as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If invalid JSON, return raw text plus status
            return {
                "status": resp.status,
                "body": text,
            }

    async def teardown(self):
        await self._destroy_weights_update_group()
        if self._server_actor is not None:
            # Ray awaits are supported in asyncio contexts
            try:
                await self._server_actor.shutdown.remote()
            except RayActorError:
                pass
            finally:
                self._server_actor = None

    async def _destroy_weights_update_group(self):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/destroy_weights_update_group")
            return await resp.json()


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Invalid inference engine URL: {url}")
    return parsed.hostname, parsed.port


def create_remote_inference_engines(
    urls: List[str],
    model_name: str,
    engine_backend: str,
    tokenizer: PreTrainedTokenizerBase,
    tensor_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
    expert_parallel_size: Optional[int] = None,
    *,
    colocate_all: bool = False,
    shared_pg=None,
    ray_server_config: Optional[Dict[str, Any]] = None,
):
    if not colocate_all:
        return [
            RemoteInferenceEngine(
                url=url,
                model_name=model_name,
                tokenizer=tokenizer,
                engine_backend=engine_backend,
                tp_size=tensor_parallel_size,
                dp_size=data_parallel_size,
                ep_size=expert_parallel_size,
            )
            for url in urls
        ]

    if engine_backend != "vllm":
        raise NotImplementedError(
            "colocate_all=True is currently supported only for vLLM remote inference engines"
        )

    if ray_server_config is None:
        raise ValueError("ray_server_config must be provided when colocate_all=True for remote engines")

    from skyrl_train.inference_engines.vllm.vllm_server import VLLMHTTPServerRayActor

    num_inference_engines = len(urls)
    tp_size = tensor_parallel_size or 1
    dp_size = data_parallel_size or 1
    per_engine_gpu_count = tp_size * dp_size

    if shared_pg is None:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * per_engine_gpu_count)]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    distributed_executor_backend = "uni" if tp_size == 1 else "ray"
    num_gpus_per_actor = int(tp_size == 1)

    engines: List[RemoteInferenceEngine] = []

    for engine_idx, url in enumerate(urls):
        host, port = _parse_host_port(url)
        base_pg_index = engine_idx * per_engine_gpu_count
        bundle_indices = list(range(base_pg_index, base_pg_index + per_engine_gpu_count)) if per_engine_gpu_count > 1 else None

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_bundle_index=base_pg_index,
            placement_group_capture_child_tasks=True,
        )

        server_config = {
            **ray_server_config,
            "model": model_name,
            "host": host,
            "port": port,
            "tensor_parallel_size": tp_size,
            "dtype": ray_server_config.get("dtype", "auto"),
            "distributed_executor_backend": distributed_executor_backend,
            "uvicorn_log_level": ray_server_config.get("uvicorn_log_level", "warning"),
            "worker_extension_cls": "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
            "trust_remote_code": True,
        }

        if "dtype" not in ray_server_config and ray_server_config.get("model_dtype"):
            server_config["dtype"] = ray_server_config["model_dtype"]

        server_actor = VLLMHTTPServerRayActor.options(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=scheduling_strategy,
        ).remote(
            server_config=server_config,
            bundle_indices=bundle_indices,
            noset_visible_devices=noset_visible_devices,
        )

        # Wait for the server to become ready before returning the client.
        ray.get(server_actor.wait_until_ready.remote(timeout_s=120))

        engines.append(
            RemoteInferenceEngine(
                url=f"{host}:{port}",
                model_name=model_name,
                tokenizer=tokenizer,
                engine_backend=engine_backend,
                tp_size=tensor_parallel_size,
                dp_size=data_parallel_size,
                ep_size=expert_parallel_size,
                server_actor=server_actor,
            )
        )

    return engines
