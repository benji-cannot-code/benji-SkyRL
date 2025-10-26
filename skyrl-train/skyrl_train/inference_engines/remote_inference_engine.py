import json
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)


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
        server_actor_handle: Optional[Any] = None,
    ):
        """Initialize the InferenceEngine."""
        self.url = f"http://{url}"
        self.model_name = model_name
        self.engine_backend = engine_backend
        self._tp_size = tp_size
        self._dp_size = dp_size
        self._ep_size = ep_size
        self.tokenizer = tokenizer
        self._server_actor_handle = server_actor_handle

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
        if self._server_actor_handle is not None:
            try:
                import ray

                ray.kill(self._server_actor_handle, no_restart=True)
            except Exception:
                pass
            finally:
                self._server_actor_handle = None

    async def _destroy_weights_update_group(self):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/destroy_weights_update_group")
            return await resp.json()


def _parse_host_port(url: str) -> Tuple[str, int]:
    if ":" not in url:
        raise ValueError(
            "Remote inference engine URLs must be of the form 'host:port' when launching colocated servers."
        )
    host, port_str = url.rsplit(":", 1)
    if not port_str:
        raise ValueError(f"Invalid port in remote inference engine URL: {url}")
    return host, int(port_str)


class _RayVLLMHTTPServer:
    def __init__(self, server_config: Dict[str, Any], bundle_indices: Optional[List[int]] = None):
        self._config = server_config
        self._bundle_indices = bundle_indices
        self._thread: Optional[threading.Thread] = None
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._started = False

    def _wait_for_server(self, host: str, port: int, timeout_seconds: int) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return
            except OSError as exc:  # pragma: no cover - simple readiness loop
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(
            f"Timed out waiting for colocated vLLM server to start on {host}:{port}"
        ) from last_error

    def _build_args(self, bind_host: str):
        from vllm.utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args([])

        config = self._config
        args.model = config["model"]
        args.tensor_parallel_size = config["tensor_parallel_size"]
        args.seed = config["seed"]
        args.dtype = config["dtype"]
        args.trust_remote_code = True
        args.enforce_eager = config["enforce_eager"]
        args.enable_prefix_caching = config["enable_prefix_caching"]
        args.enable_sleep_mode = config["enable_sleep_mode"]
        args.worker_extension_cls = config["worker_extension_cls"]
        args.host = bind_host
        args.port = config["port"]
        args.distributed_executor_backend = config["distributed_executor_backend"]

        if config.get("gpu_memory_utilization") is not None:
            args.gpu_memory_utilization = config["gpu_memory_utilization"]
        if config.get("max_num_batched_tokens") is not None:
            args.max_num_batched_tokens = config["max_num_batched_tokens"]
        if config.get("max_num_seqs") is not None:
            args.max_num_seqs = config["max_num_seqs"]
        if config.get("max_logprobs") is not None:
            args.max_logprobs = config["max_logprobs"]

        args.vllm_v1_disable_multiproc = config["vllm_v1_disable_multiproc"]

        for key, value in config.get("engine_init_kwargs", {}).items():
            setattr(args, key, value)

        if config.get("enable_lora"):
            args.enable_lora = True
            if config.get("max_lora_rank") is not None:
                args.max_lora_rank = config["max_lora_rank"]
            if config.get("max_loras") is not None:
                args.max_loras = config["max_loras"]

        if config.get("served_model_name"):
            args.served_model_name = config["served_model_name"]

        validate_parsed_serve_args(args)
        return args

    def start(self) -> str:
        if self._started:
            return self.address()

        import ray
        from skyrl_train.inference_engines.vllm.vllm_engine import setup_envvars_for_vllm
        from skyrl_train.inference_engines.vllm.vllm_server import VllmServer

        bind_host = self._config.get("bind_host") or "0.0.0.0"
        announce_host = self._config.get("announce_host")
        if not announce_host:
            if bind_host in ("0.0.0.0", "::", ""):
                announce_host = ray._private.services.get_node_ip_address().strip("[]")
            else:
                announce_host = bind_host

        args = self._build_args(bind_host)

        env_kwargs = {
            "distributed_executor_backend": args.distributed_executor_backend,
            "noset_visible_devices": self._config["noset_visible_devices"],
            "num_gpus": self._config["num_gpus_per_actor"],
        }
        setup_envvars_for_vllm(env_kwargs, self._bundle_indices)

        if self._config["vllm_v1_disable_multiproc"]:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        server = VllmServer(args)
        thread = threading.Thread(target=server.run_server_uvloop, daemon=True)
        thread.start()

        self._thread = thread
        self._host = announce_host
        self._port = args.port
        self._wait_for_server(announce_host, args.port, self._config["startup_timeout"])
        self._started = True
        return self.address()

    def address(self) -> str:
        if self._host is None or self._port is None:
            raise RuntimeError("Server has not been started yet.")
        return f"{self._host}:{self._port}"

    def stop(self) -> None:  # pragma: no cover - simple shutdown helper
        import ray

        ray.actor.exit_actor()


def launch_colocated_vllm_servers(cfg, colocate_pg) -> Tuple[List[Any], List[str]]:
    """Launch vLLM HTTP servers inside Ray for colocated remote inference."""

    import ray
    from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy

    from skyrl_train.utils import get_all_env_variables, get_ray_pg_ready_with_timeout, ray_noset_visible_devices
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    num_servers = cfg.generator.num_inference_engines
    tensor_parallel_size = cfg.generator.inference_engine_tensor_parallel_size
    data_parallel_size = cfg.generator.inference_engine_data_parallel_size
    expert_parallel_size = cfg.generator.inference_engine_expert_parallel_size

    if data_parallel_size != 1:
        raise NotImplementedError(
            "Colocated remote inference currently supports only data_parallel_size == 1."
        )
    if expert_parallel_size != 1:
        raise NotImplementedError(
            "Colocated remote inference currently supports only expert_parallel_size == 1."
        )

    per_engine_gpu_count = tensor_parallel_size * data_parallel_size

    if colocate_pg is None:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_servers * per_engine_gpu_count)]
        colocate_pg = placement_group(bundles, strategy="PACK")

    get_ray_pg_ready_with_timeout(colocate_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    num_gpus_per_actor = 1 if tensor_parallel_size == 1 else 0

    engine_init_kwargs = {}
    if cfg.generator.engine_init_kwargs:
        engine_init_kwargs = OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True)

    urls = list(cfg.generator.remote_inference_engine_urls)
    if len(urls) != num_servers:
        raise ValueError(
            "Number of remote inference engine URLs must match num_inference_engines when launching colocated servers."
        )

    RayServerActor = ray.remote(_RayVLLMHTTPServer)

    actor_handles: List[Any] = []
    addresses: List[str] = []

    for i in range(num_servers):
        base_pg_index = i * per_engine_gpu_count
        bundle_indices = (
            list(range(base_pg_index, base_pg_index + per_engine_gpu_count))
            if per_engine_gpu_count > 1
            else None
        )

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=colocate_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=base_pg_index,
        )

        host_str, port = _parse_host_port(urls[i])

        bind_host = host_str or "0.0.0.0"
        announce_host = host_str if host_str not in ("", "0.0.0.0", "127.0.0.1") else None
        if announce_host is None:
            bind_host = "0.0.0.0"

        server_config = {
            "model": cfg.trainer.policy.model.path,
            "tensor_parallel_size": tensor_parallel_size,
            "seed": cfg.trainer.seed + i,
            "dtype": cfg.generator.model_dtype,
            "enforce_eager": cfg.generator.enforce_eager,
            "enable_prefix_caching": cfg.generator.enable_prefix_caching,
            "enable_sleep_mode": True,
            "worker_extension_cls": "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
            "bind_host": bind_host,
            "announce_host": announce_host,
            "port": port,
            "distributed_executor_backend": distributed_executor_backend,
            "gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
            "max_num_batched_tokens": cfg.generator.max_num_batched_tokens,
            "max_num_seqs": cfg.generator.max_num_seqs,
            "max_logprobs": 1,
            "vllm_v1_disable_multiproc": cfg.generator.vllm_v1_disable_multiproc,
            "engine_init_kwargs": dict(engine_init_kwargs),
            "noset_visible_devices": noset_visible_devices,
            "num_gpus_per_actor": num_gpus_per_actor,
            "startup_timeout": 180,
        }

        if cfg.trainer.policy.model.lora.rank > 0:
            server_config.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": cfg.trainer.policy.model.lora.rank,
                    "max_loras": 1,
                }
            )
        else:
            server_config["enable_lora"] = False

        actor_handle = RayServerActor.options(
            num_cpus=max(1, num_gpus_per_actor),
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=scheduling_strategy,
        ).remote(server_config, bundle_indices)

        address = ray.get(actor_handle.start.remote())
        actor_handles.append(actor_handle)
        addresses.append(address)

    return actor_handles, addresses


def create_remote_inference_engines(
    urls: List[str],
    model_name: str,
    engine_backend: str,
    tokenizer: PreTrainedTokenizerBase,
    tensor_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
    expert_parallel_size: Optional[int] = None,
    server_actor_handles: Optional[List[Any]] = None,
):
    if server_actor_handles is None:
        server_actor_handles = [None] * len(urls)

    assert len(server_actor_handles) == len(urls), "server_actor_handles must match urls length"

    engines = []
    for url, handle in zip(urls, server_actor_handles):
        engines.append(
            RemoteInferenceEngine(
                url=url,
                model_name=model_name,
                tokenizer=tokenizer,
                engine_backend=engine_backend,
                tp_size=tensor_parallel_size,
                dp_size=data_parallel_size,
                ep_size=expert_parallel_size,
                server_actor_handle=handle,
            )
        )
    return engines
