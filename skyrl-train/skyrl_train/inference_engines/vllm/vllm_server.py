import os
import signal
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import uvloop
from vllm import AsyncLLMEngine
from vllm.utils import FlexibleArgumentParser, set_ulimit
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    create_server_socket,
    build_app,
    init_app_state,
)
import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from fastapi import Request
from skyrl_train.inference_engines.vllm.vllm_engine import setup_envvars_for_vllm
from skyrl_train.inference_engines.utils import get_rendezvous_addr_port
from skyrl_train.utils import get_all_env_variables, ray_noset_visible_devices

try:
    import ray
    from ray.util.placement_group import PlacementGroup, PlacementGroupSchedulingStrategy
except ImportError:  # pragma: no cover - ray is optional when running the standalone server
    ray = None
    PlacementGroup = None
    PlacementGroupSchedulingStrategy = None

from omegaconf import OmegaConf


# TODO(tgriggs): Handle errors and use best practices for vLLM server
# TODO(tgriggs): Return correct status codes.
class VllmServer:
    def __init__(self, args):
        self.server_args = args

    async def run_server(self, **uvicorn_kwargs) -> None:
        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)

        set_ulimit()

        def signal_handler(*_) -> None:
            # Interrupt server on sigterm while initializing
            raise KeyboardInterrupt("terminated")

        signal.signal(signal.SIGTERM, signal_handler)

        # TODO(tgriggs): Move this elsewhere, make configurable.
        os.environ["VLLM_USE_V1"] = "1"
        engine_args = AsyncEngineArgs.from_cli_args(self.server_args)
        engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)
        app = build_app(self.server_args)

        @app.post("/init_weight_update_communicator")
        async def _init_weight_update_communicator(request: Request):
            data = await request.json()
            master_addr = data.get("master_address")
            master_port = data.get("master_port")
            world_size = data.get("world_size")
            backend = data.get("backend")
            group_name = data.get("group_name")
            rank_offset = data.get("rank_offset")
            override_existing = data.get("override_existing", False)

            await engine.collective_rpc(
                "init_weight_update_communicator",
                args=(
                    master_addr,
                    master_port,
                    rank_offset,
                    world_size,
                    group_name,
                    backend,
                    override_existing,
                ),
            )
            return {"status": "ok"}

        @app.post("/sleep")
        async def _sleep(request: Request):
            data = await request.json()
            level = data.get("level")

            # TODO(team): remove once vllm fixes this
            # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
            await engine.reset_prefix_cache()

            await engine.sleep(level)
            return {"status": "ok"}

        @app.post("/wake_up")
        async def _wake_up(request: Request):
            data = await request.json()
            tags = data.get("tags")
            await engine.wake_up(tags)
            return {"status": "ok"}

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            await engine.reset_prefix_cache()
            return {"status": "ok"}

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            data = await request.json()
            # engine expects a list of objects
            names = [data.get("name")]
            dtypes = [data.get("dtype")]
            shapes = [data.get("shape")]
            await engine.collective_rpc(
                "update_weights",
                args=(names, dtypes, shapes),
            )
            return {"status": "ok"}

        @app.post("/destroy_weights_update_group")
        async def _destroy_weights_update_group(request: Request):
            data = await request.json()  # noqa: F841
            await engine.collective_rpc(
                "destroy_weights_update_group",
                args=(),
            )
            return {"status": "ok"}

        vllm_config = await engine.get_vllm_config()
        await init_app_state(engine, vllm_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            sock,
            host=self.server_args.host,
            port=self.server_args.port,
            log_level=self.server_args.uvicorn_log_level,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self.server_args.ssl_keyfile,
            ssl_certfile=self.server_args.ssl_certfile,
            ssl_ca_certs=self.server_args.ssl_ca_certs,
            ssl_cert_reqs=self.server_args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

        await shutdown_task

        sock.close()

    def run_server_uvloop(self, **uvicorn_kwargs) -> None:
        uvloop.run(self.run_server(**uvicorn_kwargs))


def _parse_host_port(url: str) -> Tuple[str, int]:
    if ":" not in url:
        raise ValueError(f"Invalid inference server url '{url}'. Expected format <host>:<port>.")
    host, port_str = url.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(f"Invalid port '{port_str}' in inference server url '{url}'.") from exc
    return host, port


def _to_python_dict(config_obj: Any) -> Dict[str, Any]:
    if isinstance(config_obj, dict):
        return config_obj
    return OmegaConf.to_container(config_obj, resolve=True) if config_obj is not None else {}


def _list_from_config(config_obj: Any) -> List[Any]:
    if isinstance(config_obj, list):
        return config_obj
    return list(config_obj) if config_obj is not None else []


if ray is not None:

    @ray.remote
    class ColocatedVLLMHTTPServerActor:
        """Ray actor that launches a vLLM OpenAI server inside the Ray cluster.

        The actor spins up the HTTP server in a background thread so that Ray can
        continue interacting with it (e.g. to wait for readiness). It mirrors the
        environment configuration logic used by the Ray wrapped inference engines to
        ensure that GPU resources and placement groups are respected when
        ``trainer.placement.colocate_all`` is enabled.
        """

        def __init__(
            self,
            server_kwargs: Dict[str, Any],
            engine_kwargs: Dict[str, Any],
            engine_init_kwargs: Optional[Dict[str, Any]] = None,
            bundle_indices: Optional[Sequence[int]] = None,
            noset_visible_devices: bool = False,
        ):
            self._engine_kwargs = dict(engine_kwargs)
            self._engine_init_kwargs = dict(engine_init_kwargs or {})
            self._bundle_indices = list(bundle_indices) if bundle_indices is not None else None
            self._noset_visible_devices = noset_visible_devices
            self._num_gpus = float(server_kwargs.get("num_gpus", 1))

            host = server_kwargs.get("host") or "0.0.0.0"
            self._bind_host = host
            if server_kwargs.get("port") is None:
                self._port = self._find_free_port()
            else:
                port = int(server_kwargs["port"])
                self._port = port if port > 0 else self._find_free_port()

            self._uvicorn_log_level = server_kwargs.get("uvicorn_log_level", "warning")
            self._server_exception: Optional[BaseException] = None
            self._server: Optional[VllmServer] = None

            advertised_host = server_kwargs.get("advertised_host")
            if not advertised_host:
                advertised_host = self._resolve_advertised_host(host)
            self._advertised_host = advertised_host

            self._thread = threading.Thread(target=self._run_server, daemon=True)
            self._thread.start()

        @staticmethod
        def _find_free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                return sock.getsockname()[1]

        @staticmethod
        def _resolve_advertised_host(bind_host: str) -> str:
            if bind_host and bind_host not in {"0.0.0.0", "127.0.0.1", ""}:
                return bind_host
            if ray is not None:
                try:
                    return ray._private.services.get_node_ip_address().strip("[]")
                except Exception:  # pragma: no cover - fallback if Ray internals change
                    pass
            return "127.0.0.1"

        def _build_args_namespace(self, engine_args_kwargs: Dict[str, Any]):
            parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
            parser = make_arg_parser(parser)
            try:
                args = parser.parse_args([])
            except SystemExit as exc:  # pragma: no cover - defensive
                raise RuntimeError("Failed to construct default vLLM server arguments") from exc

            for key, value in engine_args_kwargs.items():
                setattr(args, key, value)

            args.host = self._bind_host
            args.port = self._port
            args.uvicorn_log_level = self._uvicorn_log_level
            return args

        def _run_server(self) -> None:
            try:
                # Configure environment similar to the Ray wrapped inference engines.
                env_setup_kwargs = dict(self._engine_kwargs)
                env_setup_kwargs.update(self._engine_init_kwargs)
                env_setup_kwargs["noset_visible_devices"] = self._noset_visible_devices
                env_setup_kwargs["num_gpus"] = self._num_gpus
                setup_envvars_for_vllm(env_setup_kwargs, self._bundle_indices)

                if not self._noset_visible_devices and ray is not None:
                    gpu_ids = ray.get_gpu_ids()
                    if gpu_ids:
                        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(g)) for g in gpu_ids)

                if self._engine_kwargs.pop("vllm_v1_disable_multiproc", False):
                    os.environ["VLLM_V1_DISABLE_MULTIPROC"] = "1"

                engine_args_kwargs = dict(self._engine_kwargs)
                engine_args_kwargs.update(self._engine_init_kwargs)
                engine_args_kwargs.setdefault("trust_remote_code", True)
                engine_args_kwargs.setdefault("worker_extension_cls", "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap")

                args = self._build_args_namespace(engine_args_kwargs)
                validate_parsed_serve_args(args)

                self._server = VllmServer(args)
                self._server.run_server_uvloop()
            except BaseException as exc:  # pragma: no cover - server startup errors
                self._server_exception = exc
                raise

        def wait_ready(self, timeout_s: int = 180) -> str:
            deadline = time.time() + timeout_s
            url = f"http://{self._advertised_host}:{self._port}/health"

            while time.time() < deadline:
                if self._server_exception is not None:
                    raise RuntimeError("vLLM HTTP server failed to start") from self._server_exception

                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        return f"{self._advertised_host}:{self._port}"
                except requests.exceptions.RequestException:
                    pass

                time.sleep(1)

            raise TimeoutError(f"Timed out waiting for vLLM HTTP server at {url} to become ready")

        def get_address(self) -> str:
            return f"{self._advertised_host}:{self._port}"

else:  # pragma: no cover - used when Ray is not installed

    class ColocatedVLLMHTTPServerActor:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise RuntimeError("Ray must be installed to launch colocated HTTP inference servers.")


def launch_colocated_vllm_http_servers(cfg, colocate_pg: PlacementGroup) -> Tuple[List[str], List[Any]]:
    """Launch vLLM HTTP servers inside Ray when ``colocate_all`` is enabled.

    Returns a tuple of ``(urls, actor_handles)``. The URLs correspond to the
    addresses that should be used by ``RemoteInferenceEngine`` clients.
    """

    if ray is None:
        raise RuntimeError("Ray must be installed to launch colocated HTTP inference servers.")

    if PlacementGroupSchedulingStrategy is None:
        raise RuntimeError("Ray placement group utilities are required to launch colocated HTTP servers.")

    if colocate_pg is None:
        raise ValueError("A placement group must be provided when launching colocated HTTP inference servers.")

    remote_urls = _list_from_config(cfg.generator.remote_inference_engine_urls)
    num_inference_engines = cfg.generator.num_inference_engines
    tensor_parallel_size = cfg.generator.inference_engine_tensor_parallel_size
    data_parallel_size = cfg.generator.inference_engine_data_parallel_size
    expert_parallel_size = cfg.generator.inference_engine_expert_parallel_size
    per_engine_gpu_count = tensor_parallel_size * data_parallel_size

    expected_urls = num_inference_engines * data_parallel_size
    if len(remote_urls) < expected_urls:
        raise ValueError(
            "generator.remote_inference_engine_urls must provide an entry for each inference engine replica when using "
            "colocate_all with HTTP inference."
        )

    use_hybrid_engine = colocate_pg is not None
    num_gpus_per_actor = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # Match the behaviour of the Ray wrapped inference engines so that
        # training and inference workers can share GPUs when colocated.
        num_gpus_per_actor = 0.2

    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"

    env_vars = ray.get(get_all_env_variables.remote())
    noset_visible_devices = ray_noset_visible_devices(env_vars)

    engine_init_kwargs = _to_python_dict(cfg.generator.engine_init_kwargs)

    server_actors: List[Any] = []
    final_urls: List[str] = []

    for engine_idx in range(num_inference_engines):
        base_pg_index = engine_idx * per_engine_gpu_count
        data_parallel_address, data_parallel_rpc_port = get_rendezvous_addr_port(colocate_pg, base_pg_index)

        for dp_rank in range(data_parallel_size):
            url_idx = engine_idx * data_parallel_size + dp_rank
            host, port = _parse_host_port(str(remote_urls[url_idx]))

            bundle_indices = (
                list(range(base_pg_index + dp_rank * tensor_parallel_size, base_pg_index + (dp_rank + 1) * tensor_parallel_size))
                if tensor_parallel_size > 1
                else None
            )

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=colocate_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=base_pg_index + dp_rank * tensor_parallel_size,
            )

            engine_kwargs: Dict[str, Any] = {
                "model": cfg.trainer.policy.model.path,
                "seed": cfg.trainer.seed + engine_idx * data_parallel_size + dp_rank,
                "tensor_parallel_size": tensor_parallel_size,
                "dtype": cfg.generator.model_dtype,
                "distributed_executor_backend": distributed_executor_backend,
                "enable_prefix_caching": cfg.generator.enable_prefix_caching,
                "enforce_eager": cfg.generator.enforce_eager,
                "enable_sleep_mode": cfg.trainer.placement.colocate_all,
                "trust_remote_code": True,
                "worker_extension_cls": "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                "gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
                "max_num_batched_tokens": cfg.generator.max_num_batched_tokens,
                "max_num_seqs": cfg.generator.max_num_seqs,
            }

            if cfg.generator.vllm_v1_disable_multiproc:
                engine_kwargs["vllm_v1_disable_multiproc"] = True

            if expert_parallel_size > 1:
                engine_kwargs["enable_expert_parallel"] = True

            if data_parallel_size > 1:
                engine_kwargs.update(
                    {
                        "data_parallel_backend": "mp",
                        "data_parallel_size": data_parallel_size,
                        "data_parallel_rank": dp_rank,
                        "data_parallel_address": data_parallel_address,
                        "data_parallel_rpc_port": data_parallel_rpc_port,
                    }
                )

            server_kwargs = {
                "host": host,
                "port": port,
                "uvicorn_log_level": "warning",
                "num_gpus": num_gpus_per_actor,
            }

            actor = ColocatedVLLMHTTPServerActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                server_kwargs=server_kwargs,
                engine_kwargs=engine_kwargs,
                engine_init_kwargs=engine_init_kwargs,
                bundle_indices=bundle_indices,
                noset_visible_devices=noset_visible_devices,
            )

            server_actors.append(actor)

    ready_urls = ray.get([actor.wait_ready.remote(timeout_s=180) for actor in server_actors])
    final_urls.extend(ready_urls)
    return final_urls, server_actors


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    vllm_server = VllmServer(args)
    vllm_server.run_server_uvloop()
