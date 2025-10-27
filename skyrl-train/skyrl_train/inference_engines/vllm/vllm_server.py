import os
import signal
import socket
import threading
import time
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

try:
    import ray
except ModuleNotFoundError:  # pragma: no cover - ray is an optional dependency in some contexts
    ray = None

from loguru import logger

from skyrl_train.inference_engines.vllm.vllm_engine import setup_envvars_for_vllm


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

        if threading.current_thread() is threading.main_thread():
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


def _create_server_args_from_kwargs(**kwargs):
    """Create a vLLM server argparse.Namespace from keyword arguments."""

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])

    for key, value in kwargs.items():
        setattr(args, key, value)

    validate_parsed_serve_args(args)
    return args


class VllmServerRayActor:
    """A small Ray-friendly wrapper that spins up a vLLM HTTP server."""

    def __init__(
        self,
        server_kwargs: dict,
        readiness_timeout_s: int = 120,
    ) -> None:
        # Create a copy because we mutate kwargs when configuring the environment.
        server_kwargs = dict(server_kwargs)
        bundle_indices = server_kwargs.pop("bundle_indices", None)
        noset_visible_devices = server_kwargs.pop("noset_visible_devices", False)
        num_gpus = server_kwargs.pop("num_gpus", 1)

        env_kwargs = {
            "noset_visible_devices": noset_visible_devices,
            "num_gpus": num_gpus,
        }
        setup_envvars_for_vllm(env_kwargs, bundle_indices)

        self.server_args = _create_server_args_from_kwargs(**server_kwargs)
        self._server = VllmServer(self.server_args)
        self._thread = threading.Thread(target=self._server.run_server_uvloop, daemon=True)
        self._thread.start()

        host = self.server_args.host or "127.0.0.1"
        port = self.server_args.port
        self._wait_until_ready(host, port, readiness_timeout_s)
        self._url = f"{host}:{port}"
        logger.info(f"vLLM HTTP server started at {self._url}")

    def _wait_until_ready(self, host: str, port: int, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for vLLM server to start on {host}:{port}")

    def ready(self) -> str:
        """Return the URL where the server is reachable."""

        return self._url

    def shutdown(self) -> None:
        """Placeholder shutdown hook. Server exits when actor exits."""

        # The HTTP server will be terminated when the Ray actor process exits. This
        # method exists to provide a consistent API if explicit shutdown is ever
        # needed in the future.
        return None


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    vllm_server = VllmServer(args)
    vllm_server.run_server_uvloop()
