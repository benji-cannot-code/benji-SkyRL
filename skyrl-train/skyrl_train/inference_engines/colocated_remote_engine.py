import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
)
from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine

from typing import List, Optional
import asyncio
import os
import socket
import subprocess
import time
import urllib.request
import urllib.error
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout_seconds: int = 180) -> bool:
    start_time = time.time()
    while True:
        try:
            with socket.socket() as s:
                s.settimeout(1)
                s.connect((host, port))
                return True
        except (socket.timeout, ConnectionRefusedError):
            pass
        if time.time() - start_time >= timeout_seconds:
            return False
        time.sleep(0.5)


@ray.remote
class VLLMHTTPServerActor:
    """Ray actor that spawns a vLLM HTTP server as a subprocess."""

    def __init__(self):
        self.pid: Optional[int] = None
        self.host = ray._private.services.get_node_ip_address().strip("[]")
        self.port: Optional[int] = None

    def start(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: Optional[float] = 0.9,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = True,
        dtype: str = "bfloat16",
        distributed_executor_backend: str = "mp",
        host: Optional[str] = None,
        port: Optional[int] = None,
        quiet: bool = False,
    ) -> str:
        self.port = port or _get_free_port()
        if host:
            self.host = host

        # Build vLLM server CLI
        cmd = [
            "python",
            "-m",
            "skyrl_train.inference_engines.vllm.vllm_server",
            "--model",
            model_path,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            dtype,
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization or 0.9),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--worker-extension-cls",
            "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
        ]

        if enforce_eager:
            cmd.append("--enforce-eager")
        if enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        # For multi-GPU servers, prefer mp backend when launched as a single actor
        # so vLLM uses local CUDA_VISIBLE_DEVICES. For TP=1, this is also fine.
        cmd += ["--distributed-executor-backend", distributed_executor_backend]

        # Enable sleep mode when supported; useful for colocation
        cmd.append("--enable-sleep-mode")

        # Set CUDA_VISIBLE_DEVICES based on Ray's GPU allocation for this actor
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Launch server
        stdout = subprocess.DEVNULL if quiet else None
        stderr = subprocess.DEVNULL if quiet else None
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        self.pid = proc.pid

        # First wait for TCP socket
        healthy = _wait_for_server(self.host, self.port, timeout_seconds=180)
        if not healthy:
            raise RuntimeError(f"vLLM server failed to start at {self.host}:{self.port}")

        # Then wait for HTTP readiness on /v1/models
        start_time = time.time()
        while True:
            try:
                with urllib.request.urlopen(f"http://{self.host}:{self.port}/v1/models", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            if time.time() - start_time >= 180:
                raise RuntimeError(f"vLLM server HTTP not ready at {self.host}:{self.port}")
            time.sleep(0.5)

        return f"{self.host}:{self.port}"

    def kill(self):
        if self.pid is None:
            return
        try:
            os.kill(self.pid, 15)
            time.sleep(2)
        except ProcessLookupError:
            pass


class ColocatedRemoteEngine(RemoteInferenceEngine):
    """RemoteInferenceEngine that retains a reference to its Ray server actor."""

    def __init__(
        self,
        *,
        server_actor,  # ActorHandle
        url: str,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
    ):
        super().__init__(
            url=url,
            model_name=model_name,
            engine_backend="vllm",
            tokenizer=tokenizer,
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
        )
        self._server_actor = server_actor

    async def teardown(self):
        await super().teardown()
        try:
            await self._server_actor.kill.remote()
        except Exception:
            pass


def create_colocated_remote_engines(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerBase, shared_pg=None
) -> List[InferenceEngineInterface]:
    """
    Launch vLLM HTTP servers inside Ray using a colocated placement group and
    return RemoteInferenceEngine clients pointing to them.

    Only supports backend=="vllm".
    """
    assert cfg.generator.backend == "vllm", "Only vLLM backend is supported for colocated HTTP engines."

    num_inference_engines = cfg.generator.num_inference_engines
    tensor_parallel_size = cfg.generator.inference_engine_tensor_parallel_size
    data_parallel_size = cfg.generator.inference_engine_data_parallel_size

    # Prefer shared_pg if provided and TP==1 so we can truly colocate with training.
    use_shared_pg = shared_pg is not None and tensor_parallel_size == 1
    if not use_shared_pg:
        # Create a dedicated placement group for servers; each bundle reserves TP GPUs
        total_servers = num_inference_engines * data_parallel_size
        bundles = [{"GPU": tensor_parallel_size, "CPU": 1} for _ in range(total_servers)]
        pg = placement_group(bundles, strategy="PACK")
        from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout
        from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
    else:
        pg = shared_pg

    # Launch one server per DP replica for each engine index
    server_actors = []
    urls: List[str] = []

    bundle_index = 0
    for i in range(num_inference_engines):
        for dp_rank in range(data_parallel_size):
            sched = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            )
            # If using the shared PG (TP==1), reserve fractional GPU to colocate with trainers.
            # Otherwise, we created bundles sized to TP.
            num_gpus_for_actor = 0.2 if use_shared_pg else tensor_parallel_size
            actor = VLLMHTTPServerActor.options(
                num_cpus=1,
                num_gpus=num_gpus_for_actor,
                scheduling_strategy=sched,
            ).remote()
            server_actors.append(actor)
            bundle_index += 1

    print("Starting servers...")

    # Start servers
    start_refs = []
    for actor_idx, actor in enumerate(server_actors):
        start_refs.append(
            actor.start.remote(
                model_path=cfg.trainer.policy.model.path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
                enforce_eager=cfg.generator.enforce_eager,
                enable_prefix_caching=cfg.generator.enable_prefix_caching,
                dtype=cfg.generator.model_dtype,
                distributed_executor_backend="mp" if tensor_parallel_size > 1 else "uni",
                host=None,
                port=None,
                quiet=False,
            )
        )

    urls = ray.get(start_refs)

    print("Creating inference engines...")

    # Create RemoteInferenceEngine clients with attached server actors to preserve lifetime
    engines: List[InferenceEngineInterface] = []
    for url, actor in zip(urls, server_actors):
        engines.append(
            ColocatedRemoteEngine(
                server_actor=actor,
                url=url,
                model_name=cfg.trainer.policy.model.path,
                tokenizer=tokenizer,
                tp_size=tensor_parallel_size,
                dp_size=data_parallel_size,
                ep_size=cfg.generator.inference_engine_expert_parallel_size,
            )
        )

    # print("Putting engines to sleep")

    return engines

