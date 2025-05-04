import asyncio
import os
from functools import partial
from typing import Any, List, Optional

from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.gpu_executor import create_worker
from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper, ResultHandler, WorkerMonitor,
    set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (_run_task_with_lock, cuda_device_count_stateless,
                        get_distributed_init_method, get_open_port, make_async,
                        update_environment_variables)

from vllm.executor.proxy_manager_utils import GPUProxyClient

logger = init_logger(__name__)


class LiquidExecutor(DistributedGPUExecutor):
    """Python multiprocessing-based multi-GPU executor"""

    uses_ray: bool = False

    def __init__(self, shared_dict, *args, **kwargs):
        assert "vllm_config" in kwargs
        vllm_config = kwargs['vllm_config']
        self.shared_dict = shared_dict
        self.gpu_ids = vllm_config.liquid_config.gpu_ids
        self.instance_uuid = vllm_config.liquid_config.instance_uuid
        super().__init__(*args, **kwargs)
        # self.gpu_ids = self.vllm_config.liquid_config.gpu_ids
        # self.instance_uuid = self.vllm_config.liquid_config.instance_uuid

    def _init_executor(self) -> None:
        logger.info(f"init liquid executor...")
        # Set up proxy client
        self.proxy_clients: List[GPUProxyClient] = []
        for gpu_id in self.gpu_ids:
            task_queue = self.shared_dict["task_queue_dict"][gpu_id]
            result_queue = self.shared_dict["result_queue_dict"][gpu_id]
            lock = self.shared_dict["lock_dict"][gpu_id]
            proxy_client = GPUProxyClient(gpu_id, task_queue, result_queue, lock, self.instance_uuid)
            proxy_client.start()
            self.proxy_clients.append(proxy_client)

        self._run_workers("load_model")

    def _check_executor_parameters(self):
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            update_environment_variables({
                "CUDA_VISIBLE_DEVICES": (",".join(map(str, range(world_size))))
            })

        cuda_device_count = cuda_device_count_stateless()
        # Use confusing message for more common TP-only case.
        assert tensor_parallel_size <= cuda_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local gpu count ({cuda_device_count})")

        assert world_size <= cuda_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local gpu count ({cuda_device_count})")

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()
        for proxy_client in self.proxy_clients:
            proxy_client.stop()

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(execute_model_req)

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        if async_run_tensor_parallel_workers_only:
            # Run only non-driver workers and just return futures.
            return [
                worker.execute_method(method, *args, **kwargs)
                for worker in self.non_driver_workers
            ]

        # Start all remote workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.proxy_clients
        ]


        # Get the results of the workers.
        return [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Engine is healthy or not is not determined by the liquid executor"""
        pass

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

