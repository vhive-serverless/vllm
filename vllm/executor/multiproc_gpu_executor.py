import asyncio
import os
from functools import partial
from typing import Any, List, Optional

from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, make_async)
from vllm.liquid.request import LiquidRequest, LiquidOutput

logger = init_logger(__name__)


class MultiprocessingGPUExecutor(DistributedGPUExecutor):
    """Python multiprocessing-based multi-GPU executor"""

    def _init_executor(self) -> None:
        # Create the parallel GPU workers.
        if self.liquid_config is None:
            world_size = self.parallel_config.tensor_parallel_size
        else:
            world_size = len(self.liquid_config.liquid_gpu_range)

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = (",".join(
                map(str, range(world_size))))

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        from torch.cuda import device_count
        assert world_size <= device_count(), (
            "please set tensor_parallel_size to less than max local gpu count")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        if world_size == 1 and self.liquid_config is None:
            self.workers = []
            self.worker_monitor = None
        else:
            if self.liquid_config is None:
                worker_num = world_size
                non_driver_active = True
            else:
                worker_num = len(self.liquid_config.liquid_gpu_range)
                non_driver_active = False
            result_handler = ResultHandler()
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_worker,
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    ),
                    is_active=non_driver_active,
                    ) for rank in range(1, worker_num)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_worker(
            distributed_init_method=distributed_init_method)
        self._run_workers("init_device")
        if self.liquid_config is not None:
            self._run_workers("update_active_ranks", active_ranks=[0], only_active_workers=False)
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers,
                          only_active_workers=True,
                          )
    def get_active_ranks(self) -> List[int]:
        active_ranks = [0]
        for i, worker in enumerate(self.workers):
            if worker.is_active:
                rank = i+1
                active_ranks.append(rank)
        return active_ranks

    def do_liquid(self, liquid_request: LiquidRequest) -> LiquidOutput:
        shard_ids = liquid_request.shard_ids
        src = liquid_request.src
        dst = liquid_request.dst
        liquid_output = LiquidOutput(shard_ids, src, dst)
        # check if the src is active
        active_ranks = self.get_active_ranks()
        assert dst > 0 and dst < len(self.workers)+1, f"liquid dst: {dst} should be in the range between 1 and {len(self.workers)+1}"
        assert src in active_ranks, f"liquid src: {src} is not active!"
        # check if dst is active, if not, we need to update the distributed group
        if not dst not in active_ranks:
            self.workers[dst-1].is_active = True
            active_ranks = self.get_active_ranks()
            self.update_active_ranks(active_ranks)
        
        # load the shard data in liquid mode
        # self._run_workers("liquid_data", shard_ids=shard_ids, src=src, dst=dst)
         
        return liquid_output

    def update_active_ranks(self,active_ranks: List[int]):
        self._run_workers("update_active_ranks", active_ranks=active_ranks, only_active_workers=False)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

    def _driver_execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(
            execute_model_req=execute_model_req)

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_remote_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        only_active_workers: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_remote_workers_only: If True the method will be run only
                in the remote workers, not the driver worker. It will also be
                run asynchronously and return a list of futures rather than
                blocking on the results.
        """

        if only_active_workers:
            workers = []
            for worker in self.workers:
                if worker.is_active:
                    workers.append(worker)
        else:
            workers = self.workers

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in workers
        ]

        if async_run_remote_workers_only:
            # Just return futures
            return worker_outputs

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and not self.worker_monitor.is_alive(
        ):
            raise RuntimeError("Worker processes are not running")

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()


class MultiprocessingGPUExecutorAsync(MultiprocessingGPUExecutor,
                                      DistributedGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        return await self.driver_exec_model(execute_model_req)

    async def _start_worker_execution_loop(self):
        if self.liquid_config is None:
            workers = self.workers
        else:
            workers = []
            for worker in self.workers:
                if worker.is_active:
                    workers.append(worker)
        coros = [
            worker.execute_method_async("start_worker_execution_loop")
            for worker in workers
        ]
        return await asyncio.gather(*coros)
