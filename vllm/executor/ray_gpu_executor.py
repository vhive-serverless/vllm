import asyncio
import os
import pickle
from collections import defaultdict
from itertools import islice, repeat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import vllm.envs as envs
from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.ray_utils import RayWorkerWrapper, ray
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, make_async)

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

USE_RAY_COMPILED_DAG = envs.VLLM_USE_RAY_COMPILED_DAG


class RayGPUExecutor(DistributedGPUExecutor):

    def _init_executor(self) -> None:
        assert self.parallel_config.distributed_executor_backend == "ray"
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.forward_dag = None
        if USE_RAY_COMPILED_DAG:
            self.forward_dag = self._compiled_ray_dag()
            self.extra_execute_model_run_workers_kwargs[
                "use_ray_compiled_dag"] = True

    def _configure_ray_workers_use_nsight(self,
                                          ray_remote_kwargs) -> Dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update({
            "nsight": {
                "t": "cuda,cudnn,cublas",
                "o": "'worker_process_%p'",
                "cuda-graph-trace": "node",
            }
        })

        return ray_remote_kwargs

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        import time
        if self.parallel_config.tensor_parallel_size == 1:
            # For single GPU case, we use a ray worker with constrained memory.
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus = 1

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        # Divide the workers to be active_workers and inactive_workers 
        self.active_workers : List[RayWorkerWrapper] = []
        self.inactive_workers: List[RayWorkerWrapper] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        # Create the workers.
        driver_ip = get_ip()
        rank = 0
        init_workers_start = time.time()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if self.speculative_config is not None:
                worker_module_name = "vllm.spec_decode.spec_decode_worker"
                worker_class_name = "create_spec_worker"
            else:
                worker_module_name = "vllm.worker.worker"
                worker_class_name = "Worker"

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(
                worker_module_name=worker_module_name,
                worker_class_name=worker_class_name,
                trust_remote_code=self.model_config.trust_remote_code,
                rank=rank,
            )

            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
                self.driver_worker = RayWorkerWrapper(
                    worker_module_name=worker_module_name,
                    worker_class_name=worker_class_name,
                    trust_remote_code=self.model_config.trust_remote_code,
                    rank=0,
                )
            else:
                # Else, added to the list of workers.
                self.workers.append(worker)
            rank += 1

        init_workers_latency = time.time() - init_workers_start
        logger.info(f"init workers(model_runner) latency: {init_workers_latency:.1f}s")


        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")
        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True, active_worker=False)

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id])),
            "VLLM_INSTANCE_ID":
            VLLM_INSTANCE_ID,
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
        }, ) for (node_id, _) in worker_node_and_gpu_ids]
        self._run_workers("update_environment_variables",
                          all_args=all_args_to_update_environment_variables, active_worker=False)

        port = get_open_port()
        distributed_init_method = get_distributed_init_method(
            driver_ip, port)

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
            ) for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs, active_worker=False)

        self._run_workers("init_device", active_worker=False)


        all_args_to_init_distributed_group = []
        world_size = 1 + len(self.workers)
        for rank in range(world_size):
            all_args_to_init_distributed_group.append(
                {
                    "rank":rank,
                    "world_size": world_size,
                    "driver_ip": driver_ip,
                    "port": port,
                    "local_rank": node_workers[node_id].index(rank)
                }
            )
        self._run_workers("init_distributed_group_manager", all_kwargs=all_args_to_init_distributed_group, active_worker=False)
        
        self.active_workers = []
        self._update_active_workers()

        # self._run_workers("load_model",
        #                   max_concurrent_workers=self.parallel_config.
        #                   max_parallel_loading_workers,
        #                   active_worker=True,
        #                   )

    def place(self) -> None:
        self.active_workers.append(self.workers[0])
        self.should_restart_parallel_tasks = True
        self._update_active_workers()


    def _update_active_workers(self) -> None:
        active_ranks = [0]
        for w in self.active_workers:
            rank = ray.get(w.execute_method.remote("get_rank"))
            active_ranks.append(rank)


        all_args_to_update_distributed_group = []
        for _ in range(1+len(self.workers)):
            all_args_to_update_distributed_group.append(
                {
                    "active_ranks": active_ranks,
                }
            )

        import time        
        start = time.time()
        self._run_workers("update_distributed_group_manager", all_kwargs=all_args_to_update_distributed_group, active_worker=False)
        create_distributed_group_latency = time.time() - start
        logger.info(f"Create distributed group takes: {create_distributed_group_latency:.1f}s")

        self.parallel_config.tensor_parallel_size = 1 + len(self.active_workers)
        start = time.time()
        self._run_workers("reload_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers,
                          active_worker=True,
                          )
        loading_latency = time.time() - start
        logger.info(f"Load model takes: {loading_latency:.1f}s")

        start = time.time()
        num_gpu_blocks, num_cpu_blocks = (
            self.determine_num_available_blocks())

        profile_run_latency = time.time() - start

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        init_cache_latency = time.time() - start
        logger.info(f"init_cache_latency: {init_cache_latency:.1f}s, where profile run takes {profile_run_latency:.1f}s and init tensor takes: {init_cache_latency - profile_run_latency:.1f}s")



    def _driver_execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_method("execute_model",
                                                 execute_model_req)

    def _run_driver_workers(
        self,
        method: str,
        *args,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        **kwargs,
    ) -> Any:
        driver_args = args if all_args is None else all_args[0]
        driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

        # Start the driver worker after all the ray workers.
        if not use_dummy_driver:
            driver_worker_output = self.driver_worker.execute_method(
                method, *driver_args, **driver_kwargs)
        else:
            assert self.driver_dummy_worker is not None
            driver_worker_output = ray.get(
                self.driver_dummy_worker.execute_method.remote(
                    method, *driver_args, **driver_kwargs))
        return driver_worker_output
        

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_remote_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        active_worker: bool = True,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        - async_run_remote_workers_only: If True the method will be run only
          in the remote workers, not the driver worker. It will also be
          run asynchronously and return a list of futures rather than blocking
          on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        workers = self.active_workers if active_worker else self.workers

        count = len(workers)
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, 1, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, 1, None)

        if use_ray_compiled_dag:
            # Right now, compiled DAG can only accept a single
            # input. TODO(sang): Fix it.
            assert self.forward_dag is not None
            output_channels = self.forward_dag.execute(1)
            ray_worker_outputs = []
        else:
            # Start the ray workers first.
            ray_worker_outputs = [
                worker.execute_method.remote(method, *worker_args,
                                             **worker_kwargs)
                for (worker, worker_args, worker_kwargs
                     ) in zip(workers, all_worker_args, all_worker_kwargs)
            ]

        if async_run_remote_workers_only:
            # Just return futures
            return ray_worker_outputs

        driver_args = args if all_args is None else all_args[0]
        driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

        # Start the driver worker after all the ray workers.
        if not use_dummy_driver:
            driver_worker_output = self.driver_worker.execute_method(
                method, *driver_args, **driver_kwargs)
        else:
            assert self.driver_dummy_worker is not None
            driver_worker_output = ray.get(
                self.driver_dummy_worker.execute_method.remote(
                    method, *driver_args, **driver_kwargs))
        # Get the results of the ray workers.
        if workers and workers != []:
            if use_ray_compiled_dag:
                try:
                    ray_worker_outputs = [
                        pickle.loads(chan.begin_read())
                        for chan in output_channels
                    ]
                finally:
                    # Has to call end_read in order to reuse the DAG.
                    for chan in output_channels:
                        chan.end_read()
            else:
                ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        ray.get(parallel_worker_tasks)

    def _compiled_ray_dag(self):
        import pkg_resources
        required_version = "2.9"
        current_version = pkg_resources.get_distribution("ray").version
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} or greater is "
                             f"required, but found {current_version}")

        from ray.dag import InputNode, MultiOutputNode
        assert self.parallel_config.distributed_executor_backend == "ray"

        # Right now, compiled DAG requires at least 1 arg. We send
        # a dummy value for now. It will be fixed soon.
        with InputNode() as input_data:
            forward_dag = MultiOutputNode([
                worker.execute_model_compiled_dag_remote.
                bind(  # type: ignore[attr-defined]
                    input_data) for worker in self.workers
            ])
        return forward_dag.experimental_compile()

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()

    def _check_if_any_actor_is_dead(self):
        if not self.workers:
            return

        dead_actors = []
        for actor in self.workers:
            actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
            if actor_state["State"] == "DEAD":
                dead_actors.append(actor)
        if dead_actors:
            raise RuntimeError("At least one Worker is dead. "
                               f"Dead Workers: {dead_actors}. ")


class RayGPUExecutorAsync(RayGPUExecutor, DistributedGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_method = make_async(self.driver_worker.execute_method)

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        return await self.driver_exec_method("execute_model",
                                             execute_model_req)

    async def _start_worker_execution_loop(self):
        coros = [
            worker.execute_method.remote("start_worker_execution_loop")
            for worker in self.workers
        ]
        return await asyncio.gather(*coros)
