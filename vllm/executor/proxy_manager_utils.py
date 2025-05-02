import asyncio
import multiprocessing
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from multiprocessing import Queue, Process
from multiprocessing.synchronize import Lock
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, List, Optional, TextIO,
                    TypeVar, Union)


import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON
from vllm.utils import cuda_is_initialized
from vllm.executor.multiproc_worker_utils import _add_prefix, get_mp_context, Result, ResultFuture, ResultHandler, ProcessWorkerWrapper

if HAS_TRITON:
    from vllm.triton_utils import maybe_set_triton_cache_manager

logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel
_MANAGER_INSTANCE_UUID = "proxy_manager"

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'

JOIN_TIMEOUT_S = 2

@dataclass
class GPUProxyTask:
    instance_uuid: str # src instance uuid
    task_id: uuid.UUID # uuid for task
    method: str # function that is to be executed on the gpu proxy side
    args: List[Any] # arguments passed to the method
    kwargs: Dict[Any, Any]

@dataclass
class GPUProxyResult(Result):
    """Result of task dispatched to worker"""
    instance_uuid: str = ""



class GPUProxyClient:
    def __init__(self, gpu_id: int, task_queue: Queue, result_queue: Queue, lock: Lock, instance_uuid: str):
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self._has_lock = False
        self.instance_uuid = instance_uuid
        self.task_map: Dict[uuid.UUID, ResultFuture] = {}
        self.result_listener = threading.Thread(target=self._result_listener)

    def start(self):
        # Try to acquire the lock
        self.__acquire()
        self.result_listener.start()

    def stop(self):
        self.result_queue.put(None)
        self.__release()

    def __acquire(self):
        logger.info(f"{os.getpid()} try to acquire lock for gpu: {self.gpu_id}...")
        self.lock.acquire()
        self._has_lock = True
        logger.info(f"{os.getpid()} acquired lock for gpu: {self.gpu_id}!")
        # register instance 

    def __release(self):
        if self._has_lock:
            self.lock.release()
            self._has_lock = False
            logger.info(f"{os.getpid()} released lock for gpu: {self.gpu_id}!")

    def execute_method(self, method: str, *args, **kwargs) -> Generic[T]:
        # Execute a function remotely in GPUProxy, this is a sync function
        assert self._has_lock, f"{os.getpid()} haven't acquired the lock for gpu: {self.gpu_id}"
        task_id = uuid.uuid4()
        task = GPUProxyTask(
            instance_uuid=self.instance_uuid,
            task_id=task_id,
            method=method,
            args=args,
            kwargs=kwargs,
        )
        self.task_queue.put(task)
        # Register the task's future
        self.task_map[task_id] = ResultFuture()
        output = self.task_map[task_id].get()
        # Get the result, deregister the future
        self.task_map.pop(task_id) 
        return output

    def _result_listener(self):
        assert self._has_lock, f"{os.getpid()} haven't acquired the lock for gpu: {self.gpu_id}"
        for result in iter(self.result_queue.get, _TERMINATE):
            if result is None:  # Sentinel value to terminate
                logger.info(f"Received shutdown signal for GPUProxyClient: {self.gpu_id}")
                break
            assert isinstance(result, GPUProxyResult), f"Got unexpected result from result queue! Result type:{type(result)}"
            assert result.instance_uuid == self.instance_uuid, f"Got result from instance {result.instance_uuid}, however, current instance's uuid: {self.instance_uuid}"
            assert result.task_id in self.task_map, f"Got unregistered result! Result's task_id: {result.task_id}"
            self.task_map[result.task_id].set_result(result)

        logger.info(f"Exit result listener for GPUProxyClient: {self.gpu_id}")


class GPUProxyManagerClient(ProcessWorkerWrapper): # Used for the proxy manager to send ctrl messages internally
    """Local process wrapper for vllm.worker.Worker,
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, task_queue: Queue, external_result_queue: Queue, manager_result_handler: ResultHandler,
                 worker_factory: Callable[[], Any]) -> None:
        self.mp = get_mp_context()
        self._task_queue = task_queue
        self.external_result_queue = external_result_queue
        self.manager_result_queue = manager_result_handler.result_queue
        self.tasks = manager_result_handler.tasks
        self.process: BaseProcess = self.mp.Process(  # type: ignore[attr-defined]
            target=_run_gpu_proxy_process,
            name="GPUProxyProcess",
            kwargs=dict(
                worker_factory=worker_factory,
                task_queue=self._task_queue,
                manager_result_queue=self.manager_result_queue,
                external_result_queue=self.external_result_queue,
            ),
            daemon=True)

        self.process.start()

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str, args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            task = GPUProxyTask(
                instance_uuid=_MANAGER_INSTANCE_UUID,
                task_id=task_id,
                method=method,
                args=args,
                kwargs=kwargs
            )
            self._task_queue.put(task)
        except SystemExit:
            raise
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e



def _run_gpu_proxy_process(
    worker_factory: Callable[[], Any],
    task_queue: Queue,
    manager_result_queue: Queue,
    external_result_queue: Queue,
) -> None:
    """Worker process event loop"""

    # Add process-specific prefix to stdout and stderr
    process_name = get_mp_context().current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    # Initialize worker
    worker = worker_factory()
    del worker_factory

    # Accept tasks from the engine in task_queue
    # and return task output in result_queue
    logger.info("Worker ready; awaiting tasks")
    try:
        for task in iter(task_queue.get, _TERMINATE):
            assert isinstance(task, GPUProxyTask)
            output = None
            exception = None
            instance_uuid = task.instance_uuid
            task_id = task.task_id
            method = task.method
            args = task.args
            kwargs = task.kwargs
            try:
                executor = getattr(worker, method)
                output = executor(*args, **kwargs)
            except SystemExit:
                raise
            except KeyboardInterrupt:
                break
            except BaseException as e:
                logger.exception(
                    "Exception in worker %s while processing method %s.",
                    process_name, method)
                exception = e
            result = GPUProxyResult(task_id, output, exception, instance_uuid)
            if instance_uuid != _MANAGER_INSTANCE_UUID:
                external_result_queue.put(result)
            else:
                manager_result_queue.put(result)
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Worker failed")

    logger.info("Worker exiting")

